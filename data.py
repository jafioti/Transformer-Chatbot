import torch.multiprocessing as multiprocessing
import time, os, random
import torch, gc
from io import open
from Railgun import batching, tokenization, vocab
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import namedtuple

worker_flag = namedtuple("worker_flag", "item, worker_num")

class NewDataset:
    def __init__(self, init_function, load_function, chunk_size, preload=False, num_workers=len(os.sched_getaffinity(0)), **kwargs):
        self.chunk_size = chunk_size
        self.loaded_index = 0
        self.post_data = [] if preload else multiprocessing.JoinableQueue()
        self.process = [None] * num_workers
        self.num_workers = num_workers
        self.preload = preload
        self.init_function = init_function
        self.load_function = load_function
        self.kwargs = kwargs

        # Initialize
        if "pre_data" in list(kwargs.keys()) and "other" in list(kwargs.keys()):
            self.pre_data, self.other = kwargs["pre_data"], kwargs["other"]
        else:
            self.pre_data, self.other = init_function(**kwargs)
        self.max = len(self.pre_data[list(self.pre_data.keys())[0]])

        # Preload
        if "post_data" in list(kwargs.keys()) and kwargs["post_data"] is not None:
            self.post_data = kwargs["post_data"]
        elif preload:
            load_job(self.pre_data, self.other, self.post_data, self.load_function, worker_num=None)

    def __iter__(self):
        if not self.preload: self.reset()
        return iter(self.post_data) if self.preload else self

    def __len__(self):
        return len(self.pre_data[list(self.pre_data.keys())[0]])

    def __getitem__(self, i):
        if isinstance(i, slice):
            # Return altered version of self
            return self.__class__(load_function=self.load_function, init_function=self.init_function,
                preload=self.preload, num_workers=self.num_workers, chunk_size=self.chunk_size, post_data=self.post_data[slice(i.start if i.start is not None else None, i.stop if i.stop is not None else None, i.step if i.step is not None else None)] if self.preload else None, 
                pre_data={key:value[i] for (key, value) in self.pre_data.items()}, other=self.other, **self.kwargs)
        elif isinstance(i, int):
            if i + 1 > self.__len__(): raise Exception("Index out of range of the dataset!")
            if self.preload: return self.post_data[i]
            batch = self.load_function({key:value[i:i+self.batch_size] for (key, value) in self.pre_data.items()}, self.other)[0]
            return batch
        else:
            raise Exception("Index is an unknown type! (Not an int or slice)")

    def __next__(self):
        if self.post_data.empty():
            if self.process[0] is None: # We are first starting
                for i in range(self.num_workers):
                    self.start_worker(i)
            elif self.loaded_index >= self.max: # Done iterating
                if not any([self.check_worker(i) for i in range(self.num_workers)]): # Wait until all workers are done
                    raise StopIteration
            else: # Not done loading, wait for some time until queue is not empty
                while self.post_data.empty():
                    time.sleep(0.1)

        item = self.post_data.get(block=True)
        if isinstance(item, worker_flag): # Example contains marker for end of worker
            item, worker_num = item
            # Reset worker worker_num
            self.reset_worker(worker_num)
            return item
        return item
    
    def check_worker(self, worker_num):
        try:
            self.process[worker_num]._check_closed()
            return True
        except Exception: return False

    def reset_worker(self, worker_num):
        # Reset worker
        self.stop_worker(worker_num)
        if self.loaded_index < self.max:
            self.start_worker(worker_num)

    def stop_worker(self, worker_num):
        try:
            self.process[worker_num].terminate()
            while self.process[worker_num].is_alive(): # Ensure that worker is dead
                time.sleep(0.1)
            self.process[worker_num].close()
        except Exception: pass
    
    def start_worker(self, worker_num):
        start = self.loaded_index
        stop = min(self.loaded_index + self.chunk_size, self.max)
        if stop - start > 0:
            self.process[worker_num] = multiprocessing.Process(target=load_job, args=({k:v[start:stop] for k, v in self.pre_data.items()}, self.other, self.post_data, self.load_function, worker_num))
            self.process[worker_num].start()
            self.loaded_index = min(self.loaded_index + self.chunk_size, self.max)

    def shuffle(self):
        # Shuffle data
        if self.preload:
            random.shuffle(self.post_data)
        else:
            lists = batching.shuffle_lists(list(self.pre_data.values()))
            for i, key in enumerate(self.pre_data.keys()):
                self.pre_data[key] = lists[i]
        return self

    def reset(self):
        self.loaded_index = 0
        for i in range(self.num_workers):
            self.stop_worker(i)
        self.process = [None] * self.num_workers

def load_job(pre_data, other, queue, load_function, worker_num):
    examples = load_function(pre_data, other)
    if len(examples) == 0: return
    if isinstance(queue, list):
        queue.extend(examples)
    else: 
        for example in examples[:-1]: queue.put(example)
        if worker_num is None: queue.put(examples[-1])
        else: queue.put(worker_flag(item=examples[-1], worker_num=worker_num)) # Mark example as last from this worker
        queue.join()

# Data loading functions
def init_function(paths, start_index, end_index, max_length, batch_size):
    pre_data, other = {}, {}
    other["local_vocab"] = vocab.load_wordpiece_vocab()
    other["paths"] = paths
    other["max_length"] = max_length
    other["batch_size"] = batch_size

    # Create map of all filenums and linenums
    pre_data["lines"] = []
    for (filenum, path) in enumerate(paths):
        with open(path) as f:
            line = f.readline()
            linenum = 0
            while line:
                pre_data["lines"].append((filenum, linenum))
                line = f.readline()
                linenum += 1
    pre_data["lines"] = pre_data["lines"][start_index:end_index]
    return pre_data, other
    
def load_function(data, other):
    # Data is a dict of lists of pre_data
    # Sort into loading order
    data["lines"] = sorted(data["lines"], key=lambda x: (x[0], x[1]))
    # Load from files
    lines = []
    for (filenum, path) in enumerate(other["paths"][data["lines"][0][0]:data["lines"][-1][0]+1], start=data["lines"][0][0]):
        with open(path) as f:
            line = f.readline()
            linenum = 0
            while line:
                if len(data["lines"]) == len(lines): break
                if data["lines"][len(lines)] == (filenum, linenum) and len(data["lines"]) - 1 > len(lines):
                    lines.append(line)
                line = f.readline()
                linenum += 1

    inputs, targets = [], []
    for line in lines:
        inputs.append(" sep ".join(line.replace("\n", "").split("\t")[:-1]))
        targets.append(line.replace("\n", "").split("\t")[-1])

    del lines
    # Tokenize
    inputs = tokenization.tokenize_wordpiece(inputs)
    targets = tokenization.tokenize_wordpiece(targets)
    
    for i in range(len(inputs) - 1, -1, -1):
        if len(inputs[i]) == 0 or len(targets[i]) == 0:
            del inputs[i], targets[i]
        else:
            targets[i] = targets[i] + ["[EOS]"]

    # Convert to indexes
    inputs = other["local_vocab"].indexes_from_tokens(inputs)
    targets = other["local_vocab"].indexes_from_tokens(targets)

    # Limit the length of examples
    inputs, targets = batching.filter_by_length([inputs, targets], max_length=other["max_length"]-1)
    
    # Sort by length
    inputs, targets = batching.sort_lists_by_length([inputs, targets], True)

    # Convert to batches of tensors
    batches = []
    for i in range(0, len(inputs), other["batch_size"]):
        batches.append(
            (torch.LongTensor(batching.pad_batch(inputs[i:i+other["batch_size"]], other["local_vocab"].PAD_token)).transpose(0, 1),
            torch.LongTensor(batching.pad_batch(targets[i:i+other["batch_size"]], other["local_vocab"].PAD_token)).transpose(0, 1)))
    #del inputs, targets
    gc.collect()
    return batches