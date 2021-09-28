import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from torch.utils.tensorboard import SummaryWriter
import math, gc
from data import load_function, init_function, NewDataset
from Railgun import tokenization, vocab
from SidekickAI.Utilities import utils, losses, metrics
from SidekickAI.Data import Dataset
from model import ConversationalTransformer
from eval import eval_sentence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_vocab = vocab.load_wordpiece_vocab()
if __name__ == "__main__": writer = SummaryWriter()

# Model Hyperparameters
embedding_dim = 300
hidden_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
forward_expansion = 2
dropout = 0.1
max_len = 100

# Training Hyperparameters
learning_rate = 1e-4
batch_size = 32
batch_accumulations = 10
num_epochs = 5
load_model = False
start_epoch = 1
dataset_size = 8000000

def main():
    print("|TRAINING MODEL|")
    print("Loading Data...")
    #train_dataset = Dataset(path=os.path.join(os.path.expanduser('~'), 'Datasets', 'Reddit', 'reddit_conversations.train.txt'), local_vocab=local_vocab, batch_size=batch_size, load_function=load_function, end_index=dataset_size, max_length=max_len, init_function=init_function, collate_function=collate_function, preload=True)
    train_dataset = NewDataset(init_function, load_function, chunk_size=10000, preload=False, start_index=0, end_index=140000, paths=[os.path.join(os.path.expanduser('~'), 'Datasets', 'Reddit', 'utterances1.txt'), os.path.join(os.path.expanduser('~'), 'Datasets', 'Reddit', 'utterances2.txt')], max_length=max_len, batch_size=batch_size)
    test_dataset = NewDataset(init_function, load_function, chunk_size=10000, preload=True, start_index=14000000, end_index=14001000, paths=[os.path.join(os.path.expanduser('~'), 'Datasets', 'Reddit', 'utterances1.txt'), os.path.join(os.path.expanduser('~'), 'Datasets', 'Reddit', 'utterances2.txt')], max_length=max_len, batch_size=batch_size)
    print("Train Dataset Size: " + str(len(train_dataset)))
    print("Test Dataset Size: " + str(len(test_dataset)))

    # Build model, optimizer, and loss
    print("Building Model...")
    if load_model:
        model, optimizer = utils.load_model("model.pt", ConversationalTransformer, optimizer_class=torch.optim.AdamW)
        model = model.to(device)
    else:
        model = ConversationalTransformer(input_size=embedding_dim, hidden_size=hidden_size, num_tokens=local_vocab.num_tokens, pad_index=local_vocab.PAD_token, sos_index=local_vocab.SOS_token, eos_index=local_vocab.EOS_token, num_heads=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, forward_expansion=forward_expansion, seperation_index=local_vocab.indexes_from_tokens(["sep"])[0], dropout=dropout, max_len=max_len, max_turns=10).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = utils.optimizer_to_cuda(optimizer)
    print("Model Parameters: " + utils.count_parameters(model))

    print("Training With Batch Size Of " + str(batch_size * batch_accumulations) + "...")
    best_val = float('inf')
    for epoch in range(num_epochs):
        print("\n" + utils.colors.BOLD + "Epoch " + str(epoch + 1) + utils.colors.END)

        # Train epoch
        train_loss, train_acc = train_epoch(model=model, dataset=train_dataset, test_dataset=test_dataset, optimizer=optimizer, epoch=epoch, test_every=10000)

        # Quantitative evaluation
        test_loss, test_acc = test_epoch(model=model, dataset=test_dataset)
        writer.add_scalar("train_acc", train_acc, (epoch + 1) * len(train_dataset))
        writer.add_scalar("test_ppl", math.exp(test_loss), (epoch + 1) * len(train_dataset))
        writer.add_scalar("test_acc", test_acc, (epoch + 1) * len(train_dataset))
        print(str("Train PPL: " + utils.colors.BLUE + utils.colors.BOLD + "{: <10}" + utils.colors.END + " Test PPL: " + utils.colors.BLUE + utils.colors.BOLD + "{: <10}" + utils.colors.END).format(str(round(math.exp(train_loss), 2)), str(round(math.exp(test_loss), 2))))
        print(str("Train Acc: " + utils.colors.BLUE + utils.colors.BOLD + "{: <10}" + utils.colors.END + " Test Acc: " + utils.colors.BLUE + utils.colors.BOLD + "{: <10}" + utils.colors.END).format(str(round(train_acc * 100, 2)) + "%", str(round(test_acc * 100, 2)) + "%"))

        # Qualitative evaluation
        model.eval()
        sentence = "hello"
        print("Input: " + sentence)
        print("Output: " + str(eval_sentence(sentence, model, local_vocab)))
        sentence = "hello how are you?"
        print("Input: " + sentence)
        print("Output: " + str(eval_sentence(sentence, model, local_vocab)))
        sentence = "what is your name?"
        print("Input: " + sentence)
        print("Output: " + str(eval_sentence(sentence, model, local_vocab)))
        model.train()

        # Save model
        if test_loss < best_val:
            utils.save_model("model.pt", model, optimizer=optimizer)
            best_val = test_loss

def train_epoch(model, dataset, test_dataset, optimizer, epoch, test_every, scheduler=None):
    loss_meter, train_losses, train_acc = utils.ExponentialAverage(), [], []
    dataset.shuffle()
    model.train()
    with utils.train_progress_bar(total=len(dataset) // batch_size, desc="Training") as pbar:
        for iteration, (inputs, targets) in enumerate(dataset):
            try:
                # Train on batch
                inputs, targets = inputs.to(device), targets.to(device)

                # Feed through model
                model_targets = torch.cat((torch.full((1, targets.shape[1]), fill_value=local_vocab.SOS_token, device=device), targets[:-1]), dim=0)
                outputs = model(inputs, model_targets)

                # Get loss (reshape for loss and add EOS token to target)
                loss, print_loss = losses.seq_mask_crossentropy_loss(outputs, targets, mask=(targets != local_vocab.PAD_token), device=device)
                loss /= batch_accumulations
                    
                # Backprop and optimize
                loss.backward()
                if iteration % batch_accumulations == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None: scheduler.step()

                # Update progress bar
                loss_meter.update(math.exp(print_loss))
                train_losses.append(print_loss)
                train_acc.append((torch.logical_and(torch.argmax(outputs, dim=-1) == targets, targets != local_vocab.PAD_token).sum() / (targets != local_vocab.PAD_token).sum()).item())
                pbar.set_postfix(ppl=loss_meter.value)
                writer.add_scalar("ppl", loss_meter.value, epoch * (len(dataset) // batch_size) + iteration * batch_size)
            except RuntimeError as e:
                if 'out of memory' in str(e): # Handle OOM Errors
                    print("Out of Memory")
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    gc.collect()
                else: raise e
            pbar.update(1)
    return sum(train_losses) / len(train_losses), sum(train_acc) / len(train_acc)

def test_epoch(model, dataset):
    test_loss, test_acc = [], []
    model.eval()
    with torch.no_grad():
        with utils.test_progress_bar(total=len(dataset) // batch_size, desc="Testing") as pbar:
            for iteration, (inputs, targets) in enumerate(dataset):
                # Test on batch
                inputs, targets = inputs.to(device), targets.to(device)

                # Feed through model
                model_targets = torch.cat((torch.full((1, targets.shape[1]), fill_value=local_vocab.SOS_token, device=device), targets[:-1]), dim=0)
                outputs = model(inputs, model_targets)

                # Get loss (reshape for loss and add EOS token to target)
                _, print_loss = losses.seq_mask_crossentropy_loss(outputs, targets, mask=(targets != local_vocab.PAD_token), device=device)

                # Update loss
                test_loss.append(print_loss)
                test_acc.append((torch.logical_and(torch.argmax(outputs, dim=-1) == targets, targets != local_vocab.PAD_token).sum() / (targets != local_vocab.PAD_token).sum()).item())
                pbar.update(1)
    return sum(test_loss) / len(test_loss), sum(test_acc) / len(test_acc),

def test_wer_epoch(model, dataset):
    test_wer = []
    model.eval()
    with torch.no_grad():
        with utils.test_progress_bar(total=len(dataset) // batch_size, desc="Testing") as pbar:
            for iteration, (inputs, targets) in enumerate(dataset):
                # Move batch to device
                inputs = inputs.to(device)

                # Feed through model
                outputs = model(inputs)

                # Get wer
                outputs = tokenization.untokenize_bpe(local_vocab.tokens_from_indexes(torch.argmax(outputs, dim=-1).transpose(0, 1)))
                targets = tokenization.untokenize_bpe(local_vocab.tokens_from_indexes(targets.transpose(0, 1)))

                test_wer.append(metrics.word_error_rate(outputs, targets, ignore_case=True))
                pbar.update(1)

    return sum(test_wer) / len(test_wer)

if __name__ == "__main__":
    main()
    os.system('systemctl suspend')