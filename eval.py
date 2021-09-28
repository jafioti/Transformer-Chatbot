import torch

from Railgun import tokenization, vocab
from SidekickAI.Utilities import utils
from model import ConversationalTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load model and vocab
    print("Loading model...")
    model = utils.load_model("model.pt", ConversationalTransformer).to(device)
    model.eval()
    local_vocab = vocab.load_wordpiece_vocab()

    # Conversation loop
    current_conversation = []
    while True:
        # Get input
        inp = ""
        while len(inp) == 0:
            inp = input("> ")
        current_conversation.append(inp)
        # Clip conversation
        while len(tokenization.tokenize_bpe(" sep ".join(current_conversation))) >= 200 or len(current_conversation) > 2:
            del current_conversation[0]
        if len(current_conversation) == 0:
            print("Too long")
            continue
        # Process and output
        out = eval_sentence(" sep ".join(current_conversation), model, local_vocab, beam_width=None)
        print("< " + str(out))
        current_conversation.append(out)

def eval_sentence(input_sentence, model, local_vocab, beam_width=None):
    model.eval()
    # Convert the input sentence to indexes
    input_tokens = tokenization.tokenize_wordpiece(input_sentence)
    input_indexes = local_vocab.indexes_from_tokens(input_tokens)

    # Feed into eval_indexes function
    output_indexes = eval_indexes(input_indexes, model, beam_width)
    print(len(output_indexes))

    # Convert back into text
    output_tokens = local_vocab.tokens_from_indexes(output_indexes)
    output_sentence = tokenization.untokenize_wordpiece(output_tokens)
    return output_sentence

def eval_indexes(input_indexes, model, beam_width=None):
    # Make input tensor
    input_variable = torch.LongTensor([input_indexes]).transpose(0, 1).to(device)

    # Feed through model
    if beam_width: # Beam search
        with torch.no_grad(): output_variable = model.beam_search(input_variable, beam_width)
        output_indexes = output_variable[0]
    else:
        with torch.no_grad(): output_variable = model(input_variable) # Make sure the seq len dimension comes first
        output_indexes = torch.argmax(output_variable, dim=-1)[:, 0]
    return output_indexes.tolist()

if __name__ == "__main__": main()