import torch
from torch import nn
import torch.nn.functional as F
from SidekickAI.Utilities import utils
from Railgun import vocab
from model import ConversationalTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("|COMPILING MODEL IN TORCHSCRIPT|")

print("Loading Model...")
model = utils.load_model("model.pt", ConversationalTransformer).to(device)
local_vocab = vocab.load_wordpiece_vocab()
#model = ConversationalTransformer(300, 256, local_vocab.num_tokens, local_vocab.PAD_token, local_vocab.SOS_token, local_vocab.EOS_token, 8, 2, 2, 2, 4, dropout=0.1, max_len=200, max_turns=10)
model.eval()

print("Tracing Model...")
traced_model = torch.jit.script(model)

print("Saving Model...")
torch.jit.save(traced_model, "compiled_model.pt")
print("|MODEL COMPILED|")