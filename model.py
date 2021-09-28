from typing import Optional
import torch.nn as nn
import torch
from SidekickAI.Modules.Transformers import Seq2SeqTransformer

class ConversationalTransformer(nn.Module):
    '''A conversational transformer model using Sidekick Seq2SeqTransformer with turn embeddings\n
        Init Inputs:
            seperation_token: The index of the seperation token
            max_turns: The maximum number of turns an input can have. Must be 1 or greater.
        Inputs:
            src (Tensor): The input sequence of shape (src length, batch size)
            trg (Tensor) [default=None]: The target sequence of shape (trg length, batch size)
        Returns:
            output (Tensor): The return sequence of shape (target length, batch size, target tokens)'''
    def __init__(self, input_size, hidden_size, num_tokens, pad_index, sos_index, eos_index, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, seperation_index, dropout=0.1, max_len=200, max_turns=3, learned_pos_embeddings=False):
        super().__init__()
        self.hyperparameters = locals()
        self.embeddings = nn.Embedding(num_tokens, input_size)
        self.turn_embeddings = nn.Embedding(2, input_size)
        self.transformer = Seq2SeqTransformer(input_size, hidden_size, num_tokens, pad_index, sos_index, eos_index, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, num_tokens, pad_index, dropout, max_len, learned_pos_embeddings)
        self.seperation_index = seperation_index
        self.max_turns = max_turns

    def forward(self, src, trg:Optional[torch.Tensor] = None):
        '''src: (seq len, batch size)\n
        trg: (seq len, batch size) or None'''
        # Get turn embeddings
        turns = torch.cumsum((src == self.seperation_index), dim=0)
        assert not (turns > self.max_turns).any(), "Input has more turns than max turns!" # Check for max turns
        # Embed src
        turns = torch.fmod(turns, 2)
        turns[turns == 1] = 2
        turns[turns == 0] = 1
        turns[turns == 2] = 0
        src = self.embeddings(src) + self.turn_embeddings(turns)

        # Run through transformer
        return self.transformer(src, trg)

    def beam_search(self, src, beam_width):
        '''src: (seq len, batch size)'''
        # Get turn embeddings
        turns = torch.cumsum((src == self.seperation_index), dim=0)
        assert not (turns > self.max_turns).any(), "Input has more turns than max turns!" # Check for max turns
        # Embed src
        src = self.embeddings(src) + self.turn_embeddings(turns)

        return self.transformer.beam_search(src, beam_width)