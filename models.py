import torch
import torch.nn as nn
from transformers import CamembertForMaskedLM, CamembertTokenizer

def tokenizer():
    return CamembertTokenizer.from_pretrained("camembert-base")

def word_to_tkid(tokenizer, word):
    tks = tokenizer.tokenize(word)
    if len(tks) != 1:
        raise Exception('word_to_tkid')
    return tokenizer.convert_tokens_to_ids(tks[0])

# return [o[index[i]] for o in out]
def index2(src, index):
    flat_index = src.shape[1] * torch.tensor(range(src.shape[0])) + index
    return torch.flatten(src,start_dim=0, end_dim=1)[flat_index]

class MaskedLogit(nn.Module):
    def __init__(self, transformer):
        super(MaskedLogit, self).__init__()
        self.transformer = transformer

    def forward(self, index, tk, input_ids, attention_mask=None):
        out = self.transformer(input_ids, attention_mask=attention_mask)[0]
        out_index = index2(out, index)
        out_token = index2(out_index, tk)
        return out_token

def maskedLogit():
    transformer = CamembertForMaskedLM.from_pretrained("camembert-base")
    return MaskedLogit(transformer)