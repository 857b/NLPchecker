import torch
import torch.nn as nn
from transformers import CamembertModel, CamembertForMaskedLM,\
        CamembertTokenizer

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
    def __init__(self, transformer, normalize=False):
        super(MaskedLogit, self).__init__()
        self.transformer = transformer
        self.normalize   = normalize

    def forward(self, index, tk, input_ids, attention_mask=None):
        out = self.transformer(input_ids, attention_mask=attention_mask)[0]
        out_index = index2(out, index)
        out_token = index2(out_index, tk)
        if self.normalize:
            return out_token / torch.linalg.norm(out_index)
        else:
            return out_token

def maskedLogit(normalize=False):
    transformer = CamembertForMaskedLM.from_pretrained("camembert-base")
    return MaskedLogit(transformer, normalize=normalize)

hidden_state_size = 768

class ExtractHiddenState(nn.Module):
    def __init__(self, transformer):
        super(ExtractHiddenState, self).__init__()
        self.transformer = transformer
    
    def forward(self, index, input_ids, attention_mask=None):
        out = self.transformer(input_ids, attention_mask=attention_mask,
                    return_dict=True).last_hidden_state
        return index2(out, index)

def extractHiddenState():
    transformer = CamembertModel.from_pretrained("camembert-base")
    return ExtractHiddenState(transformer)

class Classifier(nn.Module):
    def __init__(self, vocab_size, hs1=128):
        super(Classifier, self).__init__()
        self.vocab_size = vocab_size
        self.lin1 = nn.Linear(hidden_state_size + vocab_size,
                        hs1, bias=True)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hs1, 1, bias=True)

    def forward(self, tk, hs0):
        tk_one_hot = nn.functional.one_hot(tk, num_classes=self.vocab_size)
        out = self.lin1(torch.cat([tk_one_hot, hs0], dim=-1))
        out = self.relu(out)
        out = self.lin2(out)
        return out.squeeze(dim=-1)

def classifier(tokenizer):
    return Classifier(tokenizer.vocab_size)
