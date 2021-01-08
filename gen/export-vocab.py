from transformers import CamembertTokenizer

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
vocab = tokenizer.get_vocab()

for w,i in vocab.items():
    if w[0] != '<':
        print(i, w)
