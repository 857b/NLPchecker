import torch
import math

def load_distances(vocab, filename='_data/vocab_dist.txt'):
    rt = {}
    for i in vocab.values():
        rt[i] = []
    with open(filename, 'r') as f:
        for line in f:
            l = line.split()
            tk0 = int(l[0])
            tk1 = int(l[1])
            d = float(l[2])
            rt[tk0].append((tk1, d))
            rt[tk1].append((tk0, d))
    return rt

def print_distances(tokenizer, distances, id_word, count=10):
    ds = [(d, j) for (j,d) in distances[id_word]]
    ds.sort()
    ds=ds[:count]
    for (d,j) in ds:
        print(tokenizer.convert_ids_to_tokens(j), ":", d)

class GenerationException(ValueError):
    pass

class Confuser:
    def __init__(self, tokenizer, distances=None):
        self.mask_token_id  = tokenizer.mask_token_id
        if distances is None:
            distances = load_distances(tokenizer.get_vocab())
        self.confusion = {}
        for tk0,nns in distances.items():
            s   = 0
            cfs = []
            for tk1,d in nns:
                tk1_p = math.exp(-d)
                cfs.append((tk1, tk1_p))
                s += tk1_p
            if s > 0:
                ns = (1 - 1 / (1 + s))
                alpha = ns / s
                for i in range(len(cfs)):
                    tk1,tk1_p = cfs[i]
                    cfs[i] = tk1,(alpha * tk1_p)
                self.confusion[tk0] = ns,cfs

    def print_confusion(self, tokenizer, id_word, count=10):
        if id_word not in self.confusion:
            return
        cs = [(-c, j) for (j,c) in self.confusion[id_word][1]]
        cs.sort()
        cs = cs[:count]
        for (c,j) in cs:
            print(tokenizer.convert_ids_to_tokens(j), ":", -c)

    def make_correct(self, seq):
        count = 0
        for tk in seq:
            if tk in self.confusion:
                count += 1
        count=torch.randint(count, [1]).item()

        for (i,tk) in enumerate(seq):
            if tk in self.confusion:
                count -= 1
                if count < 0:
                    masked_seq = seq[:]
                    masked_seq[i] = self.mask_token_id
                    return (i, tk, masked_seq)

        raise GenerationException('make_correct: no confusable token')

    """ return (index, original, replacement, masked_seq) """
    def make_error(self, seq):
        s = 0
        for tk in seq:
            if tk not in self.confusion:
                continue
            s += self.confusion[tk][0]
        s *= torch.rand(1).item()

        for (i,tk) in enumerate(seq):
            if tk not in self.confusion:
                continue
            cs = self.confusion[tk]
            if s - cs[0] < 0:
                for rep,c in cs[1]:
                    s -= c
                    if s < 0:
                        masked_seq = seq[:]
                        masked_seq[i] = self.mask_token_id
                        return (i, tk, rep, masked_seq)
            else:
                s -= cs[0]

        raise GenerationException('make_error: no confusable token')

