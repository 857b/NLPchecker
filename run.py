import torch
import models, loaders, gen
from gen import Confuser
from tqdm import tqdm
from datasets import total_allocated_bytes
import numpy as np
import matplotlib.pyplot as plt

def test_confuser():
    tokenizer = models.tokenizer()
    distances = gen.load_distances(tokenizer.get_vocab())
    confuser  = gen.Confuser(tokenizer, distances)
    for src in ["Bonjour", "monde", "manger"]:
        tk = models.word_to_tkid(tokenizer, src)
        print("---", src, "---")
        gen.print_distances(tokenizer, distances, tk)
        print("--- confusion")
        confuser.print_confusion(tokenizer, tk)


def test_maskedLogit(count=100, normalize=False):
    tokenizer = models.tokenizer()
    
    print("loading confusion...")
    confuser  = gen.Confuser(tokenizer)
    
    print("loading dataset...")
    src_dataset = loaders.make_src_dataset(tokenizer,
                    name='test', count=count)
    gen_dataset = loaders.make_gen_dataset(src_dataset, confuser)
    del src_dataset

    print("loading model...")
    model     = models.maskedLogit(normalize=normalize)
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=5)
    outputs = [[], []]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluation"):
            out = model(batch['index'], batch['token'],
                         batch['input_ids'], batch['attention_mask'])
            for l,o in zip(batch['label'], out):
                outputs[l].append(o.item())
    return outputs

def plot_maskedLogitError(outputs):
    def cumulated(vs):
        xs = np.sort(vs)
        ys = np.array(range(1,len(vs)+1)) / len(vs)
        return xs,ys
    ccx,ccy = cumulated(outputs[0])
    ecx,ecy = cumulated(outputs[1])
    ccy = 100 * ccy       # false positive
    ecy = 100 * (1 - ecy) # false negative
    plt.plot(ccx, ccy, label='false positive')
    plt.plot(ecx, ecy, label='false negative')
    plt.xlabel('threshold')
    plt.ylabel('%')
    plt.legend()
    plt.show()

def gen_with_hidden_states(count=10):
    tokenizer = models.tokenizer()
    
    print("loading model...")
    extracter = models.extractHiddenState()
    extracter.eval()

    print("loading confusion...")
    confuser  = gen.Confuser(tokenizer)
    
    print("loading dataset...")
    src_dataset = loaders.make_src_dataset(tokenizer,
                    name='train', count=count)
    
    print("generating examples...")
    gen_dataset = loaders.make_gen_dataset(src_dataset, confuser)
    del src_dataset

    print("computing hidden states...")
    c_dataset = loaders.compute_hidden_state(gen_dataset, extracter)

    return c_dataset
