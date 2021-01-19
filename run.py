import torch
import models, loaders, gen
from gen import Confuser
from tqdm import tqdm, trange
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
    
    dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=8)
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

def gen_with_hidden_states(name='train', count=10, save_path=None):
    tokenizer = models.tokenizer()
    
    print("loading dataset...")
    src_dataset = loaders.make_src_dataset(tokenizer,
                    name=name, count=count)
    
    print("loading model...")
    extracter  = models.extractHiddenState()
    extracter.eval()

    print("loading confusion...")
    confuser   = gen.Confuser(tokenizer)
    
    print("generating examples...")
    gen_dataset = loaders.make_gen_dataset(src_dataset, confuser)
    del src_dataset

    print("computing hidden states...")
    c_dataset   = loaders.compute_hidden_state(gen_dataset, extracter)

    if save_path:
        print("saving to {}".format(save_path))
        c_dataset.save_to_disk(save_path)

    return c_dataset

def load_toy_datas(data_path="_data/train_toy"):
    tokenizer = models.tokenizer()
    datas = loaders.load_from_disk(data_path)
    loaders.set_dataset_format_with_hidden(datas)
    return tokenizer, datas

def evaluate_classifier(datas, model):
    dataloader = torch.utils.data.DataLoader(datas, batch_size=8)
    confusion  = [[0, 0], [0, 0]]
    loss       = 0
    num_sample = 0
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluation", leave=None):
            pred = model(batch['token'],
                         batch['hidden_state'].to(torch.float))
            loss += criterion(pred, batch['label'].to(torch.float))
            num_sample += len(batch['token'])

            pred = torch.nn.functional.sigmoid(pred).detach().numpy()
            for p,l in zip(pred > 0.5, batch['label'].detach().numpy()):
                confusion[l][p] += 1
    
    acc = confusion[0][0] + confusion[1][1]
    return loss/num_sample, acc/num_sample, confusion
            

def train_classifier(data_path, num_epoch=3, lr=1e-2):
    datas = loaders.load_from_disk(data_path)
    loaders.set_dataset_format_with_hidden(datas)

    model = models.classifier(models.tokenizer())
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    dataloader = torch.utils.data.DataLoader(datas, batch_size=8)

    losses = []
    for _ in trange(num_epoch, desc="epoch"):
        epoch_loss = 0
        num_sample = 0

        for batch in tqdm(dataloader, desc="batches", leave=False):
            pred = model(batch['token'],
                         batch['hidden_state'].to(torch.float))
            loss = criterion(pred, batch['label'].to(torch.float))
            epoch_loss += loss.item()
            num_sample += len(batch['token'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(epoch_loss / num_sample)
    
    return model, losses
