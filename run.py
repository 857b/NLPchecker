import torch
import models, loaders, gen
from gen import Confuser
from tqdm import tqdm, trange
from util import to_numpy
import json

device = None

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    print("Warning: using CPU")

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
    model.to(device)
    
    dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=8)
    outputs = [[], []]
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluation"):
            out = model(batch['index'].to(device),
                        batch['token'].to(device),
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device))
            for l,o in zip(batch['label'], out):
                outputs[l].append(o.item())
    return outputs


def gen_with_hidden_states(name='train', count=10, save_path=None):
    tokenizer = models.tokenizer()
    
    print("loading dataset...")
    src_dataset = loaders.make_src_dataset(tokenizer,
                    name=name, count=count)
    
    print("loading model...")
    extracter  = models.extractHiddenState()
    extracter.eval()
    extracter.to(device)

    print("loading confusion...")
    confuser   = gen.Confuser(tokenizer)
    
    print("generating samples...")
    gen_dataset = loaders.make_gen_dataset(src_dataset, confuser)
    del src_dataset

    print("computing hidden states...")
    c_dataset   = loaders.compute_hidden_state(gen_dataset, extracter, device)

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
            pred = model(batch['token'].to(device),
                         batch['hidden_state'].to(torch.float).to(device))
            loss += criterion(pred, batch['label'].to(torch.float)
                                                  .to(device)).item()
            num_sample += len(batch['token'])

            pred = to_numpy(torch.sigmoid(pred))
            for p,l in zip(pred > 0.5, to_numpy(batch['label'])):
                confusion[l][p] += 1
    
    acc = confusion[0][0] + confusion[1][1]
    return loss/num_sample, acc/num_sample, confusion
            

def train_classifier(data_path, num_epoch=3, lr=5e-3, save=None,
        test_data_path=None, eval_period=10):

    datas = loaders.load_from_disk(data_path)
    loaders.set_dataset_format_with_hidden(datas)

    test_datas = None
    if test_data_path:
        test_datas = loaders.load_from_disk(test_data_path)
        loaders.set_dataset_format_with_hidden(test_datas)

    model = models.classifier(models.tokenizer())
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    dataloader = torch.utils.data.DataLoader(datas, batch_size=8)

    losses = []
    accs   = []
    test_losses = []
    test_accs   = []

    for n_epoch in trange(num_epoch, desc="epoch"):
        epoch_loss = 0
        num_sample = 0

        for batch in tqdm(dataloader, desc="batches", leave=False):
            pred = model(batch['token'].to(device),
                         batch['hidden_state'].to(torch.float)
                                              .to(device))
            loss = criterion(pred, batch['label'].to(torch.float)
                                                 .to(device))
            epoch_loss += loss.item()
            num_sample += len(batch['token'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(epoch_loss / num_sample)

        if save and n_epoch % eval_period == 0:
            torch.save(model.state_dict(),
                    '{}/epoch_{}.json'.format(save, n_epoch))

            _,train_acc,_ = evaluate_classifier(datas, model)
            accs.append(train_acc)
            if test_datas is not None:
                test_loss,test_acc,_ = evaluate_classifier(test_datas, model)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
    
    return model, {
            'train_loss': (1, losses),
            'train_acc':  (eval_period, accs),
            'test_loss':  (eval_period, test_losses),
            'test_acc':   (eval_period, test_accs)
        }
