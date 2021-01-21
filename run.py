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

# --- Maksed Logit ---

def seuil_optimal(output):
    s_0 = output[0][:]
    s_0.sort()
    s_1 = output[1][:]
    s_1.sort()

    i_0 = 0
    i_1 = 0
    while i_0/len(s_0) < 1 - i_1/len(s_1):
        if s_0[i_0] < s_1[i_1]:
            i_0 += 1
        else:
            i_1 += 1
    return min(s_0[i_0], s_1[i_1]), i_0/len(s_0)

def evaluate_maskedLogit(data_path, seuil=None):
    print("loading dataset...")
    datas = loaders.load_from_disk(data_path)
    loaders.set_dataset_format(datas)
    
    print("loading model...")
    model     = models.maskedLogit()
    model.eval()
    model.to(device)

    dataloader = torch.utils.data.DataLoader(datas, batch_size=16)
    
    outputs = [[], []]
    confusion = [[0,0],[0,0]]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluation"):
            out = model(batch['index'].to(device),
                        batch['token'].to(device),
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device))
            for l,o in zip(to_numpy(batch['label']), out):
                outputs[l].append(o.item())
                if seuil is not None:
                    confusion[l][o < seuil] += 1
    return outputs, confusion

# --- Classifier ---

def load_classifier(checkpoint):
    model = models.classifier(models.tokenizer())
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device)

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
        test_data_path=None, eval_period=10,
        model_params=None, save_period=1):

    datas = loaders.load_from_disk(data_path)
    loaders.set_dataset_format_with_hidden(datas)

    test_datas = None
    if test_data_path:
        test_datas = loaders.load_from_disk(test_data_path)
        loaders.set_dataset_format_with_hidden(test_datas)

    model = models.classifier(models.tokenizer())
    if model_params:
        model.load_state_dict(torch.load(model_params, map_location=device))
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    dataloader = torch.utils.data.DataLoader(datas, batch_size=16)

    losses      = []
    periods     = []
    period_losses = []
    test_losses = []
    test_accs   = []

    period_loss   = 0
    period_sample = 0
    num_checkpoint = 0
    save_i = 0

    for n_epoch in trange(num_epoch, desc="epoch"):
        epoch_loss = 0
        epoch_sample = 0

        for batch in tqdm(dataloader, desc="batches", leave=False):
            pred = model(batch['token'].to(device),
                         batch['hidden_state'].to(torch.float)
                                              .to(device))
            loss = criterion(pred, batch['label'].to(torch.float)
                                                 .to(device))
            epoch_loss    += loss.item()
            epoch_sample  += len(batch['token'])
            period_loss   += loss.item()
            period_sample += len(batch['token'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if period_sample > eval_period:
                save_i += 1
                if save and save_i % save_period == 0:
                    torch.save(model.state_dict(),
                            '{}/checkpoint_{}.json'
                            .format(save, num_checkpoint))

                periods.append(period_sample)
                period_losses.append(period_loss/period_sample)

                if test_datas is not None:
                    test_loss,test_acc,_ = evaluate_classifier(test_datas, model)
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
                    print("{} :: train loss: {}   test loss: {}   test acc: {}"
                            .format(num_checkpoint,
                                    period_loss/period_sample,
                                    test_loss, test_acc))
                else:
                    print("train loss: {}".format(num_checkpoint,
                                    period_loss/period_sample))

                num_checkpoint += 1
                period_sample = 0
                period_loss = 0
                

        losses.append(epoch_loss / epoch_sample)
    
    return model, {
            'epoch_loss':  losses,
            'periods':     periods,
            'period_loss': period_losses,
            'test_loss':   test_losses,
            'test_acc':    test_accs
        }
