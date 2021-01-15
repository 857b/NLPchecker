from datasets import load_dataset, load_from_disk
import torch

from gen import GenerationException

""" dataset allocine : test | train | validation """
def make_src_dataset(tokenizer, name='test', count=None):
    src_dataset = load_dataset("allocine")[name]
    if count:
        src_dataset = src_dataset.select(range(count))

    return src_dataset.map(
            lambda x: tokenizer(x['review'], truncation=True,
                            padding='max_length', max_length=50),
        batched=True, remove_columns=['review', 'label'])

def set_dataset_format(dataset):
    dataset.set_format(type='torch',
            columns=['label', 'index', 'correct', 'token',
                     'input_ids', 'attention_mask'])

def make_gen_dataset(src_dataset, confuser):
    def generator(src):
        indexes = []; corrects = []; tokens = []
        seqs    = []; labels   = []; att_masks = []
        def app(i, cr, tk, sq, la, mk):
            indexes.append(i); corrects.append(cr); tokens.append(tk)
            seqs.append(sq); labels.append(la), att_masks.append(mk)
        for src_seq, att_mask\
                in zip(src['input_ids'], src['attention_mask']):
            try:
                i,tk,seq = confuser.make_correct(src_seq)
                app(i, tk, tk, seq, 0, att_mask)
            except GenerationException:
                pass
            
            try:
                i,cr,tk,seq = confuser.make_error(src_seq)
                app(i, cr, tk, seq, 1, att_mask)
            except GenerationException:
                pass
        return {'index':     indexes, 'correct':corrects, 'token':tokens,
                'input_ids': seqs,    'label':  labels,
                'attention_mask': att_masks}

    gen_dataset = src_dataset.map(generator, batched=True, batch_size=1,
                        load_from_cache_file=False)
    set_dataset_format(gen_dataset)
    return gen_dataset

def set_dataset_format_with_hidden(dataset):
    dataset.set_format(type='torch',
            columns=['label', 'index', 'correct', 'token',
                     'input_ids', 'attention_mask', 'hidden_state'])

def compute_hidden_state(dataset, extracter):
    def run_extracter(src):
        indexes   = torch.tensor(src['index'])
        in_ids    = torch.tensor(src['input_ids'])
        att_masks = torch.tensor(src['attention_mask'])

        with torch.no_grad:
            hidden_states = extracter(indexes, input_ids, att_masks)
        return {'hidden_state' : hidden_states}
    
    c_dataset = dataset.map(run_extracter, batched=True, batch_size=1,
                        load_from_cache_file=False)

