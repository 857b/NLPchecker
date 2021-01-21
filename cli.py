import argparse
import sys
import run
import torch
import json

parser = argparse.ArgumentParser()

parser.add_argument('--gen-dataset', dest='gen',
                type=str, default=None)
parser.add_argument('--src-data', dest='src_data',
                type=str, default=None)
parser.add_argument('--src-count', dest='src_count',
                type=int, default=1000)

parser.add_argument('--train-data', dest='train_data',
                type=str, default=None)
parser.add_argument('--test-data', dest='test_data',
                type=str, default=None)
parser.add_argument('--lr', dest='lr',
                type=float, default=5e-3)
parser.add_argument('--epochs', dest='num_epoch',
                type=int, default=5)
parser.add_argument('--eval-period', dest='eval_period',
                type=int, default=20000)
parser.add_argument('--model-params', dest='model_params',
                type=str, default=None)
parser.add_argument('save-period', dest='save_period',
                type=int, default=10)

parser.add_argument('--train', dest='train',
                type=str, default=None)

args = parser.parse_args()

if args.gen:
    if args.src_data is None:
        sys.exit(1)

    run.gen_with_hidden_states(name=args.src_data, count=args.src_count,
                               save_path=args.gen)
elif args.train:
    if args.train_data is None:
        sys.exit(1)

    model, loss = run.train_classifier(args.train_data,
                    num_epoch=args.num_epoch, lr=args.lr, save=args.train,
                    test_data_path=args.test_data,
                    eval_period=args.eval_period,
                    model_params=args.model_params,
                    save_period=args.save_period)
    torch.save(model.state_dict, '{}/final.json'.format(args.train))
    with open('{}/loss.json'.format(args.train), 'w') as f:
        json.dump(loss, f)
