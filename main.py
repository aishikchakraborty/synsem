# coding: utf-8
import argparse
import hashlib
import math
import os
import numpy as np
import random
import time
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import _pickle as pickle
import model
from random import shuffle


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='rnn',
                    help='type of model. Options are [retro, rnn, cbow]')
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--lex', '-l', action="append", type=str, default=[], dest='lex_rels',
                    help='list of type of lexical relations to capture. Options | syn | hyp | mer')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--wn_hid', type=int, default=100,
                    help='Dimension of the WN subspace')
parser.add_argument('--margin', type=float, default=2,
                    help='define the margin for the max-margin loss')
parser.add_argument('--n_margin', type=float, default=2,
                    help='define the margin for the negative sampling max-margin loss')
parser.add_argument('--patience', type=int, default=1,
                    help='How long before you reduce the LR.')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=14,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--random_seed', type=int, default=13370,
                    help='random seed')
parser.add_argument('--numpy_seed', type=int, default=1337,
                    help='numpy random seed')
parser.add_argument('--ss_t', type=float, default=1e-5,
                    help="subsample threshold")
parser.add_argument('--torch_seed', type=int, default=133,
                    help='pytorch random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpu', type=int, default=0,
                    help='use gpu x')
parser.add_argument('--log-interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='models/',
                    help='path to save the final model')
parser.add_argument('--save-emb', type=str, default='embeddings/',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--adaptive', action='store_true',
                    help='Use adaptive softmax. This speeds up computation.')
parser.add_argument('--neg_wn_ratio', type=float, default=10,
                    help='Negative sampling ratio for lexical subspaces.')
parser.add_argument('--distance', type=str, default='cosine',
                    help='Type of distance to use. Options are [pairwise, cosine]')
parser.add_argument('--optim', type=str, default='adam',
                    help='Type of optimizer to use. Options are [sgd, adagrad, adam]')
parser.add_argument('--reg', action='store_true', help='Regularize.')
parser.add_argument('--max_vocab_size', type=int, default=None,
                    help='Vocab size to use for the dataset.')
parser.add_argument('--pivot-lang', type=str, default='en',
                    help='Target pivot language')
args = parser.parse_args()

print(args)

if args.random_seed is not None:
    random.seed(args.random_seed)
if args.numpy_seed is not None:
    np.random.seed(args.numpy_seed)
if args.torch_seed is not None:
    torch.manual_seed(args.torch_seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:" + str(args.gpu) if args.cuda else "cpu")

class Dataset():
    def __init__(self, path):
        self.path = path
        self.x = []
        self.y = []
        self.pos_x = []
        self.pos_y = []

    def load_vocab(self):
        self.vocab = pickle.load(open(os.path.join(args.data, 'vocab.pb'), 'rb'))
        self.w2idx = {w:idx for idx, w in enumerate(self.vocab)}

    def load_dataset(self):
        self.load_vocab()
        f = open(self.path, 'r')
        for lines in f:
            lines = json.loads(lines.rstrip('\n'))
            self.x.append(self.w2idx[lines['x']])
            self.y.append(self.w2idx[lines['y']])
            self.pos_x.append(lines['pos_x'])
            self.pos_y.append(lines['pos_y'])


        self.pos_classes_x = set(self.pos_x)
        self.pos_classes_y = set(self.pos_y)

        p2idx_x = {p:idx for idx, p in enumerate(self.pos_classes_x)}
        p2idx_y = {p:idx for idx, p in enumerate(self.pos_classes_y)}
        self.pos_x = [p2idx_x[p] for p in self.pos_x]
        self.pos_y = [p2idx_y[p] for p in self.pos_y]

d = Dataset(os.path.join(args.data, 'crosslingual/dictionaries/train.' + args.pivot_lang + '.txt'))
d.load_dataset()
print('Loaded and binarized datasets')

emb = torch.tensor(pickle.load(open(os.path.join(args.data, 'joined_emb.pb'), 'rb'))).to(device)

lr = args.lr
best_val_loss = None

synsem = model.Synsem(len(d.vocab), args.emsize, len(d.pos_classes_x), len(d.pos_classes_y), emb).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adagrad(synsem.parameters(), lr=lr) if args.optim == 'adagrad' \
                else torch.optim.Adam(synsem.parameters(), lr=lr) if args.optim == 'adam' \
                else torch.optim.SGD(synsem.parameters(), lr=lr)

milestones = [10, 20, 30, 40]
print(milestones)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3)

def train(epoch):
    # Turn on training mode which enables dropout.
    synsem.train()
    total_loss_ = 0.
    translation_loss_ = 0.
    reconstruction_loss_ = 0.
    semantic_loss_ = 0.
    syntactic_loss_ = 0.
    start_time = time.time()

    num_batches = len(d.x)//args.batch_size

    for i in range(num_batches):
        batch_x = torch.LongTensor(d.x[i*args.batch_size : (i+1)*args.batch_size]).to(device)
        batch_y = torch.LongTensor(d.y[i*args.batch_size : (i+1)*args.batch_size]).to(device)
        batch_pos_x = torch.LongTensor(d.pos_x[i*args.batch_size : (i+1)*args.batch_size]).to(device)
        batch_pos_y = torch.LongTensor(d.pos_y[i*args.batch_size : (i+1)*args.batch_size]).to(device)

        optimizer.zero_grad()

        translation_loss, reconstruction_loss, semantic_loss, pos_x, pos_y = synsem(batch_x, batch_y)
        syntactic_loss = (criterion(pos_x, batch_pos_x) + criterion(pos_y, batch_pos_y))
        total_loss = translation_loss + reconstruction_loss + semantic_loss + syntactic_loss

        total_loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(synsem.parameters(), args.clip)
        optimizer.step()

        total_loss_ += total_loss.item()
        translation_loss_ += translation_loss.item()
        reconstruction_loss_ += reconstruction_loss.item()
        semantic_loss_ += semantic_loss.item()
        syntactic_loss_ += syntactic_loss.item()

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss_ / args.log_interval
            cur_translation_loss = translation_loss_ / args.log_interval
            cur_semantic_loss = semantic_loss_ / args.log_interval
            cur_syntactic_loss = syntactic_loss_ / args.log_interval

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.10f} | ms/batch {:5.2f} | loss {:5.2f} | translation loss {:5.2f} | semantic loss {:5.2f} | syntactic loss {:5.2f} |'
                    .format(epoch, i, len(num_batches), optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
                        cur_loss, cur_translation_loss, cur_semantic_loss, cur_semantic_loss))


            start_time = time.time()
            total_loss_ = 0
            translation_loss_ = 0
            reconstruction_loss_ = 0
            semantic_loss_ = 0
            syntactic_loss_ = 0

    print()



model_name = os.path.join(args.save, 'model_' + args.data + '_'  + str(args.emsize) + '_' + args.pivot_lang + '.pt')
syn_emb_name = os.path.join(args.save_emb, 'syn_emb_' + args.data + '_'  + str(args.emsize) + '_' + args.pivot_lang + '.pb')
sem_emb_name = os.path.join(args.save_emb, 'sem_emb_' + args.data + '_'  + str(args.emsize) + '_' + args.pivot_lang + '.pb')
w_name = os.path.join(args.save_emb, 'W_' + args.data + '_'  + str(args.emsize) + '_' + args.pivot_lang + '.pb')

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch)

        print('Saving Model')
        with open(model_name, 'wb') as f:
            torch.save(model, f)
        print('Saving learnt embeddings : %s' % emb_name)
        pickle.dump(synsem.syn_emb.weight.data, open(syn_emb_name, 'wb'))
        pickle.dump(synsem.sem_emb.weight.data, open(sem_emb_name, 'wb'))
        pickle.dump(synsem.W.weight.data, open(w_name, 'wb'))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
