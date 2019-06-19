# coding: utf-8
import argparse
import json
import time
import math
import os
import io
import nltk
import numpy as np
import _pickle as pickle
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from random import shuffle
import codecs
import random
import string
from tqdm import tqdm
import glob
import spacy
from collections import Counter
import en_core_web_sm
stopwords = nltk.corpus.stopwords.words('english')

parser = argparse.ArgumentParser(description='Preprocessing for bilingual dictionaries')
parser.add_argument('--data', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--out-dir', type=str, default='data/',
                    help='location of the output directory')
parser.add_argument('--pivot-lang', type=str, default='en',
                    help='Target pivot language')
parser.add_argument('--emb-dim', type=int, default=300,
                    help='Embedding Dimension')
parser.add_argument('--lower', action='store_true',
                    help='Lowercase lemmas from wordnet.')

args = parser.parse_args()
word2idx = {'<unk>':0}
idx2word = ['<unk>']
emb_dict = {'<unk>':np.random.rand(300,).tolist()}

nlp_en = spacy.load("en")
nlp_de = spacy.load("de")
nlp_fr = spacy.load("fr")
nlp_pt = spacy.load("pt")
nlp_it = spacy.load("it")
nlp_es = spacy.load("es")

# nlp_xx = spacy.load("xx_core_web_sm")

# taken from https://stackoverflow.com/questions/8689795/how-can-i-remove-non-ascii-characters-but-leave-periods-and-spaces-using-python
def remove_non_ascii(text):
    printable = set(string.printable)
    cleaned_text = filter(lambda x: x in printable, text)
    return ''.join([ch for ch in cleaned_text]) # a hack to convert filter object to string for Python 3

def preprocessing(text):
    return [tok for tok in text if tok not in stopwords and tok not in string.punctuation]

def add_word(word):
    if word not in word2idx:
        idx2word.append(word)
        word2idx[word] = len(idx2word) - 1
    return word2idx[word]

def return_spacy_model(lng):
    if lng == 'en':
        return nlp_en
    elif lng == 'es':
        return nlp_es
    elif lng == 'it':
        return nlp_it
    elif lng == 'pt':
        return nlp_pt
    elif lng == 'de':
        return nlp_de
    elif lng == 'fr':
        return nlp_fr

def load_embeddings(lng):
    emb_path = os.path.join(args.data, 'monolingual/wiki.' + lng + '.vec')
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for idx, lines in enumerate(f):
            if idx == 0:
                continue
            word, vect = lines.rstrip('\n').split(' ', 1)
            word = (lng + '_' + word.lower())
            vect = np.fromstring(vect, sep=' ')
            if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                vect[0] = 0.01
            assert vect.shape == (args.emb_dim, ), idx
            emb_dict[word] = vect.tolist()

    return emb_dict

def create_vocab(in_path, add_eos=True):
    for w in get_tokens_from_file(in_path, add_eos):
        add_word(w)

def create_joint_embeddings():
    embs = []
    for w in idx2word:
        if w in emb_dict:
            embs.append(emb_dict[w])
        else:
            embs.append(emb_dict['<unk>'])
    print('Final Embedding Shape : ' + str(np.array(embs).shape))
    return np.array(embs)

def preprocess():
    all_lng = ['de', 'es', 'fr', 'it', 'pt', 'en']
    tgt_lng = args.pivot_lang
    src_lng = all_lng.remove(tgt_lng)
    fout = open(os.path.join(args.data, 'crosslingual/dictionaries/train.' + tgt_lng + '.txt'), 'w')
    load_embeddings(tgt_lng)
    print('Loaded embeddings for language: ' + str(tgt_lng))
    for src in tqdm(src_lng, total=len(src_lng)):
        fin = open(os.path.join(args.data, 'crosslingual/dictionaries/' + src + '-' + tgt_lng + '.0-5000.txt'), 'r')
        for lines in fin:
            lines  = lines.rstrip('\n').split()
            add_word(src + '_' + lines[0])
            add_word(src + '_' + lines[1])
            nlp_src = return_spacy_model(src)
            nlp_tgt = return_spacy_model(tgt)
            src_pos = nlp_src(lines[0])[0].pos_
            tgt_pos = nlp_tgt(lines[1])[0].pos_


            output = {'x': src + '_' + lines[0], 'y': src + '_' + lines[1], 'pos_x': src_pos, 'pos_y': tgt_pos}
            fout.write(str(json.dumps(output) + '\n'))
            load_embeddings(src)
            print('Loaded embeddings for language: ' + str(src))

    if not os.path.isfile(os.path.join(args.data, 'joined_emb.pb')):
        embs = create_joint_embeddings()
        print('Created joint embeddings')
        pickle.dumps(embs, open(os.path.join(args.data, 'joined_emb.pb'), 'wb'))
        print('Saved joint embeddings')

preprocess()
