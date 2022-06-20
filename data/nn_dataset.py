import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, filepath, vocab, max_len=128, set_name=''):
        super(TextDataset, self).__init__()
        self.sentence_pairs, self.labels = self.load_data(filepath, set_name)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, item):
        sentence_pair = self.sentence_pairs[item]
        label = self.labels[item]
        # seq_len = sum([len(sent.split()) for sent in sentence_pair]) + 3 # sos + 2 eos

        output = {'sent_pair': self.tokenize(sentence_pair),
                  'label': label,
                  # 'seq_len': seq_len,
                  }
        return {key: torch.tensor(value) for key, value in output.items()}

    def tokenize(self, sent_pair):
        """ Tokenize a sentence-pair."""
        tokenized_pair = []
        for sent in sent_pair:
            tokens = sent.split()
            for i, token in enumerate(tokens):
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
            tokens = [self.vocab.sos_index] + tokens + [self.vocab.eos_index]
            
            # Truncate if longer than max_len
            tokens = tokens[:self.max_len]
            # Pad if shorter than max_len
            padding = [self.vocab.pad_index for _ in range(self.max_len - len(tokens))]
            tokens.extend(padding)

            tokenized_pair.append(tokens)
        return tokenized_pair

    def load_data(self, path, set_name):
        """ Load data from preprocessed pair-score data. """
        f = open(path, 'r', encoding='utf-8')
        sentence_pairs, labels = [], []
        for line in f.readlines():
            sentence_pair, label = line.split('\t')[0: 2], line.split('\t')[-1]
            labels.append(float(label) / 5.) # normalize the score
            sentence_pairs.append([sent.strip() for sent in sentence_pair])

        print(f"Length of {set_name} dataset: ", len(sentence_pairs))
        return sentence_pairs, labels
