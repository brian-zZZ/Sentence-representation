import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


def load_data(path, set_name):
    """ Load data from original STS-B dataset. """
    f = open(path, 'r', encoding='utf-8')
    sentence_pairs, labels = [], []
    for line in f.readlines():
        label, sentence_pair = line.split('\t')[4], line.split('\t')[5: 7]
        labels.append(float(label) / 5.) # normalize the score
        sentence_pairs.append([sent.strip() for sent in sentence_pair])

    print(f"Length of {set_name} dataset: ", len(sentence_pairs))
    return sentence_pairs, labels


class BertTextDataset(Dataset):
    def __init__(self, filepath, set_name):
        super(BertTextDataset, self).__init__()
        self.sentence_pairs, self.labels = load_data(filepath, set_name)

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, item):
        sentence_pair = self.sentence_pairs[item]
        label = self.labels[item]
        return sentence_pair, label


class BatchTextCall(object):
    """Call function for tokenizing and getting batch text.
    """
    def __init__(self, tokenizer, max_len=512, bert_type='albert', siamese=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bert_type = bert_type
        self.siamese = siamese

    def tokenize_text(self, batch_text):
        source = self.tokenizer(batch_text, max_length=self.max_len,
                                truncation=True, padding='max_length', return_tensors='pt')
        token = source.get('input_ids').squeeze(1)
        mask = source.get('attention_mask').squeeze(1)

        # For models that supports NSP task, directly get token_type_ids
        if self.bert_type in ['bert', 'xlnet', 'albert']: # self.bert_type.spilt('_')[0]
            segment = source.get('token_type_ids').squeeze(1)
        # For models that didn't supports NSP task, manually construct token_type_ids
        else:
            segment = []
            for texts in batch_text:
                # Initialize seg
                if not self.siamese:
                    sent1_len, sent2_len = [len(sent.split(' ')) for sent in texts] # sentence pair
                    # <s> sent1 </s> </s> sent2 </s>
                    seg = [0] + [0] * sent1_len + [0, 1] + [1] * sent2_len + [1]
                else:
                    sent_len = len(texts.split(' ')) # single sentence
                    # <s> sent </s>
                    seg = [0] * (sent_len + 2)
                # Pad or truncate
                if len(seg) < self.max_len: # pad
                    seg += [self.tokenizer.pad_token_id] * (self.max_len - len(seg))
                elif len(seg) > self.max_len: # truncate
                    seg = seg[:self.max_len]
                segment.append(seg)

            segment = torch.tensor(segment)

        return token, segment, mask

    def __call__(self, batch):
        if not self.siamese:
            batch_text = [item[0] for item in batch]
            token, segment, mask = self.tokenize_text(batch_text)
        
        else:
            batch_sent1 = [item[0][0] for item in batch]
            batch_sent2 = [item[0][1] for item in batch]
            tok1, seg1, mask1 = self.tokenize_text(batch_sent1)
            tok2, seg2, mask2 = self.tokenize_text(batch_sent2)
            token, segment, mask = (tok1, tok2), (seg1, seg2), (mask1, mask2)

        batch_label = [item[1] for item in batch]
        label = torch.tensor(batch_label)

        return token, segment, mask, label
