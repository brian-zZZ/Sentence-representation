import os
import torch
import torch.nn as nn
import numpy as np
import torchtext.vocab as torchvocab
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


class Embedders(nn.Module):
    def __init__(self, args):
        super(Embedders, self).__init__()
        self.args = args
        self.embedding = gen_embedding(args)
        # self.maxpool = nn.MaxPool1d(args.max_len * 2)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.embed_dim, args.proj_dim) if not args.not_siamese else nn.Linear(args.embed_dim, 1)

    def single_forward(self, x):
        x = self.embedding(x) # [batch_size, 2 * seq_len, emb_dim]

        if self.args.pooling_strategy == 'max':
            # x = self.maxpool(x.permute(0, 2, 1)).squeeze() # [batch_size, emb_dim]
            x = torch.max(x, dim=1).values
        elif self.args.pooling_strategy == 'mean':
            x = torch.mean(x, dim=1) # avg-embedding [batch_size, embed_dim]

        return x

    def forward(self, x):
        if not self.args.not_siamese: # siamese by default
            embed_sent1 = self.single_forward(x[:, 0, :])
            embed_sent2 = self.single_forward(x[:, 1, :])
            embed_sent1, embed_sent2 = self.fc(embed_sent1), self.fc(embed_sent2)
            x = torch.cosine_similarity(embed_sent1, embed_sent2, dim=-1)
        else:
            # merge 2-sent into a sent-pair
            x = x.reshape(x.shape[0], -1) # [batch_size, 2 * seq_len] <- [batch_size, 2, seq_len]
            x = self.single_forward(x)
            x = self.fc(self.dropout(x)) # [batch_size, 1]

        return x
    
def gen_embedding(args):
    """ Generate specific type of embedding. """
    # If without pre-trained embedding, return a initial embedding
    if args.embedder_type == 'None':
        embedding = nn.Embedding(args.vocab_size, args.embed_dim, padding_idx=0)

    # Otherwise, load the pre-trained embedding
    # Word2Vec: load directly or converted from glove
    elif args.embedder_type == 'word2vec':
        # If exists a pre-trained word2vec embedding vectors, load directly
        if not args.convert and os.path.exists(args.word2vec_pretrained):
            print(f"Loading pre-trained {args.embedder_type} embedding vectors")
            weight = torch.tensor(np.load(args.word2vec_pretrained))

        # Else if exists a glove pre-trained word2vec embedding, convert it, then load it
        else:
            # Convert
            word2vec_vectors_pth = args.glove_txt_pth + '.w2v'
            if os.path.exists(word2vec_vectors_pth):
                print("Found existing Word2Vec embedding vectors")
            else:
                print("Converting GloVe embedding vectors to Word2Vec embedding vectors...")
                glove2word2vec(args.glove_txt_pth, args.glove_txt_pth+'.w2v')
            print(f"Loading pre-trained {args.embedder_type} embedding")
            wv = KeyedVectors.load_word2vec_format(args.glove_txt_pth+'.w2v', binary=False)
            # Load and resize (i.e. decrease vocab size) pre-trained embedding
            glove = torchvocab.GloVe(name=args.token_scale, dim=args.embed_dim, cache=args.glove_pretrained)
            weight = torch.zeros(args.vocab_size, args.embed_dim)
            for i in range(glove.vectors.size(0)):
                try:
                    idx = args.vocab.stoi[wv.index_to_key[i]] # word2vec vocab idx to my vocab idx
                    weight[idx, :] = torch.from_numpy(wv.get_vector(args.vocab.itos[idx]).copy())
                except: # tokens in pre-training but don't in my vocab
                    continue
        
        embedding = nn.Embedding.from_pretrained(weight) # same shape as weight
    
    # GloVe: load directly
    else: # args.embedder_type == 'glove'
        # Load and resize (i.e. decrease vocab size) pre-trained embedding
        print(f"Loading pre-trained {args.embedder_type} embedding")
        glove = torchvocab.GloVe(name=args.token_scale, dim=args.embed_dim, cache=args.glove_pretrained)
        weight = torch.zeros(args.vocab_size, args.embed_dim)
        for i in range(glove.vectors.size(0)):
            try:
                idx = args.vocab.stoi[glove.itos[i]] # glove vocab idx to my vocab idx
                weight[idx, :] = glove.get_vecs_by_tokens(args.vocab.itos[idx])
            except: # tokens in pre-training but don't in my vocab
                continue
        embedding = nn.Embedding.from_pretrained(weight) # same shape as weight
            
        # Or keep vocab size unchanged. Higher cost, lower accuracy.
        # embedding = nn.Embedding(glove.vectors.size(0), glove.vectors.size(1))
        # embedding.weight.data.copy_(glove.vectors)

    return embedding
