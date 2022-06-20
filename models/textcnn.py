import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedders import gen_embedding


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        num_chains = 2 if args.siamese else 1 # siamese has two chains
        self.siamese = args.siamese
        
        self.embed = gen_embedding(args)
        
        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (ks, args.embed_dim)) for ks in args.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.reg = nn.Linear(len(args.kernel_sizes) * args.kernel_num * num_chains, 1, bias=True)

    def single_forward(self, x):
        x = self.embed(x) # (batch_size, seq_len, embed_dim)
        x = x.unsqueeze(1) # (batch_size, 1, seq_len, embed_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, kernel_num, seq_len), ...] * len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch_size, kernel_num), ...] * len(kernel_sizes)
        x = torch.cat(x, 1)
        return x

    def siamese_forward(self, x):
        encode_x1 = self.single_forward(x[:, 0])
        encode_x2 = self.single_forward(x[:, 1])

        # element-wise absolute difference
        absDiff = torch.abs(encode_x1 - encode_x2)
        # element-wise multiplication
        mul = torch.mul(encode_x1, encode_x2)
        # concatenation
        x = torch.cat((absDiff, mul), 1) # (batch_size, len(kernel_sizes) * kernel_num * 2)

        return x

    def forward(self, x):
        if self.siamese:
            x = self.siamese_forward(x)
        else:
            x = x.reshape(x.shape[0], -1)
            x = self.single_forward(x)
        x = self.reg(self.dropout(x))
        
        return x
