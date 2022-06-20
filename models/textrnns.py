import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab

from models import gen_embedding


class TextRNNs(nn.Module):
    def __init__(self, args):
        super(TextRNNs, self).__init__()
        num_direct = 2 if args.bidirectional else 1
        num_chains = 2 if args.siamese else 1 # siamese has two chains
        self.siamese = args.siamese

        self.embedding = gen_embedding(args)

        if args.rnn_type == 'rnn':
            RNNs = nn.RNN
        elif args.rnn_type == 'lstm':
            RNNs = nn.LSTM
        elif args.rnn_type == 'gru':
            RNNs = nn.GRU
        else:
            raise Exception(f"RNN type must be rnn, lstm or gru, but {args.rnn_type} is given.")
        self.encoder = RNNs(input_size=args.embed_dim,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            batch_first=True,
                            dropout=args.dropout if args.num_layers > 1 else 0,
                            bidirectional=args.bidirectional)
        self.maxpool = nn.MaxPool1d(args.max_len) if self.siamese else nn.MaxPool1d(args.max_len * 2)
        self.dropout = nn.Dropout(args.dropout)
        self.reg = nn.Linear((args.hidden_size * num_direct + args.embed_dim) * num_chains, 1)

    def single_forward(self, sent):
        embed = self.embedding(sent)  # [batch_size, seq_len, embed_dim]
        out, _ = self.encoder(embed) # [batch_size, seq_len, hidden_size * num_direct]
        out = torch.cat((embed, out), 2) # [batch_size, seq_len, embed_dim + hidden_size * num_direct]
        out = F.relu(out)
        out = self.maxpool(out.permute(0, 2, 1)).squeeze() # [batch_size, embed_dim + hidden_size * num_direct]
        return out

    def siamese_forward(self, x):
        encode_sent1 = self.single_forward(x[:, 0])
        encode_sent2 = self.single_forward(x[:, 1])

        # element-wise absolute difference
        absDiff = torch.abs(encode_sent1 - encode_sent2)
        # element-wise multiplication
        mul = torch.mul(encode_sent1, encode_sent2)
        # concatenation
        x = torch.cat((absDiff, mul), 1) # (batch_size, len(kernel_sizes) * kernel_num * 2)
        
        return x

    def forward(self, x):
        if self.siamese:
            x = self.siamese_forward(x) # [batch_size, 1]
        else:
            x = x.reshape(x.size(0), -1) # [batch_size, seq_len * 2] <- [batch_size, 2, seq_len]
            x = self.single_forward(x) # [batch_size, embed_dim + hidden_size * num_direct]
        x = self.reg(self.dropout(x)) # [batch_size, 1]

        return x
