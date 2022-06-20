import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig


class BERTs(nn.Module):
    def __init__(self, args):
        super(BERTs, self).__init__()
        self.args = args
        final_dim = 1  if not args.siamese else args.final_dim
        model_config = AutoConfig.from_pretrained(args.pretrained_path)

        self.BERT = AutoModel.from_pretrained(args.pretrained_path, config=model_config)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(args.hidden_size, final_dim)
        self.sigmoid = nn.Sigmoid()

    def single_forward(self, batch_token, batch_segment, batch_attention_mask):
        with torch.no_grad():
            # Froze the parameters
            out = self.BERT(batch_token,
                            attention_mask=batch_attention_mask,
                            token_type_ids=batch_segment,
                            output_hidden_states=True)

        if self.args.pooling_strategy == 'cls':
            out = out.last_hidden_state[:, 0, :]  # CLS-token [batch, embed_dim] <- [batch_size, seq_len, embed_dim]
        elif self.args.pooling_strategy == 'mean':
            # out = out.last_hidden_state.mean(dim=1)
            out = ((out.last_hidden_state * batch_attention_mask.unsqueeze(-1)).sum(1) / \
                    batch_attention_mask.sum(-1).unsqueeze(-1))
        elif self.args.pooling_strategy == 'max':
            out = out.last_hidden_state.max(dim=1).values
        
        out = self.fc(self.dropout(out))
        return out

    def forward(self, batch_token, batch_segment, batch_attention_mask):
        if self.args.siamese:
            batch_data = batch_token, batch_segment, batch_attention_mask
            encoded_x1 = self.single_forward(*[ele[0] for ele in batch_data])
            encoded_x2 = self.single_forward(*[ele[1] for ele in batch_data])
            # element-wise absolute difference
            out = torch.cosine_similarity(encoded_x1, encoded_x2, dim=-1)
        else:
            out = self.single_forward(batch_token, batch_segment, batch_attention_mask)
            out = self.sigmoid(out)
        
        return out
