import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer

from data import BertTextDataset, BatchTextCall
from models import BERTs
from engine import Trainer
from utils import basic_parser


transformers.logging.set_verbosity_error()

Huggingface_PT_Online_Hub = {# BERT: supports NSP
                              "bert": "bert-base-uncased",
                              "bert_nli": "sentence-transformers/bert-base-nli-mean-tokens",
                              "bert_simcse": "princeton-nlp/sup-simcse-bert-base-uncased",
                              # ALBERT: supports NSP
                              "albert": "albert-base-v2",
                              # RoBERTa
                              # "roberta": "roberta-base",
                              # "roberta_nli": "sentence-transformers/roberta-base-nli-mean-tokens",
                              "roberta_simcse": "princeton-nlp/sup-simcse-roberta-base",
                              # XLNet
                              # "xlnet": "xlnet-base-cased",
                              # XLM-RoBERTa
                              # "xlmr": "xlm-roberta-base",
                              # BART
                              # "bart": "facebook/bart-base",
                              }

PreTrained_Local_BaseDir = './huggingface_pretrained/' # specify to the local directory
HUggingface_PT_Local_Hub = {k: PreTrained_Local_BaseDir+v for k, v in Huggingface_PT_Online_Hub.items()}


class BertTrainer(Trainer):
    """ Inherited from Trainer, build BERT-specific Trainer. """
    def __init__(self, args, model, bert_flag=True):
        super(BertTrainer, self).__init__(args, model, bert_flag)
        # Prepare data for bert models
        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
        dataset_call = BatchTextCall(tokenizer, max_len=args.max_len,
                                     bert_type=args.bert_type, siamese=args.siamese)
        # Generate dataloaders
        train_dataset = BertTextDataset(os.path.join(args.data_dir, "sts-train.csv"), "train")
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=os.cpu_count(), collate_fn=dataset_call)
        valid_dataset = BertTextDataset(os.path.join(args.data_dir, "sts-dev.csv"), "dev")
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=os.cpu_count(), collate_fn=dataset_call)
        test_dataset = BertTextDataset(os.path.join(args.data_dir, "sts-test.csv"), "test")
        self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=os.cpu_count(), collate_fn=dataset_call)

        # Config optimizer for bert models
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        num_train_optimization_steps = len(self.train_dataloader) * args.epoch
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                            int(num_train_optimization_steps * args.warmup_proportion),
                                            num_train_optimization_steps)

    def epoch_iterator(self, mode='train', epoch=None):
        """ A rewritten epoch-iterator for bert models. """
        if mode == 'train':
            self.model.train()
            dataloader = self.train_dataloader
            # Wrap the training dataloader with tqdm to visualize the training procedure
            dataloader = tqdm(dataloader, desc="Training epoch {epoch}".format(epoch=epoch))
        else :
            if mode == 'dev':
                dataloader = self.valid_dataloader
            else:
                dataloader = self.test_dataloader
                # self.model.load_state_dict(torch.load(self.args.save_path))
            self.model.eval()

        total_loss = 0
        predict_all = np.array([], dtype=float)
        labels_all = np.array([], dtype=float)
        for _, elements in enumerate(dataloader):
            # Get batch data  
            if torch.cuda.is_available():
                token, segment, mask, label = [(e[0].cuda(), e[1].cuda()) if isinstance(e, tuple) \
                                                else e.cuda() for e in elements]
            
            # Forward pass & loss computation
            self.model.zero_grad()
            out = self.model(token, segment, mask).squeeze(-1)
            loss = self.loss_func(out, label)
            total_loss += loss.detach().item()

            # Backward pass & update
            if mode == 'train':
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            label = label.data.cpu().numpy()
            predict = out.data.cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predict)
        spr, _ = spearmanr(labels_all, predict_all)
        
        # Calculate statastic metrics
        stats =  {'spr': spr,
                  'loss': total_loss / len(dataloader)}
        return stats

def args_parser():
    model_name = 'BERTs'
    parser = basic_parser()
    # Model config
    parser.add_argument("--local_or_online", type=str, default='local', choices=['local', 'online'])
    parser.add_argument("--bert_type", type=str, default="albert",
                                       choices=["bert", "bert_nli", "bert_simcse", "albert", "roberta_simcse"])
    parser.add_argument("--siamese", action='store_true', default=False)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--pooling_strategy", type=str, default='mean', choices=['cls', 'mean', 'max'])
    parser.add_argument("--final_dim", type=int, default=128, help='The final dim of fc layer if siamese.')
    # Training config
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=15)
    # Local log
    parser.add_argument("--local_log", type=str, default=f"./logs/{model_name}/stats_logs",
                                       help='Set None to suppress local logging')
    # Wandb config
    parser.add_argument("--enable_wandb", default=False, action='store_true')
    parser.add_argument("--project", type=str, default=model_name)
    parser.add_argument('--wandb_ckpt', type=str, default=f"./logs/{model_name}/wandb")
    # Tensorboard config
    parser.add_argument("--enable_tb", default=False, action='store_true')
    parser.add_argument("--tb_log", type=str, default=f"./logs/{model_name}/tb_logs")
    
    args = parser.parse_args()
    return args

def main(args):
    # Refine args
    args.pretrained_path = Huggingface_PT_Online_Hub[args.bert_type] if args.local_or_online=='online' \
                           else HUggingface_PT_Local_Hub[args.bert_type]
    
    siamese_str = '_siamese' if args.siamese else ''
    specification = args.bert_type + siamese_str + f'_{args.pooling_strategy}' + f'_{args.word_type}'
    args.project += '_' + specification

    args.save_path = os.path.join(args.save_path_base, args.project+'.ckpt')
    specific_path = []
    for pth in [args.local_log, args.wandb_ckpt, args.tb_log]:
        basename = os.path.basename(pth)
        prefix_path = pth.split(basename)[0]
        pth = os.path.join(prefix_path, specification, basename)
        specific_path.append(pth)
    args.local_log, args.wandb_ckpt, args.tb_log = specific_path

    # Enlarge max_len if more GPU-Memory is available to obtain better performance
    if not args.siamese:
        if args.batch_size <= 128:
            args.max_len = 512
    else:
        if args.batch_size <= 128:
            args.max_len = 256
    # Decrease patience of early-stopping if epoch > 10 to prevent overfitting
    if args.epoch > 10:
        args.patience = 2

    # Make directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.local_log, exist_ok=True) if args.local_log else None
    os.makedirs(args.wandb_ckpt, exist_ok=True) if args.enable_wandb else None
    os.makedirs(args.tb_log, exist_ok=True) if args.enable_tb else None
    print("Config: ", args)

    trainer = BertTrainer(args, BERTs, bert_flag=True)
    trainer.runner()


if __name__ == "__main__":
    args = args_parser()
    main(args)
