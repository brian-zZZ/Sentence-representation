import os
import time
import datetime
import json
import argparse
import numpy as np
from typing import List
from tqdm import tqdm
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import WordVocab, TextDataset
from utils import EarlyStopping, TensorboardLogger, WandbLogger


class Trainer:
    def __init__(self, args, model, bert_flag=False):
        self.args = args
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        torch.backends.cudnn.benchmark = True

        # Prepare data and optimizer for non-bert models
        if not bert_flag:
            vocab = WordVocab.load_vocab(args.vocab_path)
            args.vocab_size = len(vocab)
            args.vocab = vocab
            train_dataset = TextDataset(os.path.join(args.data_dir, "sts-train-" + \
                                        ('subword' if args.word_type=='subword' else 'word') + ".ps"),
                                        vocab, args.max_len, 'train')
            self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=os.cpu_count())
            valid_dataset = TextDataset(os.path.join(args.data_dir, "sts-dev-" + \
                                        ('subword' if args.word_type=='subword' else 'word') + ".ps"),
                                        vocab, args.max_len, 'dev')
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=os.cpu_count())
            test_dataset = TextDataset(os.path.join(args.data_dir, "sts-test-" + \
                                       ('subword' if args.word_type=='subword' else 'word') + ".ps"),
                                       vocab, args.max_len, 'test')
            self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=os.cpu_count())
        # Build regression model
        self.model = model(args)
        if torch.cuda.is_available():
            self.model.cuda()
        # Config optimizer for non-bert models
        if not bert_flag:
            self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Define loss function
        self.loss_func = nn.MSELoss()

        # Initialize the Best model to save
        self.best_model = self.model
        # Initialize early-stopping monitor
        self.early_stopping = EarlyStopping(mode=args.mode, patience=args.patience, verbose=args.verbose)
        # Build loggers
        self.local_logger = open(os.path.join(args.local_log,
                                 f"log-{args.project}-{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt"),
                                 mode="a", encoding="utf-8") if args.local_log else None
        self.tb_logger = TensorboardLogger(args.tb_log) if args.enable_tb else None
        self.wandb_logger = WandbLogger(args) if args.enable_wandb else None

    def epoch_iterator(self, mode='train', epoch=None):
        if mode == 'train':
            self.model.train()
            dataloader = self.train_dataloader
            # Wrap the training dataloader with tqdm to visualize the training procedure
            dataloader = tqdm(dataloader, desc="Training epoch {epoch}".format(epoch=epoch))
        else:
            if mode == 'dev':
                dataloader = self.valid_dataloader
            else: # 'test'
                dataloader = self.test_dataloader
                # self.model.load_state_dict(torch.load(self.args.save_path))
            self.model.eval()

        total_loss = 0
        predict_all = np.array([], dtype=float)
        labels_all = np.array([], dtype=float)
        for _, batch_dict in enumerate(dataloader):
            # Get batch data
            batch_dict = {key: value.cuda() if torch.cuda.is_available() else value.cpu() \
                          for key, value in batch_dict.items()}
            sent_pair = batch_dict['sent_pair']
            label = batch_dict['label']

            # Forward pass & loss computation
            self.model.zero_grad()
            out = self.model(sent_pair).squeeze(-1)
            loss = self.loss_func(out, label)
            total_loss += loss.detach().item()

            # Backward pass & update
            if mode == 'train':
                loss.backward()
                self.optimizer.step()
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

    def train(self, epoch=True):
        """ Train one epoch. """
        return self.epoch_iterator(mode='train', epoch=epoch)

    def dev(self):
        """ Validate one epoch. """
        return self.epoch_iterator(mode='dev')

    def test(self):
        """ Test one epoch. """
        return self.epoch_iterator(mode='test')

    def runner(self):
        """ """
        total_time0 = time.time()
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        for epoch in range(self.args.epoch):
            epoch_time0 = time.time()

            # Training & validating
            train_stats = self.train(epoch)
            dev_stats = self.dev()
            train_spr, train_loss = train_stats['spr'], train_stats['loss']
            dev_spr, dev_loss = dev_stats['spr'], train_stats['loss']
            print("Epoch %03d: spearmanr = %.4f mse = %.4f cost time %.2f secs" % (epoch, train_spr,
                   train_loss, time.time() - epoch_time0))
            print("-> Valid:  spearmanr = %.4f mse = %.4f" % (dev_spr, dev_loss))
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'dev_{k}': v for k, v in dev_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            # Local statistic logging
            if self.local_logger:
                self.local_logger.write(json.dumps(log_stats) + "\n")
            # Tensorboard logging
            if self.tb_logger:
                self.tb_logger.update(train_spr=train_spr, head="spr")
                self.tb_logger.update(train_loss=train_loss, head="loss")
                self.tb_logger.update(dev_spr=dev_spr, head="spr")
                self.tb_logger.update(dev_loss=dev_loss, head="loss")
                self.tb_logger.set_step()
                self.tb_logger.flush()
            # W&B logging
            if self.wandb_logger:
                self.wandb_logger.log_epoch_metrics(log_stats)

            # Performance estimation
            self.early_stopping(dev_spr)
            if self.early_stopping.saving_state:
                self.best_model = self.model
            if self.early_stopping.early_stop:
                print(f"Early stopping after {self.args.patience} epochs training without improvement.")
                break
        
        # Save only the best model finally to reduce saving IO cost
        print("===== " * 5)
        torch.save(self.best_model.state_dict(), self.args.save_path)
        
        # Testing
        self.model = self.best_model
        test_stats = self.test()
        test_spr, test_loss = test_stats.values()
        print("Test: spearmanr = %.4f mse = %.4f" % (test_spr, test_loss))
        # Local logging
        if self.local_logger:
            test_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
            self.local_logger.write(json.dumps(test_stats) + "\n")
            self.local_logger.close()
        # Tensorboard logging
        if self.tb_logger:
            self.tb_logger.update(test_spr=test_spr, head="test", step=epoch)
            self.tb_logger.update(test_loss=test_loss, head="test", step=epoch)
            self.tb_logger.flush()
        # W&B logging
        if self.wandb_logger:
            self.wandb_logger._wandb.log({"Global Test/test_spr": test_spr,
                                          "Global Test/test_loss": test_loss}, commit=False)
            self.wandb_logger.log_checkpoints()
            # self.wandb_logger._wandb.finish()
        
        total_time_str = str(datetime.timedelta(seconds=int(time.time() - total_time0)))
        print('Total time: {}'.format(total_time_str))
