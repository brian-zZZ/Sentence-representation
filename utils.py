import argparse
import wandb
import torch
from tensorboardX import SummaryWriter


def basic_parser():
    parser = argparse.ArgumentParser(description='Regression on STS-B')
    # Evironment config
    parser.add_argument("--gpu", type=str, default="0")
    # Dataset config
    parser.add_argument("--data_dir", type=str, default="./STSB/")
    parser.add_argument("--word_type", type=str, default='subword', choices=['subword', 'word'])
    parser.add_argument("--vocab_path_base", type=str, default="./STSB/stsb-vocab")
    parser.add_argument("--max_len", type=int, default=128)
    # Early-stopping config
    parser.add_argument("--mode", type=str, default='max', choices=['max', 'min'])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--verbose", action='store_true', default=False)
    # Saving config
    parser.add_argument("--save_path_base", type=str, default="./ckpt/")

    return parser


class EarlyStopping:
    """Early stop the training if criterion keep getting worse after a given patience."""
    def __init__(self, mode='max', patience=7, delta=0, verbose=False):
        """
        Args:
            patience (int): Number of epochs for stop to wait since last criterion improved.
                            Default: 7.
            mode (str):     One of `min`, `max`. In `min` mode, early stop will occur when the
                            quantity monitored has stopped increasing.
                            Default: 'max'.
            verbose (bool): If True, when ceriterion is getting worse, show the patience message. 
                            Default: False.
            delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.
        """
        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_criterion = None
        self.saving_state = False
        self.early_stop = False

    def __call__(self, criterion):
        if self.mode == 'min':
            criterion_cur = -criterion
            def meth(best_criterion, delta):
                return best_criterion + delta
        elif self.mode == 'max':
            criterion_cur = criterion
            def meth(best_criterion, delta):
                return best_criterion - delta
        else:
            raise Exception("Mode must be 'min' or 'max', 'max' for default.")

        # Initialization
        if self.best_criterion is None:
            self.best_criterion = criterion_cur
        # Comparasion
        elif criterion_cur < meth(self.best_criterion, self.delta): # Getting worse
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}.')
            # Early stop when conuter exceed
            if self.counter >= self.patience:
                self.early_stop = True
            # Update best model saving state
            self.saving_state = False
        else: # Getting better
            # Update global best criterion
            self.best_criterion = criterion_cur
            # Update best model saving state
            self.saving_state = True
            # Within patience, reset counter
            self.counter = 0


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    def __init__(self, args):
        self.args = args 
        self._wandb = wandb

        # Initialize a W&B run 
        if self._wandb.run is None:
            self._wandb.init(
                project=args.project,
                config=args)

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/dev metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'dev' in k:
                self._wandb.log({f'Global Dev/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.wandb_ckpt
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model")

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Dev/*', step_metric='epoch')
