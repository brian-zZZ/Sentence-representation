import os
from typing import List

from engine import Trainer
from utils import basic_parser
from models import Embedders

def args_parser():
    model_name = 'Embedders'
    parser = basic_parser()
    # Embedding config
    parser.add_argument("--glove_pretrained", type=str, default="./pretrained_vectors/glove.6B")
    parser.add_argument("--word2vec_pretrained", type=str,
                            default="./pretrained_vectors/word2vec/word2vec-google-news-300.model.vectors.npy")
    parser.add_argument("--embed_dim", type=int, default=300, choices=[50, 100, 200, 300],
                                       help="Default 300 for GloVe and Word2Vec pre-trained word vectors.")
    parser.add_argument("--embedder_type", type=str, default='glove', choices=['glove', 'word2vec'],
                                           help='None represents without pre-trained embedding.')
    parser.add_argument("--convert", action='store_true', default=False,
                                     help="Whether or not build Word2Vec embedding by converting GloVe embedding.")
    # Model config
    parser.add_argument("--not_siamese", action='store_true', default=False)
    parser.add_argument("--pooling_strategy", type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int, default=128,
                                      help='Projection dim, i.e. final fc dim of non-siamese model.')
    # Training config
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=100)
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
    # Refine paths
    args.save_path = os.path.join(args.save_path_base, args.project+'.ckpt')
    args.vocab_path = args.vocab_path_base + ('-subword.pkl' if args.word_type=='subword' else '-word.pkl')
    args.glove_txt_pth = os.path.join(args.glove_pretrained,
                                      os.path.basename(args.glove_pretrained)+f'.{args.embed_dim}d'+'.txt')
    args.token_scale = os.path.basename(args.glove_pretrained).split('.')[1] # '6B'

    siamese_str = '_siamese' if not args.not_siamese else ''
    convert_str = '_convert' if (args.embedder_type=='word2vec' and args.convert) else ''
    specification = args.embedder_type + convert_str + siamese_str + f'_{args.pooling_strategy}' + f'_{args.word_type}'
    args.project += '_' + specification

    args.save_path = os.path.join(args.save_path_base, args.project+'.ckpt')
    specific_path = []
    for pth in [args.local_log, args.wandb_ckpt, args.tb_log]:
        basename = os.path.basename(pth)
        prefix_path = pth.split(basename)[0]
        pth = os.path.join(prefix_path, specification, basename)
        specific_path.append(pth)
    args.local_log, args.wandb_ckpt, args.tb_log = specific_path

    # Set higher patience for training from scratch
    args.patience = 30

    # Make directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.local_log, exist_ok=True) if args.local_log else None
    os.makedirs(args.wandb_ckpt, exist_ok=True) if args.enable_wandb else None
    os.makedirs(args.tb_log, exist_ok=True) if args.enable_tb else None
    print("Config: ", args)

    trainer = Trainer(args, Embedders)
    trainer.runner()


if __name__ == "__main__":
    args = args_parser()
    main(args)
