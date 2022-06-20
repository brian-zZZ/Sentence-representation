import os
import csv
import argparse
import pandas as pd
from bpe.learn_bpe import learn_bpe
from bpe.apply_bpe import BPE
from data import WordVocab

def extract_texts(sts_path):
    """ Extract texts from STS raw dataset. Extract texts only.
    """
    # Store the extracted data to path './processed' by default 
    save_texts_path = os.path.join(sts_path, "processed")
    os.makedirs(save_texts_path, exist_ok=True)

    sts_files = []
    for file in os.listdir(sts_path):
        if file.split('.')[-1] == 'csv':
            sts_files.append(file)

    # Extract the train, dev, and test sets respectively
    save_texts_file_dirs = []
    for sts_file in sts_files:
        sts_file_dir = os.path.join(sts_path, sts_file)
        in_file = open(sts_file_dir, 'r', encoding='utf-8')
        in_data = in_file.readlines()

        save_texts_file_dir = os.path.join(save_texts_path, sts_file.split('.')[0]+'.texts')
        save_texts_file = open(save_texts_file_dir, 'w', encoding='utf-8')
        save_texts_file_dirs.append(save_texts_file_dir)
        # Write-in
        texts = []
        for line in in_data:
            text_pair = line.split('\t')[5: 7]
            texts.append(text_pair[0])
            texts.append(text_pair[1])
        texts = [text.strip().lower() + '\n' for text in texts]
        save_texts_file.writelines(texts)

        in_file.close()
        save_texts_file.close()

    # Merge the three plain text data sets generated above into one data set for unified encoding and TF counting
    sts_all_texts = open(os.path.join(save_texts_path, 'sts-all.texts'), 'w', encoding='utf-8')
    all_texts_data = []
    for i, save_texts_file_dir in enumerate(save_texts_file_dirs):
        texts_file = open(save_texts_file_dir, 'r', encoding='utf-8')
        texts_data = texts_file.readlines()
        all_texts_data += texts_data
    sts_all_texts.writelines(all_texts_data)
    sts_all_texts.close()


def bpe(sts, s, v=False):
    """ A wrapped BPE encoder based on source https://github.com/rsennrich/subword-nmt/.
        Atake subword level Byte-pair Encoding to reduce the impact of rare words on performance.
    """
    sts_texts = sts + "processed/sts-all.texts"
    sts_bpe_ref = sts + "processed/sts-bpe.ref"

    sts_texts_file = open(sts_texts, 'r', encoding='utf-8')
    sts_bpe_ref_file = open(sts_bpe_ref, 'w', encoding='utf-8')
    learn_bpe(infile=sts_texts_file, outfile=sts_bpe_ref_file,
              num_symbols=s, verbose=v, num_workers=os.cpu_count())
    sts_texts_file.close()
    sts_bpe_ref_file.close()

    sts_bpe_ref_file = open(sts_bpe_ref, 'r', encoding='utf-8')
    Bpe = BPE(codes=sts_bpe_ref_file)
    for split_set in ["train", "dev", "test", "all"]:
        split_texts_dir = sts + "processed/sts-" + split_set + ".texts"
        split_bpe_file = open(sts+"processed/sts-"+split_set+".bpe", 'w', encoding='utf-8')
        Bpe.process_lines(filename=split_texts_dir, outfile=split_bpe_file, num_workers=os.cpu_count())
        print(split_set, "-set byte-pair encoding done!")
        split_bpe_file.close()
    sts_bpe_ref_file.close()


def wrap_pair_score(sts, word_type):
    """ Wrap a text pair and the corresponding score. These data will be used as input to tokenizer.
    """
    for split_set in ["train", "dev", "test"]:
        split_src_file = open(sts+"sts-"+split_set+".csv", 'r', encoding='utf-8')
        src_data = split_src_file.readlines()
        split_src_file.close()
        split_sents_file = open(sts+"processed/sts-"+split_set+(".bpe" if word_type=='subword' else '.texts'),
                              'r', encoding='utf-8')
        bpes_data = split_sents_file.readlines()
        split_sents_file.close()

        pair_score_list = []
        split_pair_score = open(sts+"sts-"+split_set+('-subword' if word_type=='subword' else '-word')+".ps",
                                'w', encoding='utf-8')
        for i in range(0, len(bpes_data), 2):
            pair_score = bpes_data[i].strip() + '\t' + bpes_data[i+1].strip() + '\t' \
                            + src_data[i//2].split('\t')[4] + '\n'
            pair_score_list.append(pair_score)
        split_pair_score.writelines(pair_score_list)
        split_pair_score.close()


def args_parser():
    parser = argparse.ArgumentParser("Data preprocessor")
    parser.add_argument("-d", "--dataset_path", type=str, default="./STSB/")
    parser.add_argument("-t", "--word_type", type=str, default="word", choices=["subword", "word"],
                                             help="subword: bpe, word: original whole word.")
    parser.add_argument("-c", "--corpus_path", type=str, default="./STSB/processed/sts-all")
    parser.add_argument("-o", "--vocab_path", type=str, default="./STSB/stsb-vocab")
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()
    return args


def main(args):
    if args.word_type == 'subword':
        args.corpus_path += '.bpe'
        args.vocab_path += '-subword.pkl'
    elif args.word_type == 'word':
        args.corpus_path += '.texts'
        args.vocab_path += '-word.pkl'

    # Extract plain texts from source dataset
    extract_texts(args.dataset_path)

    if args.word_type == 'subword':
        # Byte-pair Encoding
        bpe(args.dataset_path, 10000, False)

    # Wrap text pair and score
    wrap_pair_score(args.dataset_path, args.word_type)

    # Build vocab
    with open(args.corpus_path, "r", encoding='utf-8') as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)
    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.vocab_path)


if __name__ == '__main__':
    args = args_parser()
    main(args)
