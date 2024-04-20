import stanza
import random
import torch
import argparse
import logging
import os
import nltk
import csv
import spacy
import sys
import numpy as np
from tqdm import trange
from utils_glue import processors

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("msg")

class dependency_tree:
    """
        Dependency Tree.
        Parameters:

        * `id`- token id of this node.
        * `text`- token text of this node.
        * `children`- children list of this node, containing tokens that point at this token in the dependency tree.
    """
    def __init__(self, token_id, token_text, children=[]):
        self.token_id = token_id
        self.token_text = token_text
        self.children = children

    def add(self, word):
        self.children.append(dependency_tree(word.id, word.text))


def set_seed(args):
    """
        set random seed.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def is_root(word):
    if word == "root":
        return 'ROOT'
    else:
        return word


def random_prune_with_ratio(doc, rate):
    def random_select_tokens(tokens, ratio):
        ids = range(0, len(tokens))
        pruned_index = sorted(random.sample(ids, int(len(ids) * rate)), reverse=True)
        for index in pruned_index:
            del tokens[index:index+1]
        return tokens
    words = doc.sentences[0].words
    result = random_select_tokens(words, rate)
    pruned_sent = ""
    for word in result:
        pruned_sent += "{} ".format(word.text)
    return pruned_sent

def build_dependency_tree(doc):
    # print(*[f'id: {word.id}\tword: {is_root(word.text)}\t\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
    level_tree = []
    for word in doc.sentences[0].words:
        if word.head == 0:
            level_tree.append([word])
    level = 0
    count = 1
    total = len(doc.sentences[0].words)
    while True:
        level_nodes = []
        if(count == total):
            break
        for parent in level_tree[level]:
            for word in doc.sentences[0].words:
                if(word.head == parent.id):
                    level_nodes.append(word)
                    count += 1
        level_tree.append(level_nodes)
        level += 1
    return level_tree

def cal_leaf_weights(words, weights):
    leaf_pts = [50 for i in range(len(words)+1)]
    for word in words:
        if word.head != 0:
            leaf_pts[word.head] = 0
    for i in range(len(weights)):
        weights[i] += leaf_pts[i]
    return weights


def POS_weights(pos):
    if pos == "SYM" or pos == "PART" or pos == "PUNCT" or pos == "DET" or pos == "ADP" or pos == "CCONJ" or pos == "INTJ" or pos == "X":# least important in understanding
        return 9
    elif pos == "ADV" or pos == "NUM" or pos == "AUX" or pos == "PRON":
        return 6
    elif pos == "ADJ" or pos == "SCONJ" or pos == "PROPN":
        return 3    
    elif pos == "NOUN" or pos == "VERB":# Most important in understanding
        return 0


def cal_weights(doc, tree):
    words = doc.sentences[0].words
    weights = [0 for i in range(len(words)+1)]
    # Calculate Level weights
    for level in range(len(tree)):
        for i in tree[level]:
            weights[i.id] += 100*(level)
    # Calculate Leaf weights
    weights = cal_leaf_weights(words, weights)
    # Calculate POS weights
    for word in words:
        # print(word.text + "\t" + word.upos)
        weights[word.id] += POS_weights(word.upos)
    return weights

def slim_prune_with_ratio(doc, weights, ratio):
    words = doc.sentences[0].words
    num_to_prune = (int)(ratio*len(words))
    ids_to_drop = []
    for _ in range(num_to_prune):
        max_weight = 0
        max_id = 0
        for i in range(1,len(weights)):
            if weights[i] > max_weight and i not in ids_to_drop:
                max_weight = weights[i]
                max_id = i
        ids_to_drop.append(max_id)
    pruned_sent = ""
    for word in words:
        if word.id not in ids_to_drop:
            pruned_sent += "{} ".format(word.text)
    
    # for i in range(1,len(weights)):
        # print(words[i-1].text + ": {}".format(weights[i]) if i not in ids_to_drop else words[i-1].text + ": {}\t√".format(weights[i]))
    return pruned_sent


def prune(sentence, args, nlp):
    strategy = args.reduction_strategy
    ratio = args.reduction_ratio
    doc = nlp(sentence)
    if strategy == "slim":
        tree = build_dependency_tree(doc)
        weights = cal_weights(doc, tree)
        pruned_sent = slim_prune_with_ratio(doc, weights, ratio)

    if strategy == "bottom_level":
        tree = build_dependency_tree(doc)
        ids_to_drop = [i.id for i in tree[-1]]
        word_texts = [i.text for i in doc.sentences[0].words]
        pruned_sent = ""
        for i in range(len(word_texts)):
            if i+1 not in ids_to_drop:
                pruned_sent += "{} ".format(word_texts[i])
    if strategy == "random":
        pruned_sent = random_prune_with_ratio(doc, ratio)

    return pruned_sent


def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


def prune_dev_file(args):
    # Read original file.
    read_file_path = os.path.join(args.data_dir,'dev.tsv')
    write_dir = os.path.join(args.data_dir,'../../pruned/{}/{}/{}'.format(args.reduction_ratio, args.reduction_strategy, args.task_name))
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    lines = _read_tsv(read_file_path)
    
    # Prune the sentences in the original file.
    nlp = stanza.Pipeline('en', download_method=None, processors={'tokenize': 'spacy'})
    if args.task_name == "sst-2":
        for i in trange(1,len(lines),desc="{} - {} - {}".format(args.task_name, args.reduction_strategy, args.reduction_ratio)):
            lines[i][0] = prune(lines[i][0], args, nlp)
    elif args.task_name == "cola":
        for i in trange(1,len(lines),desc="{} - {} - {}".format(args.task_name, args.reduction_strategy, args.reduction_ratio)):
            lines[i][3] = prune(lines[i][3], args, nlp)
    elif args.task_name == "mrpc":
        for i in trange(1,len(lines),desc="{} - {} - {}".format(args.task_name, args.reduction_strategy, args.reduction_ratio)):
            lines[i][3] = prune(lines[i][3], args, nlp)
            lines[i][4] = prune(lines[i][4], args, nlp)
    elif args.task_name == "qnli":
        for i in trange(1,len(lines),desc="{} - {} - {}".format(args.task_name, args.reduction_strategy, args.reduction_ratio)):
            lines[i][1] = prune(lines[i][1], args, nlp)
            lines[i][2] = prune(lines[i][2], args, nlp)
    elif args.task_name == "qqp":
        for i in trange(1,len(lines),desc="{} - {} - {}".format(args.task_name, args.reduction_strategy, args.reduction_ratio)):
            lines[i][3] = prune(lines[i][3], args, nlp)
            lines[i][4] = prune(lines[i][4], args, nlp)
    elif args.task_name == "rte":
        for i in trange(1,len(lines),desc="{} - {} - {}".format(args.task_name, args.reduction_strategy, args.reduction_ratio)):
            lines[i][1] = prune(lines[i][1], args, nlp)
            lines[i][2] = prune(lines[i][2], args, nlp)
    elif args.task_name == "sts-b":
        for i in trange(1,len(lines),desc="{} - {} - {}".format(args.task_name, args.reduction_strategy, args.reduction_ratio)):
            lines[i][7] = prune(lines[i][7], args, nlp)
            lines[i][8] = prune(lines[i][8], args, nlp)
    elif args.task_name == "wnli":
        for i in trange(1,len(lines),desc="{} - {} - {}".format(args.task_name, args.reduction_strategy, args.reduction_ratio)):
            lines[i][1] = prune(lines[i][1], args, nlp)
            lines[i][2] = prune(lines[i][2], args, nlp)
    else:
        raise KeyError("Task name not found! {}".format(args.task_name))
        
    # Write the pruned file.
    with open(os.path.join(write_dir, 'dev.tsv'), 'w', encoding="utf-8", newline='') as f:
        f_writer = csv.writer(f, delimiter='\t')
        f_writer.writerows(lines)
    f.close()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    

    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--reduction_strategy", type=str, default="slim",
                        help="Strategy of reduction, e.g. slim/random/bottom_level.")
    parser.add_argument("--reduction_ratio", type=float, default=0.3,
                        help="Random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))         

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
        torch.cuda.set_device("cuda:0")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    
    # Set seed.
    set_seed(args)

    # Prune the dev data.
    prune_dev_file(args)

    # nlp = stanza.Pipeline('en', download_method=None, processors={'tokenize': 'spacy'})
    # sentence = "It is found throughout Central and South America, with a maximum range extending north to Minnesota and south to Tierra del Fuego."
    # pruned_sent = prune(sentence, args, nlp)
    # print(pruned_sent)

if __name__ == "__main__":
    main()