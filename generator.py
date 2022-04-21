import os
import sys
import csv
import json
import gzip
import random
import itertools

from tqdm import tqdm
from typing import *

from nltk import PCFG, Tree
from nltk import nonterminals, Nonterminal, Production

def generate(grammar: PCFG, start: str = None, depth: int = None) -> Iterator:
    """
    Generates an iterator of all sentences from a CFG.

    :param grammar: The Grammar used to generate sentences.
    :param start: The Nonterminal from which to start generate sentences.
    :param depth: The maximal depth of the generated tree.
    :param n: The maximum number of sentences to return.

    :return: An iterator of lists of terminal tokens.
    """
    start = grammar.start() if not start else start
    depth = sys.maxsize if depth is None else depth
    items = [start]
    tree  = _generate(grammar,items, depth)
    return tree[0]

def _generate(grammar: PCFG, items: List[str], depth: int = None):
    '''
    Generates a sentence from the passed grammar.
    
    :param grammar: the grammar used to generate a sentence
    :param items: the starting node
    :param depth: the maximum tree depth
    
    :return result: a sentence as a nested list of nodes
    '''
    if depth > 0:
        result = []
        for i in items:
            p = random.random()
            total_rule_prob = 0.
            if isinstance(i, Nonterminal):
                for prod in grammar.productions(lhs=i):
                    total_rule_prob += prod.prob()
                    if p < total_rule_prob:
                        expansion = _generate(grammar, prod.rhs(), depth - 1)
                        result += [Tree(i, expansion)]
                        break
            else:
                result += [i]
                break
        
        return result

def create_csv_file(filename: str, grammar: PCFG, ex_generator: Callable, n: int = 10) -> None:
    '''
    Creates a csv file containing paired sentences from the grammar for Seq2Seq models
    
    :param filename: the name of the output file
    :param grammar: the grammar used to generate sentences
    :param ex_generator: a function used to generate pairs of sentences with tags for a Seq2Seq language model
    :param n: the number of sentences to generate
    '''
    filename = filename + '.csv' if not filename.endswith('.csv') else filename
    
    with open(filename, 'w') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar=' ')
        output_writer.writerow(['SRC', 'TRG'])
        for _ in range(n):
            (pos), src, (neg), targ = ex_generator(grammar)
            output_writer.writerow([(pos) + src + ' ' + (neg) + targ])

def create_data_path(d: str) -> None:
    '''
    Creates a path if one does not exist. Treats final split as file prefix.
    '''
    split_d = os.path.split(d)
    if len(split_d) > 1:
        if not os.path.isdir(split_d[0]):
            print(f'Creating directory @ "{split_d[0]}"')
            os.makedirs(split_d[0])

def create_dataset_json(
    grammar: PCFG, 
    ex_generator: Callable, 
    file_prefix: str = '', 
    **splits: Dict[str,int]
) -> None:
    """
    Create a dataset json file that can be read using the datasets module's dataset loader.
    :param grammar: PCFG: a PCFG object
    :param ex_generator: function: a function that creates a pair of sentences and associated tags from the grammar
    :param file_prefix: str: an identifier to add to the beginning of the output file names
    :param splits: kwargs mapping a set label to the number of examples to generate for that set
                   ex: train=10000, dev=1000, test=10000
    """
    file_prefix = file_prefix + '_' if file_prefix and not (file_prefix[-1] in ['-', '_']) else ''
    create_data_path(os.path.join('data', file_prefix))
    
    for name, n_examples in splits.items():
        prefixes = {}
        l = []
        print('Generating examples')
        for n in tqdm(range(n_examples)):
            source, pfx, target = ex_generator(grammar)
            prefixes[pfx] = 1 if not pfx in prefixes else prefixes[pfx] + 1
            l += [{'translation': {'src': source, 'prefix': pfx, 'tgt': target}}]
        
        for pfx in prefixes:
            print(f'{name} prop {pfx} examples: {prefixes[pfx]/n_examples}')
        
        if l:
            print('Saving examples to data/' + file_prefix + name + '.json.gz')
            with gzip.open('data/' + file_prefix + name + '.json.gz', 'wt', encoding='utf-8') as f:
                for ex in tqdm(l):
                    json.dump(ex, f, ensure_ascii=False)
                    f.write('\n')

def combine_dataset_jsons(
    file_prefix: str = '',
    *files: Tuple[str],
) -> None:
    '''
    Combines dataset jsons.
    :param file_prefix: str: a prefix (without extension) to give to the combine file
    :param *files: Tuple[str]: tuple of strings containing the files to combine 
                               (in the order they should be put into the resulting file)
    '''
    create_data_path(os.path.join('data', file_prefix))
    
    combined = ''
    for file in files:
        with gzip.open(os.path.join('data', file + ('.json.gz' if not file.endswith('.json.gz') else '')), 'rt', encoding='utf-8') as in_file:
            combined += in_file.read()
    
    with gzip.open(os.path.join('data', file_prefix + '.json.gz'), 'wt', encoding='utf-8') as f:
        f.write(combined)