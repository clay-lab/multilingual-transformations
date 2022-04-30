import re
import os
import sys
import csv
import json
import gzip
import random
import shutil
import itertools

from tqdm import tqdm
from typing import *
from functools import partial
from itertools import permutations

from nltk import PCFG, Tree
from nltk import nonterminals, Nonterminal, Production

def generate(
	grammar: PCFG, 
	start: str = None, 
	depth: int = None
) -> Iterator:
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

def _generate(
	grammar: PCFG, 
	items: List[str], 
	depth: int = None
):
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

def create_csv_file(
	filename: str, 
	grammar: PCFG, 
	ex_generator: Callable, 
	n: int = 10
) -> None:
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
	Creates a path if one does not exist. Treats final split as a file prefix.
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
	overwrite: bool = False,
	**splits: Dict[str,int]
) -> None:
	"""
	Create a dataset json file that can be read using the datasets module's dataset loader.
	:param grammar: PCFG: a PCFG object
	:param ex_generator: function: a function that creates a pair of sentences and associated tags from the grammar
	:param file_prefix: str: an identifier to add to the beginning of the output file names
	:param overwrite: bool: whether to overwrite existing datasets with matching names
	:param splits: kwargs mapping a set label to the number of examples to generate for that set
				   ex: train=10000, dev=1000, test=10000
	"""
	file_prefix = file_prefix + '_' if file_prefix and not (file_prefix[-1] in ['-', '_']) else ''
	create_data_path(os.path.join('data', file_prefix))
	
	for name, n_examples in splits.items():
		if not os.path.exists(os.path.join('data', file_prefix + name + '.json.gz')) or overwrite:
			prefixes = {}
			l = []
			print(f'Generating {name} examples')
			for n in tqdm(range(n_examples)):
				source, pfx, target = ex_generator(grammar)
				prefixes[pfx] = 1 if not pfx in prefixes else prefixes[pfx] + 1
				l += [{'translation': {'src': source, 'prefix': pfx, 'tgt': target}}]
			
			for pfx in prefixes:
				print(f'{name} prop {pfx} examples: {prefixes[pfx]/n_examples}')
			
			if l:
				print('Saving examples to data/' + file_prefix + name + '.json.gz')
				with gzip.open(os.path.join('data', file_prefix + name + '.json.gz'), 'wt', encoding='utf-8') as f:
					for ex in tqdm(l):
						json.dump(ex, f, ensure_ascii=False)
						f.write('\n')
		
			print('')
		else:
			print(f'{name} dataset already exists. Skipping. Use overwrite=True to force regeneration.')

def combine_dataset_jsons(
	file_prefix: str = '',
	*files: Tuple[str],
	overwrite: bool = False,
) -> None:
	'''
	Combines dataset jsons.
	:param file_prefix: str: a prefix (without extension) to give to the combine file
	:param *files: Tuple[str]: tuple of strings containing the files to combine 
							   (in the order they should be put into the resulting file)
	:param overwrite: bool: whether to overwrite existing files
	'''
	if not os.path.exists(os.path.join('data', file_prefix + '.json.gz')) or overwrite:
		create_data_path(os.path.join('data', file_prefix))
		
		combined = ''
		for file in files:
			with gzip.open(os.path.join('data', file + ('.json.gz' if not file.endswith('.json.gz') else '')), 'rt', encoding='utf-8') as in_file:
				combined += in_file.read()
		
		with gzip.open(os.path.join('data', file_prefix + '.json.gz'), 'wt', encoding='utf-8') as f:
			f.write(combined)

def create_negation_datasets(
	configs: Dict[str,List] = None, 
	**kwargs
) -> None:
	'''
	Create json datasets according to the passed configs.
	:params configs: (List[Dict]): This should be in the following format:
								   A dict mapping a language id to a List of arguments.
								   Each list of arguments consists of a Dict mapping str to floats, a PCFG, and an example generator function.
								   The dict maps strings to a list containing a float and a dictionary containing splits.
								   Each float is passed to the ex_generator function, with splits mapping strings to numbers that define how many examples to create for each split
								   	when that float is passed to ex_generator.
								   The PCFG is the grammar from which to generate examples.
								   The example generator function should take the grammar and the probability of generating a negative example as argument.
								   example:
								  		configs = {
								  			'en': [
									  			{
									  				'neg': [
									  					0.5, 
									  					{
									  						'train': 100000, 
									  						'dev': 1000, 
									  						'test': 10000
									  					}
									  				], 
									  				'pos': [
									  					0., 
									  					{
									  						'train': 500
									  					}
									  				]
									  			}, 
									  			english_grammar.not_grammar,  
									  			english_grammar.neg_or_pos
								  			],
								  		 	'de': [
								  		 		{
								  		 			'neg': [
								  		 				0.5, 
								  		 				{
								  		 					'train': 100000, 
								  		 					'dev': 1000, 
								  		 					'test': 10000
								  		 				}
								  		 			], 
								  		 			'pos': [
								  		 				0., 
								  		 				{
								  		 					'train': 500
								  		 				}
								  		 			]
								  		 		}, 
								  		 		german_grammar.nicht_grammar, 
								  		 		german_grammar.neg_or_pos
								  		 	]
								  		 }
								  		This config will create for each split neg_en dataset consisting of approximately 50% positive-negative/positive-positive examples, 
								  			and a pos_en dataset consisting of 100% positive-positive examples, and likewise for german.
	:param kwargs: passed to create_dataset_json
	If no argument is passed, attempt to load the configs from a file ./data/config.json
	'''
	configs = load_configs() if configs is None else configs
	
	for lang in configs:
		print(f'Creating datasets for {lang}')
		prob_map 		= configs[lang][0]
		
		# if we're loading from a file, we have to store these as strings,
		# so we need to import the actual objects
		if isinstance(configs[lang][1],str) and isinstance(configs[lang][2],str):
			module1 		= configs[lang][1].split('.')[0]
			module2 		= configs[lang][2].split('.')[0]
			
			exec(f'import {module1}, {module2}')
			
			grammar 		= eval(configs[lang][1])
			ex_generator 	= eval(configs[lang][2])
		else:
			grammar 		= configs[lang][1]
			ex_generator 	= configs[lang][2]
		
		for dataset_type in prob_map:
			p 			= prob_map[dataset_type][0]
			splits 		= prob_map[dataset_type][1]
			file_prefix = f'{dataset_type}_{lang}/{dataset_type}_{lang}'
			p_ex_generator = partial(ex_generator, neg_p=p)
			create_dataset_json(grammar, p_ex_generator, file_prefix, **kwargs, **splits)
		
		print('')

def combine_language_datasets_for_negation(
	langs: List[str], 
	**kwargs
) -> None:
	'''
	Creates dataset jsons for each len 2 permutation of languages passed.
	:param langs: List[str]: a list of language ids corrseponding to directories in ./data/
							  Each language id must have two directories in data associated with it.
							  One is neg_{lang} and the other is pos_{lang}.
							  The neg_{lang} directories must contain a file named neg_{lang}_train.json.gz.
							  The pos_{lang} directories must contain a file named pos_{lang}_train.json.gz.
	
	:outputs: For each possible two-way permutation of languages in langs:
			  a directory in data named neg_{lang1}_{lang2}, with the following datasets jsons.
			  neg_{lang1}_{lang2}_train.json.gz, containing positive-positive/negative training examples from lang1 
			  	and positive-positive training examples from lang2.
	'''
	langs = list(load_config().keys()) if langs is None else langs
	
	all_pairs = permutations(langs, 2)
	for lang1, lang2 in all_pairs:
		print(f'Creating datasets for {lang1} -> {lang2}')
		dirname 	= f'neg_{lang1}_{lang2}'
		file_prefix = os.path.join(dirname, f'neg_{lang1}_{lang2}_train')
		
		# create the training dataset with pos-neg/pos examples from lang1 and pos-pos examples from lang2
		combine_dataset_jsons(
			file_prefix, 
			os.path.join(f'neg_{lang1}', f'neg_{lang1}_train.json.gz'), 
			os.path.join(f'pos_{lang2}', f'pos_{lang2}_train.json.gz'),
			**kwargs
		)
		
		print('')

def create_and_combine_negation_datasets(
	configs: Dict[str,List] = None, 
	**kwargs
) -> None:
	'''
	Create and then combine negation datasets for each combination of languages in configs.keys().
	
	:param configs: Dict[str,List]: passed to create_negation_datasets
	:param kwargs: passed to create_negation_datasets, 
				   combine_language_datasets_for_negation,
				   and create_mt5_scripts
	 			   (useful to set overwrite=True)
	
	:outputs: see outputs of create_negation_datasets and combine_language_datasets_for_negation.
	'''
	configs = load_config() if configs is None else configs
	
	create_negation_datasets(configs, **kwargs)
	combine_language_datasets_for_negation(list(configs.keys()), **kwargs)
	create_mt5_scripts(list(configs.keys()), **kwargs)

def create_mt5_scripts(
	langs: List[str] = None, 
	overwrite: bool = False
) -> None:
	'''
	Creates finetuning and eval scripts for the passed configs for mt5.
	
	:params langs: (List[str]): a list of language abbreviations with files in the ./data/ directory.
	
	If no argument is passed, attempt to load the language ids from a file ./data/config.json
	'''
	script = '\n'.join([
		'#!/bin/bash\n',
		'#SBATCH --job-name=MT5-base-finetune-neg-[TRAIN-LANG]',
		'#SBATCH --output=joblogs/%x_%j.txt',
		'#SBATCH --nodes=1',
		'#SBATCH --cpus-per-task=1',
		'#SBATCH --mem=30GB',
		'#SBATCH --time=15:00:00',
		'#SBATCH --gpus=v100:1',
		'#SBATCH --partition=gpu',
		'#SBATCH --mail-type=ALL',
		'',
		'clear_hf_local_dataset_cache',
		'',
		'module load CUDA',
		'module load cuDNN',
		'module load miniconda',
		'',
		'source activate /gpfs/loomis/project/frank/ref4/conda_envs/py38',
		'',
		'python models/run_seq2seq.py \\',
		"	--model_name_or_path 'google/mt5-base' \\",
		'	--do_train \\',
		'	--task translation_src_to_tgt \\',
		'	--train_file data/neg_[TRAIN_LANG]/neg_[TRAIN_LANG]_train.json.gz \\',
		'	--validation_file data/neg_[DEV_LANG]/neg_[DEV_LANG]_dev.json.gz \\',
		'	--output_dir outputs/mt5-finetuning-neg-[TRAIN-LANG]-bs128/ \\',
		'	--per_device_train_batch_size=4 \\',
		'	--gradient_accumulation_steps=32 \\',
		'	--per_device_eval_batch_size=16 \\',
		'	--overwrite_output_dir \\',
		'	--predict_with_generate \\',
		'	--num_train_epochs 10.0'
	]) + '\n'
	
	eval_script = script.replace('finetune', 'eval')
	eval_script = eval_script.replace('--time=15:00:00', '--time=04:00:00')
	eval_script = eval_script.replace('--do_train \\', '--do_eval \\\n	--do_learning_curve \\')
	eval_script = eval_script.replace('[DEV_LANG]', '[TEST_LANG]')
	eval_script = re.sub(r'_dev(\.|_)', '_test\\1', eval_script)
	eval_script = eval_script.replace('--per_device_train_batch_size=4', '--per_device_train_batch_size=8')
	eval_script = eval_script.replace('	--gradient_accumulation_steps=32 \\\n', '')
	eval_script = eval_script.replace(
		'	--predict_with_generate \\\n	--num_train_epochs 10.0', 
		'	--predict_with_generate \\'
	)
	
	langs 		= list(load_config().keys()) if langs is None else langs
	all_pairs 	= list(permutations(langs, 2))
	langs 		= [tuple([lang]) for lang in langs] + all_pairs
	
	# create directories if not existant
	os.makedirs(os.path.join('scripts', 'finetune'), exist_ok=True)
	os.makedirs(os.path.join('scripts', 'eval'), exist_ok=True)
	
	# create the scripts for each language and pair of languages
	for lang in langs:
		print(f'Creating scripts for {" -> ".join(lang)}')
		lang_ft_script = script
		lang_ev_script = eval_script
		
		if len(lang) == 1:
			train_lang 		= lang[0]
			dev_lang 		= lang[0]
			train_dash_lang = lang[0]
			test_lang 		= lang[0]
		else:
			train_lang 		= '_'.join(lang)
			dev_lang 		= lang[0]
			train_dash_lang = '-'.join(lang)
			test_lang 		= lang[1]
			lang_ft_script 	= lang_ft_script.replace('--time=15:00:00', '--time=20:00:00')
				
		lang_ft_script = lang_ft_script.replace('[TRAIN_LANG]', train_lang)
		lang_ft_script = lang_ft_script.replace('[DEV_LANG]', dev_lang)
		lang_ft_script = lang_ft_script.replace('[TRAIN-LANG]', train_dash_lang)
		if not os.path.exists(os.path.join('scripts', 'finetune', f'finetune_mt5_neg_{train_lang}_bs128.sh')) or overwrite:
			with open(os.path.join('scripts', 'finetune', f'finetune_mt5_neg_{train_lang}_bs128.sh'), 'wt') as out_file:
				out_file.write(lang_ft_script)
			
		lang_ev_script = lang_ev_script.replace('[TRAIN_LANG]', train_lang)
		lang_ev_script = lang_ev_script.replace('[TEST_LANG]', test_lang)
		lang_ev_script = lang_ev_script.replace('[TRAIN-LANG]', train_dash_lang)
		if not os.path.exists(os.path.join('scripts', 'eval', f'eval_mt5_neg_{train_lang}_bs128.sh')) or overwrite:
			with open(os.path.join('scripts', 'eval', f'eval_mt5_neg_{train_lang}_bs128.sh'), 'wt') as out_file:
				out_file.write(lang_ev_script)
			
		# if we have multiple languages, create a zero-shot version of the eval script
		if len(lang) == 2:
			lang_zs_ev_script 	= eval_script.replace(
				'#SBATCH --job-name=MT5-base-eval-neg-[TRAIN-LANG]',
				'#SBATCH --job-name=MT5-base-eval-neg-[TRAIN-ZS-LANG]-zs'
			)
			train_lang 			= lang[0]
			train_dash_lang 	= lang[0]
			
			lang_zs_ev_script 	= lang_zs_ev_script.replace('[TRAIN_LANG]', train_lang)
			lang_zs_ev_script 	= lang_zs_ev_script.replace('[TEST_LANG]', test_lang)
			lang_zs_ev_script 	= lang_zs_ev_script.replace('[TRAIN-ZS-LANG]', '-'.join(lang))
			lang_zs_ev_script 	= lang_zs_ev_script.replace('[TRAIN-LANG]', train_dash_lang)
			
			if not os.path.exists(os.path.join('scripts', 'eval', f'eval_mt5_neg_{"_".join(lang)}_bs128_zs.sh')) or overwrite:
				with open(os.path.join('scripts', 'eval', f'eval_mt5_neg_{"_".join(lang)}_bs128_zs.sh'), 'wt') as out_file:
					out_file.write(lang_zs_ev_script)
	
def load_config(path: 'str or Pathlike' = None) -> Dict[str,List]:
	'''
	Loads a dataset creation config file from disk.
	
	:param path: str or Pathlike: the path to the config file.
						   If no path is provided, attempt to load
						   ./data/config.json as the config.
	'''
	if path is None:
		path = os.path.join('data', 'config.json')
	
	with open(path, 'rt', encoding='utf-8') as in_file:
		configs = json.load(in_file)
	
	return configs
