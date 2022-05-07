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
from contextlib import suppress

from nltk import PCFG, Tree
from nltk import nonterminals, Nonterminal, Production

GERMAN_MODAL_MAP = {
	'kann': 'können',
	'darf': 'dürfen',
	'muss': 'müssen',
	'soll': 'sollen'
}

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

def format_tree_string(
	t: Tree, 
	lang: str, 
	pfx: str = None
) -> str:
	"""
	Convert a tree to a string.
	:param t: Tree: an NLTK Tree
	:param lang: str: the name of the language that generate the string
	:param pfx: str: whether the sentence is positive or negative. only used for turkish
	:returns str: the flattened version of the tree as a string.
	"""
	t = t.copy(deep=True)
	
	if not lang == 'tu':
		t = ' '.join(t.leaves())
		t = t.strip()
		t = t[0].upper() + t[1:]
		t = t.replace(' , ', ', ')
		t = t.replace('  ', ' ')
		t += '.'
	else:
		# can't due this in header due to a circular import
		from turkish_grammar import vowelharmony, vowelharmony_n
		
		t = ''.join(t.leaves())
		t = vowelharmony(t) if not pfx == 'neg' else vowelharmony_n(t)
		t = (t[0].upper() if not t[0] == 'i' else 'İ') + t[1:] + '.'
		
	return t

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

def get_labels(t: Tree) -> List[str]:
	'''
	Get the labels of an NLTK tree.
	
	:param t: Tree: the tree whose labels to return
	:returns labels: a list of the labels of the Tree as strings,
					 corresponding to the linear order in which they would be printed.
	'''
	labels = [t.label().symbol()]
	for child in t:
		if isinstance(child, Tree):
			labels.extend(get_labels(child))
	
	return labels

def get_pos_labels(t: Tree) -> List[str]:
	'''
	Get the part-of-speech labels from an NLTK tree.
	This returns only the labels for the terminal nodes.
	
	:param t: Tree: the tree whose labels to return
	:returns labels: a list of the labels of the terminal nodes of the tree as strings,
					 corresponding to the linear order in which they would be printed.
	'''
	labels = []
	for child in t:
		if isinstance(child, Tree) and not isinstance(child[0], str):
			labels.extend(get_pos_labels(child))
		elif isinstance(child, str):
			pass
		elif not isinstance(child.label(), str):
			labels.append(child.label().symbol())
		else:
			labels.append(child.label())
	
	return labels

def grep_next_subtree(
	t: Tree,
	expr: str
) -> Tree:
	"""
	Get the next subtree whose label matches the expr.
	:param t: Tree: the tree to search.
	:param expr: a regex to search when searching the tree
	"""
	try:
		subt = next(t.subtrees(filter = lambda x: re.search(expr, x.label().symbol())))
	except StopIteration:
		subt = None
	
	return subt

def get_english_pos_seq(t: Tree) -> str:
	'''Remove unwanted info from English pos tags for comparison purposes and return as a string.'''
	pos_seq = get_pos_labels(t)
	pos_seq = [
		pos_tag
			.replace('Sg', '')
			.replace('Pl', '')
			.replace('Nom', '')
			.replace('Acc', '')
			.replace('TV', 'V')
			.replace('IV', 'V')
			.replace('comma', '')
			.replace('PN', 'N')
			.replace('NTand', '')
		for pos_tag in pos_seq
	]
	pos_seq = '[' + '] ['.join([l for tag in [pos_tag.split() for pos_tag in pos_seq if pos_tag] for l in tag]) + ']'
	
	return pos_seq 

def get_english_example_metadata(
	source: Tree,
	pfx: str,
	target: Tree
) -> Dict:
	"""
	Gets metadata about the passed example, consisting of a seq2seq mapping with a source, prefix, and target.
	:param source: Tree: the source Tree
	:param pfx: str: the task prefix passed to the model
	:param target: the target Tree
	:returns metadata: a dictionary recording the following properties for the example:
					   - transitivity of the main verb (v_trans)
					   - definiteness of main clause subject/object (subj_def, obj_def)
					   - number of main clause subject/object (subj_num, obj_num)
					   - the identity of the main auxiliary (main_aux)
					   - how many adverbial clauses before the main clause
					   - how many adverbial clauses after the main clause
					   - the number of adverbial clauses
					   - the PoS sequence of the source and target
	"""
	source = source.copy(deep=True)
	target = target.copy(deep=True)
	
	metadata = {}
	
	# transitivity of main clause verb
	main_clause 	= grep_next_subtree(source, 'S2')
	main_clause_mp 	= grep_next_subtree(main_clause, 'MP')
	main_clause_vp 	= grep_next_subtree(main_clause_mp, 'VP')
	main_clause_v 	= grep_next_subtree(main_clause, '(IV|TV)')
	metadata.update({'v_trans': 'intransitive' if main_clause_v.label().symbol() == 'IV' else 'transitive'})
	
	# definiteness of main clause subject
	main_clause_subj = grep_next_subtree(main_clause, 'NPNom')
	if main_clause_subj[0].label().symbol() == 'PN':
		metadata.update({'subj_def': 'definite'})
	else:
		main_clause_subj_det = grep_next_subtree(main_clause, 'Det')
		if main_clause_subj_det[0] == 'the':
			metadata.update({'subj_def': 'definite'})
		else:
			metadata.update({'subj_def': 'indefinite'})
	
	# definiteness of main clause object
	if main_clause_v.label().symbol() == 'TV':
		main_clause_obj = grep_next_subtree(main_clause_vp, 'NPAcc')
		main_clause_obj_det = grep_next_subtree(main_clause_obj, 'Det')
		if main_clause_obj_det[0] == 'the':
			metadata.update({'obj_def': 'definite'})
		else:
			metadata.update({'obj_def': 'indefinite'})
	else:
		metadata.update({'obj_def': None})
	
	# number of main clause subject
	if main_clause_subj[0].label().symbol() == 'PN':
		metadata.update({'subj_num': 'sg'})
	else:
		metadata.update({'subj_num': 'sg' if 'NSgNom' in get_pos_labels(main_clause_subj) else 'pl'})
	
	# number of main clause object
	if main_clause_v.label().symbol() == 'TV':
		if 'Sg' in main_clause_obj_det.label().symbol():
			metadata.update({'obj_num': 'sg'})
		else:
			metadata.update({'obj_num': 'pl'})
	else:
		metadata.update({'obj_num': None})
	
	# main auxiliary
	main_clause_m = grep_next_subtree(main_clause_mp, 'M$')
	metadata.update({'main_aux': main_clause_m[0]})
	
	# number of AdvPs before and after main clause
	labels = get_labels(source)
	main_clause_start = labels.index('S2')
	pre_main_advps 	= len([l for i, l in enumerate(labels) if l == 'AdvP' and i < main_clause_start])
	post_main_advps = len([l for i, l in enumerate(labels) if l == 'AdvP' and i > main_clause_start])
	metadata.update({
		'pre_main_advps'	: pre_main_advps, 
		'post_main_advps'	: post_main_advps,
		'total_advps'		: pre_main_advps + post_main_advps
	})
	
	# get pos seq with details suppressed
	# remove null determiners from PoS sequence
	for child in source.subtrees(filter = lambda x: 'DetPl' in x.label().symbol()):
		if child[0] == '':
			child.set_label('')
		
	source_pos_seq = get_english_pos_seq(source)
	metadata.update({'source_pos_seq': source_pos_seq})
	
	if pfx == 'pos':
		metadata.update({'target_pos_seq': source_pos_seq})
	else:
		tgt_main_clause 	= grep_next_subtree(target, 'S2')
		tgt_main_clause_mp 	= grep_next_subtree(tgt_main_clause, 'MP')
		tgt_main_clause_m 	= grep_next_subtree(tgt_main_clause_mp, 'M$')
		
		# remove null determiners
		# we do this here because we can't iterate through the tree as easily once the labels are strings,
		# which is what we set below
		for child in target.subtrees(filter = lambda x: 'DetPl' in x.label().symbol()):
			if child[0] == '':
				child.set_label('')
		
		tgt_main_clause_m.set_label('M Neg')
		tgt_pos_seq 		= get_english_pos_seq(target)
		metadata.update({'target_pos_seq': tgt_pos_seq})
	
	return metadata

def get_german_pos_seq(t: Tree) -> str:
	'''Remove unwanted info from German pos tags for comparison purposes and return as a string.'''
	pos_seq = get_pos_labels(t)
	pos_seq = [re.sub('(Sg|Pl)', '', pos_tag) for pos_tag in pos_seq]
	pos_seq = [re.sub('(Nom|Acc)', '', pos_tag) for pos_tag in pos_seq]
	pos_seq = [re.sub('(TV|IV)', 'V', pos_tag) for pos_tag in pos_seq]
	pos_seq = [re.sub('(3|comma|Pres)', '', pos_tag) for pos_tag in pos_seq]
	pos_seq = [re.sub('(Masc|Fem|Neut)', '', pos_tag) for pos_tag in pos_seq]
	pos_seq = [re.sub('(PN|Pron)', 'N', pos_tag) for pos_tag in pos_seq]
	pos_seq = [re.sub('NTand', 'Conj', pos_tag) for pos_tag in pos_seq]
	pos_seq = [l for tag in [pos_tag.split() for pos_tag in pos_seq] for l in tag]
	
	pos_seq = '[' + '] ['.join([pos_tag for pos_tag in pos_seq if pos_tag]) + ']'
	
	return pos_seq 

def get_german_example_metadata(
	source: Tree,
	pfx: str,
	target: Tree
) -> Dict:
	"""
	Gets metadata about the passed example, consisting of a seq2seq mapping with a source, prefix, and target.
	:param source: Tree: the source Tree
	:param pfx: str: the task prefix passed to the model
	:param target: the target Tree
	:returns metadata: a dictionary recording the following properties for the example:
					   - transitivity of the main verb (v_trans)
					   - definiteness of main clause subject/object (subj_def, obj_def)
					   - number of main clause subject/object (subj_num, obj_num)
					   - the identity of the main auxiliary (main_aux)
					   - whether the main clause is subject initial or not
					   - how many adverbial clauses before the main clause
					   - how many adverbial clauses after the main clause
					   - the number of adverbial clauses
					   - pos tags for source and target
	"""
	source = source.copy(deep=True)
	target = target.copy(deep=True)
	
	metadata = {}
	
	# transitivity of main clause verb
	main_clause 	= grep_next_subtree(source, 'S(Inv)?2')
	main_clause_mp 	= grep_next_subtree(main_clause, 'MP(Sg|Pl)(Inv)?')
	main_clause_vp 	= grep_next_subtree(main_clause_mp, 'VP')
	main_clause_vp 	= grep_next_subtree(main_clause_vp, 'VP') # do this twice because of how the german grammar is set up
	main_clause_vp 	= grep_next_subtree(main_clause_vp, '(IVP|TVP(Masc|Fem|Neut)?)')
	main_clause_v 	= grep_next_subtree(main_clause_vp, '(IV|TV)$')
	metadata.update({'v_trans': 'intransitive' if main_clause_v.label().symbol() == 'IV' else 'transitive'})
	
	# definiteness of main clause subject
	main_clause_subj = grep_next_subtree(main_clause, 'NP(Sg|Pl)Nom')
	main_clause_subj = grep_next_subtree(main_clause_subj, '(NP(Masc|Fem)(Sg|Pl)Nom|PN)')
	if main_clause_subj.label().symbol() == 'PN':
		metadata.update({'subj_def': 'definite'})
	else:
		main_clause_subj_det = grep_next_subtree(main_clause, 'Det')
		if main_clause_subj_det[0] in ['der', 'die', 'das']:
			metadata.update({'subj_def': 'definite'})
		else:
			metadata.update({'subj_def': 'indefinite'})
	
	# definiteness of main clause object
	if main_clause_v.label().symbol() == 'TV':
		main_clause_obj 	= grep_next_subtree(main_clause_vp, 'NP(Masc|Fem|Neut)(Sg|Pl)Acc')
		main_clause_obj_det = grep_next_subtree(main_clause_obj, 'Det')
		if main_clause_obj_det[0] in ['den', 'die', 'das']:
			metadata.update({'obj_def': 'definite'})
		else:
			metadata.update({'obj_def': 'indefinite'})
	else:
		metadata.update({'obj_def': None})
	
	# number of main clause subject
	if main_clause_subj.label().symbol() == 'PN':
		metadata.update({'subj_num': 'sg'})
	else:
		metadata.update({
			'subj_num': 'sg' 
				if any(
					l for l in get_pos_labels(main_clause_subj) 
					if re.match('N(Masc|Fem|Neut)SgNom', l)
				)
				else 'pl'
		})
	
	# number of main clause object
	if main_clause_v.label().symbol() == 'TV':
		if 'Pl' in main_clause_obj_det.label().symbol():
			metadata.update({'obj_num': 'pl'})
		else:
			metadata.update({'obj_num': 'sg'})
	else:
		metadata.update({'obj_num': None})
	
	# main auxiliary
	main_clause_m = grep_next_subtree(main_clause_mp, 'M(?!P)')
	metadata.update({'main_aux': GERMAN_MODAL_MAP.get(main_clause_m[0], main_clause_m[0])})
	
	# is the main clause subject initial?
	labels = get_labels(source)
	metadata.update({'main_clause_subj_initial': not 'SInv2' in labels})
	
	# number of AdvPs before and after main clause
	main_clause_start 	= labels.index('S2') if 'S2' in labels else labels.index('SInv2')
	pre_main_advps 		= len([l for i, l in enumerate(labels) if l == 'AdvP' and i < main_clause_start])
	post_main_advps 	= len([l for i, l in enumerate(labels) if l == 'AdvP' and i > main_clause_start])
	metadata.update({
		'pre_main_advps'	: pre_main_advps, 
		'post_main_advps'	: post_main_advps,
		'total_advps'		: pre_main_advps + post_main_advps
	})
	
	# get pos seq with details suppressed	
	source_pos_seq = get_german_pos_seq(source)
	metadata.update({'source_pos_seq': source_pos_seq})
	
	if pfx == 'pos':
		metadata.update({'target_pos_seq': source_pos_seq})
	else:
		tgt_main_clause 		= grep_next_subtree(target, '(S2|SInv2)')
		if not 'kein' in ' '.join(tgt_main_clause.leaves()):
			tgt_main_clause_mp 	= grep_next_subtree(tgt_main_clause, 'MP(Sg|Pl)(Inv)?')
			tgt_main_clause_vp 	= grep_next_subtree(tgt_main_clause_mp, 'VP')
			tgt_main_clause_vp 	= grep_next_subtree(tgt_main_clause_vp, 'VP') # do this twice because of how the german grammar is set up
			tgt_main_clause_vp 	= grep_next_subtree(tgt_main_clause_vp, '(IVP|TVP(Masc|Fem|Neut)?)')
			tgt_main_clause_v 	= grep_next_subtree(tgt_main_clause_vp, '(IV|TV)$')
			tgt_main_clause_v.set_label(f'Neg {tgt_main_clause_v.label().symbol()}')
		else:
			main_clause_indef_det = next(
				tgt_main_clause.subtrees(
					filter = lambda x: len(x.leaves()) == 1 and 'kein' in x.leaves()[0]
				)
			)
			main_clause_indef_det.set_label(f'Neg {main_clause_indef_det.label().symbol()}')
			
		tgt_pos_seq = get_german_pos_seq(target)
		metadata.update({'target_pos_seq': tgt_pos_seq})
	
	return metadata

def get_turkish_pos_seq(t: Tree) -> str:
	'''Remove unwanted info from Turkish pos tags for comparison purposes and return as a string.'''
	pos_seq = get_pos_labels(t)
	pos_seq = [l for tag in [pos_tag.split() for pos_tag in pos_seq] for l in tag]
	pos_seq = [pos_tag.split('_',1)[0] for pos_tag in pos_seq]
	pos_seq = [re.sub('P$', '', pos_tag) for pos_tag in pos_seq]
	pos_seq = [re.sub('(Tense|Person[1-3])', '', pos_tag) for pos_tag in pos_seq]
	pos_seq = [l for tag in [pos_tag.split() for pos_tag in pos_seq] for l in tag]
	
	pos_seq = '[' + '] ['.join([pos_tag for pos_tag in pos_seq if pos_tag]) + ']'
	
	return pos_seq

def get_turkish_example_metadata(
	source: Tree,
	pfx: str,
	target: Tree
) -> Dict:
	"""
	Gets metadata about the passed example, consisting of a seq2seq mapping with a source, prefix, and target.
	:param source: Tree: the source Tree
	:param pfx: str: the task prefix passed to the model
	:param target: the target Tree
	:returns metadata: a dictionary recording the following properties for the example:
					   - transitivity of the main verb (v_trans)
					   - definiteness of main clause subject/object (subj_def, obj_def)
					   - number of main clause subject/object (subj_num, obj_num)
					   - the identity of the main auxiliary (main_aux)
					   - how many adverbial clauses before the main clause
					   - how many adverbial clause after the main clause
					   - the number of adverbial clauses
					   - pos tags
	(not all of these are currently in the turkish grammar, so we'll use a default value)
	"""
	source = source.copy(deep=True)
	target = target.copy(deep=True)
	
	metadata = {}
	
	main_clause 	= grep_next_subtree(source, 'S')
	main_clause_vp 	= grep_next_subtree(main_clause, 'VP')
	main_clause_v 	= grep_next_subtree(main_clause_vp, 'V_(in)?trans')
	metadata.update({'v_trans': 'intransitive' if main_clause_v.label().symbol() == 'V_intrans' else 'transitive'})
	
	# placeholders
	metadata.update({'subj_def': None})
	metadata.update({'obj_def': None})
	metadata.update({'subj_num': 'sg'})
	metadata.update({'obj_num': 'sg'})
	
	# no aux in turkish grammar, so use the main clause verb stem instead for now
	metadata.update({'main_v': grep_next_subtree(main_clause_v, 'stem')[0].strip()})
	
	# currently no adverbial clauses in turkish
	pre_main_advps = 0
	post_main_advps = 0
	metadata.update({'pre_main_advps': pre_main_advps})
	metadata.update({'post_main_advps': post_main_advps})
	metadata.update({'total_advps': pre_main_advps + post_main_advps})
	
	source_pos_seq = get_turkish_pos_seq(source)
	metadata.update({'source_pos_seq': source_pos_seq})
	
	if pfx == 'pos':
		metadata.update({'target_pos_seq': source_pos_seq})
	else:
		# this may need to be changed later based on what they said about turkish negation
		# if those other strategies are added, more will be needed here
		tgt_main_clause 	= grep_next_subtree(target, 'S')
		tgt_main_clause_vp 	= grep_next_subtree(tgt_main_clause, 'VP')
		tgt_main_clause_v 	= grep_next_subtree(tgt_main_clause_vp, 'stem')
		tgt_main_clause_v.set_label(f'{tgt_main_clause_v.label().symbol()} Neg')
		tgt_pos_seq 		= get_turkish_pos_seq(target)
		metadata.update({'target_pos_seq': tgt_pos_seq})
	
	return metadata

def get_example_metadata(
	grammar: PCFG,
	*args, **kwargs,
) -> Dict:
	"""
	Gets metadata about the passed example, consisting of a seq2seq mapping with a source, prefix, and target.
	:param grammar: the grammar that generated the example
	:param args: passed to get_lang_example_metadata()
	:param kwargs: passed to get_lang_example_metadata()
	:returns metadata: a dictionary recording language-specific properties for the example
	"""
	function_map = {
		'en': get_english_example_metadata,
		'de': get_german_example_metadata,
		'tu': get_turkish_example_metadata
	}
	
	try:
		metadata = function_map[grammar.lang](*args, **kwargs)
	except KeyError:
		metadata = {}
	
	return metadata	

def create_dataset_json(
	grammar: PCFG, 
	ex_generator: Callable, 
	file_prefix: str = '',
	overwrite: bool = False,
	**splits: Dict[str,int]
) -> None:
	"""
	Create a dataset json file that can be read using the datasets module's dataset loader.
	Also outputs a companion json that records various linguistic properties of each sentence.
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
		metadata = []
		if not os.path.exists(os.path.join('data', file_prefix + name + '.json.gz')) or overwrite:
			prefixes = {}
			l = []
			print(f'Generating {name} examples')
			for n in tqdm(range(n_examples)):
				source, pfx, target = ex_generator(grammar)
				metadata.append(get_example_metadata(grammar, source, pfx, target))
				prefixes[pfx] = 1 if not pfx in prefixes else prefixes[pfx] + 1
				l += [{
					'translation': {
						'src'	: format_tree_string(source, grammar, pfx), 
						'prefix': pfx, 
						'tgt'	: format_tree_string(target, grammar, pfx)
					}
				}]
			
			for pfx in prefixes:
				print(f'{name} prop {pfx} examples: {prefixes[pfx]/n_examples}')
			
			if l:
				print('Saving examples to data/' + file_prefix + name + '.json.gz')
				with gzip.open(os.path.join('data', file_prefix + name + '.json.gz'), 'wt', encoding='utf-8') as f:
					for ex in tqdm(l):
						json.dump(ex, f, ensure_ascii=False)
						f.write('\n')
				
				print('Saving metadata to data/' + file_prefix + name + '_metadata.json.gz')
				with gzip.open(os.path.join('data', file_prefix + name + '_metadata.json.gz'), 'wt', encoding='utf-8') as f:
					for ex in tqdm(metadata):
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
	:param configs: (List[Dict]): This should be in the following format:
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
	configs = load_configs(configs) if configs is None or isinstance(configs,str) else configs
	
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
	langs = list(load_config(langs).keys()) if langs is None or isinstance(langs,str) else langs
	
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
	configs = load_config(configs) if configs is None or isinstance(configs,str) else configs
	
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
		'#SBATCH --time=10:00:00',
		'#SBATCH --gpus=v100:1',
		'#SBATCH --partition=gpu',
		'#SBATCH --mail-type=END,FAIL,INVALID_DEPEND',
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
	eval_script = eval_script.replace('--do_train \\', '--do_eval \\\n	--do_learning_curve \\')
	eval_script = eval_script.replace('[DEV_LANG]', '[TEST_LANG]')
	eval_script = re.sub(r'_dev(\.|_)', '_test\\1', eval_script)
	eval_script = eval_script.replace('--per_device_train_batch_size=4', '--per_device_train_batch_size=8')
	eval_script = eval_script.replace('	--gradient_accumulation_steps=32 \\\n', '')
	eval_script = eval_script.replace(
		'	--predict_with_generate \\\n	--num_train_epochs 10.0', 
		'	--predict_with_generate \\'
	)
	
	langs 		= list(load_config(langs).keys()) if langs is None or isinstance(langs,str) else langs
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

if __name__ == '__main__':
	
	create_and_combine_negation_datasets()