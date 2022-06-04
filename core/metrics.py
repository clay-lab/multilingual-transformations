import os
import re
import sys
import gzip
import json
import nltk

from tqdm import tqdm
from typing import *
from inspect import signature, getmembers
from grammars import german_grammar, turkish_grammar, english_grammar
from grammars.generator import get_english_pos_seq, get_german_pos_seq, get_turkish_pos_seq
from itertools import cycle
from statistics import mean
from collections import defaultdict

# language-specific regexes to match neg word(s)
NEG_WORD_REGEXES = {
	'en': re.compile('not'),
	'de': re.compile('(nicht|k(?=ein))'),
	'tu': re.compile(r'(m(i|ı|u|ü)y|m(adı|edi)|m(aya|eye))\w+'),
}

NEG_STR_REGEXES = {
	'en': re.compile(r'\[M\] \[Neg\]'),
	'de': re.compile(r'(\[Neg\] \[V\]|\[Neg\] \[Det\])'),
	'tu': re.compile(r'\[V\] \[Neg\]'),
}

GRAMMARS = {
	'en': english_grammar.not_grammar,
	'de': german_grammar.nicht_grammar,
	'tu': turkish_grammar.turkish_grammar,
}

POS_SEQ_FUNCS = {
	'en': get_english_pos_seq,
	'de': get_german_pos_seq,
	'tu': get_turkish_pos_seq,
}

# language-specific lowercase function to deal with Turkish ı/i's
LOWERCASE = defaultdict(lambda: lambda s: s.lower())
LOWERCASE.update({
	'tu': lambda s: s.replace('İ', 'i').replace('I', 'ı').lower(),
})

def parse_to_pos(
	sentence: str,
	trn_lang: str,
	tgt_lang: str,
	comparison_pos_seq: str = None,
) -> str:
	'''
	Parse a predicted sentence to a sequence of part of speech tags.
	First, parse according to the target language, and then see if there are any
	leftovers from the training language. If there are words that are PoS ambiguous,
	parse them in the way that best matches the comparison_pos_sequence.
	
		params:
			sentence (str): the sentence to parse to a PoS sequence
			trn_lang (str): the language of the training data
			tgt_lang (str): the language of the target data
			comparison_pos_seq (str): a pos sequence to compare the sentence to.
									  this is used to deal with ambiguious words.
									  if a word matches multiple categories, we choose
									  the one that is most favorable to matching it to the comparison sequence
		
		returns:
			str: the PoS sequence of the sentence as a string
	'''
	if tgt_lang == 'tu' or trn_lang == 'tu':
		return '' # not currently implemented for Turkish
	
	tgt_grammar = GRAMMARS[tgt_lang]
	trn_grammar = GRAMMARS[trn_lang]
	
	tgt_word_to_pos = [{LOWERCASE[tgt_lang](w).strip(): POS_SEQ_FUNCS[tgt_lang]([str(p.lhs())])} for p in tgt_grammar.productions() for w in p.rhs() if not isinstance(w,nltk.grammar.Nonterminal)]
	trn_word_to_pos = [{LOWERCASE[trn_lang](w).strip(): POS_SEQ_FUNCS[trn_lang]([str(p.lhs())])} for p in trn_grammar.productions() for w in p.rhs() if not isinstance(w,nltk.grammar.Nonterminal)]
	
	sentence = re.sub(r'[\?\.,]', '', sentence)
	sentence = LOWERCASE[tgt_lang](format_sentences([sentence])[0])
	sentence = LOWERCASE[trn_lang](format_sentences([sentence])[0])
	
	# negation is not in the grammar, but is done by editing strings.
	# so we add in a special PoS for it manually here
	sentence = NEG_WORD_REGEXES[tgt_lang].sub(' [Neg] ', sentence)
	sentence = NEG_WORD_REGEXES[trn_lang].sub(' [Neg] ', sentence)
	
	words = sentence.split()
	pos_seq = words.copy()
	
	for i, word in enumerate(pos_seq):
		for d in tgt_word_to_pos + trn_word_to_pos:
			if word in d.keys():
				pos_seq[i] = d[word]
				break
			elif word == '[Neg]':
				break
			# elif tgt_lang == 'tu' or trn_lang == 'tu':
			# 	if re.sub('y?i$', '', word) in d.keys():
			# 		pos_seq[i] = d[re.sub('y?i$', '', word)]
			# 		break
			# 	elif len(pos_seq) >= i+2 and f'{word} {pos_seq[i+1]}' in d.keys():
			# 		pos_seq[i] = d[f'{word} {pos_seq[i+1]}']
			# 		break
			# 	elif i != 0 and f'{pos_seq[i-1]} {word}' in d.keys():
			# 		pos_seq[i] = ''
			# 		break
		else:
			pos_seq[i] = '[Unk]'
	
	# # this runs if there are any words that haven't been parsed to a POS by the target language (i.e.,
	# # if the training language is mixed in, as is common)
	# for i in [i for i, pos in enumerate(pos_seq) if not pos.startswith('[') and not pos.endswith(']')]:
	# 	pos_seq[i] = trn_word_to_pos.get(pos_seq[i], '[Unk]')
	
	# deal with ambiguous pos terms if any don't match
	comparison_pos_seq = comparison_pos_seq.split() if comparison_pos_seq is not None else comparison_pos_seq
	
	if comparison_pos_seq is not None and  any(pred != comp for pred, comp in zip(pos_seq, comparison_pos_seq)) and len(pos_seq) == len(comparison_pos_seq):
		# get a dict of lists mapping each ambiguious word in the target and training languages to a list of the possible other tags they could have
		tgt_ambiguities = {w: list(set([d[w] for d in tgt_word_to_pos if w in d])) for d in tgt_word_to_pos for w in d.keys()}
		tgt_ambiguities = {w: l for w, l in tgt_ambiguities.items() if len(l) > 1}
		trn_ambiguities = {w: list(set([d[w] for d in trn_word_to_pos if w in d])) for d in trn_word_to_pos for w in d.keys()}
		trn_ambiguities = {w: l for w, l in trn_ambiguities.items() if len(l) > 1}
		
		# if a word is ambiguous, we want to give it the best possible change of matching.
		# to do this, we check if the comparison pos tag for that position is one of the ones that matches that ambiguous word
		# if so, replace it with that pos tag in the predicted pos tag
		for i, word in enumerate(words):
			if not pos_seq[i] == comparison_pos_seq[i] and (word in trn_ambiguities or word in tgt_ambiguities):
				if comparison_pos_seq[i] in tgt_ambiguities[word]:
					pos_seq[i] = comparison_pos_seq[i]
				elif comparison_pos_seq[i] in trn_ambiguities[word]:
					pos_seq[i] = comparison_pos_seq[i]
	
	return ' '.join(pos_seq)

class metric():
	'''
	A class to simplify the construction of useful metrics functions. Can be used as a function decorator.
	When called, the result returned is the proportion of true vs. false results
	from the function to each row of passed arguments (with Nones excluded).
	It also stores the individual results with the passed arguments in metric.results,
	as well as the total rows passed, the number of rows included in the results (i.e., excluding Nones),
	the number of true points, the number of false points, the number of omitted (i.e., None) points,
	the actual arguments passed, and the mean (which is the same as the proportiion returned).
	
	Use as follows:
		
		@metric
		def m(x, y):
			return x == y
		
		Now, you can call with 
			m([1, 2, ...], [1, 1, ...])
		to get the proportion of equal values at identical indices in each list.
	'''
	def __init__(self, fun: Callable) -> 'metric':
		'''
		Constructor to simplify the definition of vectorized metric 
		functions that report mean accuracy on some measure.
		
			params:
				fun (Callable)			: a function that returns a value to be interpreted as a boolean
								
			returns:
				metric_fun (Callable)	: a function that returns the mean of applying the original fun to each tuple
										  of zipped arguments passed to it, with length 1 arguments repeated for each call.
										  note that arguments unused by the function will be ignored to facilitate the construction
										  of identical calls.
		'''
		def wrapper(self, *args: Tuple, **kwargs: Dict) -> float:
			'''
			Return the proportion of truthy responses from passing each tuple of zipped (kw)args to fun.
			Non-list/tuple arguments are put in a list to facilitate this.
			If an argument is of length 1, it is repeated out to the maximum length.
			All arguments not of length 1 must have the same number of elements.
					
				params:
					*args (tuple)	: passed to fun
					**kwargs (dict) : passed to fun
				
				returns:
					prop (float)	: the mean of the result of applying fun to each tuple of zipped args and kwargs,
									  with None omitted. If every value is None, returns None
			'''
			# in our implementation, we want to be able to pass the 
			# same arguments in the same order to each metric for ease of use.
			# but not every metric will have every argument defined for it. 
			# this filters out arguments that are not used by the function,
			# so they don't get passed to it.
			sig 	= signature(fun)
			names 	= [p.name for p in sig.parameters.values()]
			kwnames = [name for name in names if name in kwargs.keys()]
			args 	= args[:min(len(names),len(args))]
			kwargs 	= {k: v for k, v in kwargs.items() if k in kwnames}	
			
			# convert single elements to lists so we can iterate
			args 	= [[arg] if not isinstance(arg,(list,tuple)) else arg for arg in args]
			kwargs 	= {k : [v] if not isinstance(v,(list,tuple)) else v for k, v in kwargs.items()}
			
			# check lengths to make sure we can pad if needed
			args_lens 	= [len(arg) for arg in args]
			kwargs_lens = [len(v) for v in kwargs.values()]
			assert len(set(l for l in args_lens + kwargs_lens if not l == 1)) <= 1, 'All arguments must be a single value or have the same length!'
			
			# pad len 1 arguments to support vectorization
			if max([*args_lens, *kwargs_lens]) > 1:
				args 	= [cycle(arg) if len(arg) == 1 else arg for arg in args]
				kwargs 	= {k: cycle(v) if len(v) == 1 else v for k, v in kwargs.items()}
				
			# this zips over the args and kwargs
			# by zipping over the args and the kwarg values
			# and then repacking the kwargs values into a dictionary
			# that gets unpacked and passed to the function
			# it allows us to define metrics very flexibly, 
			# since all we need to do is make sure that
			# they return something that can be cast to boolean
			self.results = [
				(
					(
						tuple(each_step_args[:len(args)]),
						dict(zip(
							kwargs.keys(),
							each_step_args[len(args):]
						))
					),
					fun(
						*each_step_args[:len(args)], 
						**dict(zip(
							kwargs.keys(), 
							each_step_args[len(args):]
						))
					)
				)
			 	for each_step_args in zip(*args, *kwargs.values())
			]
			
			self.total_points 		= len(self.results)
			
			# we omit Nones from the mean
			filtered_results 		= [bool(res[-1]) for res in self.results if res is not None]
			
			self.included_points 	= len(filtered_results)
			self.omitted_points 	= self.total_points - self.included_points
			self.true_points 		= len([res for res in filtered_results if res])
			self.false_points 		= len([res for res in filtered_results if not res])
			
			# if everything is none, return none; otherwise we can get the mean
			self.mean 				= mean(filtered_results) if filtered_results else None
			
			return self.mean
		
		return_fun 					= wrapper
		
		# make some attributes of the returned function reflect its original definition for clarity, ...
		return_fun.__name__ 		= fun.__name__
		
		self.name 					= return_fun.__name__
		
		# , ... except for the return type, which should match the new type 
		# (the original type would be too misleading)
		sig 						= signature(fun)
		return_type 				= signature(wrapper).return_annotation
		sig 						= sig.replace(return_annotation=return_type)
		
		return_fun.__signature__ 	= sig
		self.signature 				= sig
		self._original_fun 			= fun
		self.fun 					= return_fun
		self.total_points			= 0
		self.included_points 		= 0
		self.true_points 			= 0
		self.false_points 			= 0
		self.omitted_points 		= 0
		self.arguments 				= []
		self.mean 					= None
	
	def __call__(self, *args, **kwargs) -> float:
		'''Calls the metric's function with the passed arguments.'''
		self.arguments = [*args, kwargs]
		return self.fun(self, *args, **kwargs)
	
	def __repr__(self) -> str:
		'''Get a string formatted for printing.'''
		return str(self)
	
	def __str__(self) -> str:
		'''Get a string formatted for printing.'''
		return f'metric(\n\t' + \
			f'name={self.name},\n\t' + \
			(f'mean={self.mean:.2f},\n\t' if self.mean else 'mean=None,\n\t') + \
			f'total_points={self.total_points},\n\t' + \
			f'included_points={self.included_points}\n\t' + \
			f'true_points={self.true_points}\n\t' + \
			f'false_points={self.false_points}\n\t' + \
			f'omitted_points={self.omitted_points}\n' + \
		')'
	
	def to_list(self) -> List:
		'''Returns the current results as a formatted list of dicts.'''
		if not hasattr(self, 'results'):
			return []
		
		# the first part names the args passed without using keywords
		return [{
				**{
					k: res[0][0][i] 
					for k, i in zip(
						list(self.signature.parameters.keys()), 
						range(len(res[0][0]))
					)
				},
				**res[0][1],
				self.name: res[1]
			} for res in self.results
		]
	
	def to_dict(self) -> Dict:
		'''Returns the current results as a formatted dict of lists.'''
		l = self.to_list()
		
		return {k: [d[k] for d in l] for k in l[0]} if l else {}
	
	def to_dataframe(self) -> 'pd.DataFrame':
		'''Returns the current results as a pandas data frame.'''
		import pandas as pd
		
		return pd.DataFrame(self.to_list())

@metric
def exact_match(
	pred_sentence: str, 
	gold_sentence: str
) -> bool:
	'''Do the passed sentences match exactly?'''
	return pred_sentence == gold_sentence

@metric
def ignorecase_exact_match(
	pred_sentence: str, 
	gold_sentence: str, 
	tgt_lang: str
) -> bool:
	'''Do the sentences match when converted to lowercase?'''
	return LOWERCASE[tgt_lang](pred_sentence) == LOWERCASE[tgt_lang](gold_sentence)

@metric
def replace_negation_ignorecase_exact_match(
	pred_sentence: str, 
	gold_sentence: str, 
	trn_lang: str, 
	tgt_lang: str
) -> bool:
	'''Do the sentences match when the neg words are replaced with identical values?'''
	for lang in [trn_lang, tgt_lang]:
		pred_sentence 	= NEG_WORD_REGEXES[lang].sub('[Neg]', pred_sentence)
		gold_sentence 	= NEG_WORD_REGEXES[lang].sub('[Neg]', gold_sentence)
	
	# ensure spaces around neg to account for language-specific behavior
	# e.g., English "cannot" instead of "can not", Turkish negation is inside the word
	pred_sentence 	= re.sub(r'(?<!\s)(\[Neg\])', ' \\1', pred_sentence)
	pred_sentence 	= re.sub(r'(\[Neg\])(?!\s)', '\\1 ', pred_sentence)
	
	gold_sentence 	= re.sub(r'(?<!\s)(\[Neg\])', ' \\1', gold_sentence)
	gold_sentence 	= re.sub(r'(\[Neg\])(?!\s)', '\\1 ', gold_sentence)
	
	pred_sentence 	= LOWERCASE[tgt_lang](pred_sentence)
	gold_sentence 	= LOWERCASE[tgt_lang](gold_sentence)
	
	return pred_sentence == gold_sentence

"""
@metric
def trn_lang_negation_in_prediction(
	pred_sentence: str, 
	trn_lang: str
) -> re.Match:
	'''Is the neg word from the training language in the sentence?'''
	return bool(NEG_WORD_REGEXES[trn_lang].search(LOWERCASE[trn_lang](pred_sentence)))

@metric
def tgt_lang_negation_in_prediction(
	pred_sentence: str,
	tgt_lang: str
) -> re.Match:
	'''Is the neg word from the target language in the sentence?'''
	return bool(NEG_WORD_REGEXES[tgt_lang].search(LOWERCASE[tgt_lang](pred_sentence)))

@metric
def first_word_match(
	pred_sentence: str, 
	gold_sentence: str
) -> Union[bool,'NoneType']:
	'''Does the first word of each sentence match? If either sentence is empty, returns None.'''
	pred_words = pred_sentence.split()
	gold_words = gold_sentence.split()
	
	# this accounts for an instance when the model has predicted no text
	if pred_words and gold_words:
		return pred_words[0] == gold_words[0]

@metric
def second_word_match(
	pred_sentence: str, 
	gold_sentence: str
) -> Union[bool,'NoneType']:
	'''
	Do the second words of each sentence match? 
	If either sentence has only one word, returns None.
	'''
	pred_words = pred_sentence.split()
	gold_words = gold_sentence.split()
	
	if len(pred_words) > 1 and len(gold_words) > 1:
		return pred_words[1] == gold_words[1]
"""

@metric
def one_trn_lang_negation(
	pred_sentence: str,
	trn_lang: str
) -> bool:
	'''Is there exactly 1 training language negation in the sentence?'''
	return len(NEG_WORD_REGEXES[trn_lang].findall(LOWERCASE[trn_lang](pred_sentence))) == 1

@metric
def one_tgt_lang_negation(
	pred_sentence: str,
	tgt_lang: str
) -> bool:
	'''Is there exactly 1 target language negation in the sentence?'''
	return len(NEG_WORD_REGEXES[tgt_lang].findall(LOWERCASE[tgt_lang](pred_sentence))) == 1

@metric
def one_negation(
	pred_sentence: str,
	trn_lang: str,
	tgt_lang: str
) -> bool:
	'''Is there exactly one negation in the sentence?'''
	for lang in [trn_lang, tgt_lang]:
		pred_sentence 	= NEG_WORD_REGEXES[lang].sub('[Neg]', pred_sentence)
	
	return len(re.findall(r'\[NEG\]', pred_sentence)) == 1

@metric
def pos_exact_match(
	pred_sentence: str,
	tgt_pos_seq: str,
	trn_lang: str,
	tgt_lang: str,
) -> bool:
	'''Does the part of speech sequence of the prediction match that of the target?'''
	pred_pos_seq = parse_to_pos(pred_sentence, trn_lang, tgt_lang, tgt_pos_seq)
	
	return pred_pos_seq == tgt_pos_seq if not '[Unk]' in pred_pos_seq else None

@metric
def trn_neg_structure(
	pred_sentence: str,
	tgt_pos_seq: str,
	trn_lang: str,
	tgt_lang: str
) -> bool:
	'''Is the training language negation structure in the prediction? Returns None if the negation structure matches both languages.'''
	pred_pos_seq = parse_to_pos(pred_sentence, trn_lang, tgt_lang, tgt_pos_seq)
	
	if not NEG_STR_REGEXES[tgt_lang].findall(pred_pos_seq):
		return bool(NEG_STR_REGEXES[trn_lang].findall(pred_pos_seq))

@metric
def tgt_neg_structure(
	pred_sentence: str,
	tgt_pos_seq: str,
	trn_lang: str,
	tgt_lang: str
) -> bool:
	'''Is the target language negation structure in the prediction? Returns None if the negation structure matches both languages.'''
	pred_pos_seq = parse_to_pos(pred_sentence, trn_lang, tgt_lang, tgt_pos_seq)
	
	if not NEG_STR_REGEXES[trn_lang].findall(pred_pos_seq):
		return bool(NEG_STR_REGEXES[tgt_lang].findall(pred_pos_seq))

"""
@metric
def zero_negation(
	pred_sentence: str,
	trn_lang: str,
	tgt_lang: str
) -> bool:
	'''Is there any negation in the sentence?'''
	for lang in [trn_lang, tgt_lang]:
		pred_sentence = NEG_WORD_REGEXES[lang].sub('[Neg]', pred_sentence)
	
	return len(re.findall(r'\[NEG\]', pred_sentence)) == 0
"""

def format_sentences(sentences: List[str]) -> List[str]:
	'''
	Format sentences for comparison purposes.
	Remove extra whitespace and add a space before punctuation to facilitate word-level comparisons.
	
		params:
			sentences (list[str]): a list of strings to format
	'''
	sentences = [sentence.strip() for sentence in sentences]
	sentences = [re.sub(r'(?<!\s)([\?\.,])', ' \\1', sentence) for sentence in sentences]
	sentences = [re.sub(r'\s+', ' ', sentence) for sentence in sentences]
	
	return sentences

# this gets a list of all the metrics functions defined 
# in this file so we can use it as a default argument
# for compute_metrics below
all_metrics = [
	eval(name) 
	for name, obj in getmembers(sys.modules[__name__]) 
		if isinstance(obj, metric)
]

def compute_metrics(
	pred_file: str, 
	gold_file: str,
	metrics: List[metric] = all_metrics, 
	return_results: str = None,
) -> Dict:
	'''
	Computes metrics on a prediction file and a gold file.
	
		params:
			pred_file (str)			: a file containing sentences predicted by the model.
			gold_file (str)			: a file containing the target sentences.
									  the pred_file and gold_file should have corresponding 
									  sentences in the same order.
			metrics (List[metric])	: a list of metrics to run on the passed files.
									  (these are defined above in this file).
									  Default runs all metrics defined in this file.
			return_results (bool)	: whether and in what format to return the individual results.
									  default returns only the mean accuracy.
									  pass 'list', 'dict', or 'df'/'dataframe' to get the individual results
									  in that format.
		
		returns:
			props (Dict[str,float])	: a dictionary mapping the name of each metric to the
									  proportion of sentences that pass that metric.
	'''
	RETURN_RESULTS_MAP = {
		'list': lambda x: x.to_list(),
		'dict': lambda x: x.to_dict(),
		'df': lambda x: x.to_dataframe(),
		'dataframe': lambda x: x.to_dataframe(),
	}
	
	with open(pred_file, 'r', encoding='utf-8') as pred_f:
		pred_lines	= pred_f.readlines()
	
	open_fn 		= gzip.open if gold_file.endswith('.gz') else open
	
	with open_fn(gold_file, 'rt', encoding='utf-8') as gold_f:
		gold_lines 	= gold_f.readlines()
	
	metadata_file = gold_file.replace('.json', '_metadata.json')
	if os.path.isfile(metadata_file):
		open_fn 	= gzip.open if metadata_file.endswith('.gz') else open
		with open_fn(metadata_file, 'rt', encoding='utf-8') as metadata_f:
			metadata_lines = metadata_f.readlines()
		
		metadata_jsons 	= [json.loads(metadata_line) for metadata_line in metadata_lines]
		if all('source_pos_seq' in metadata_json.keys() for metadata_json in metadata_jsons):
		 	source_pos_seq = [metadata_json['source_pos_seq'] for metadata_json in metadata_jsons]
		else:
			source_pos_seq = None
		
		if all('target_pos_seq' in metadata_json.keys() for metadata_json in metadata_jsons):
			target_pos_seq = [metadata_json['target_pos_seq'] for metadata_json in metadata_jsons]
		else:
			target_pos_seq = None
	else:
		source_pos_seq = None
		target_pos_seq = None
	
	gold_file 		= re.sub(r'\.gz$', '', gold_file)
	
	pred_lines 		= format_sentences(pred_lines)
	
	if gold_file.endswith('.json'):
		gold_jsons 	= [json.loads(gold_line) for gold_line in gold_lines]
		gold_lines 	= [gold_json['translation']['tgt'] for gold_json in gold_jsons]
		src_lines 	= [gold_json['translation']['src'] for gold_json in gold_jsons]
		src_lines 	= format_sentences(src_lines)
	else:
		gold_lines 	= [gold_line.strip().split('\t')[1] for gold_line in gold_lines]
		src_lines 	= None
	
	gold_lines		= format_sentences(gold_lines)
	
	# if neg_only and gold_file.endswith('.json'):
	# 	gold_line_indices = [i for i, line in enumerate(gold_jsons) if line['translation']['prefix'] == 'neg']
	# 	gold_lines = [line for i, line in enumerate(gold_lines) if i in gold_line_indices]
	# 	pred_lines = [line for i, line in enumerate(pred_lines) if i in gold_line_indices]
	
	trn_lang 		= re.findall(r'outputs[/\\](.*?)[/\\$]', pred_file)[0]
	trn_lang 		= re.findall(r'neg-(.*?)-', trn_lang)[0]
	tgt_lang 		= re.findall(r'neg_(.*?)_', os.path.split(pred_file)[-1])[0]
	
	props = {}
	for m in tqdm(metrics):
		m(
			pred_sentence=pred_lines,
			gold_sentence=gold_lines,
			src_sentence=src_lines,
			src_pos_seq=source_pos_seq,
			tgt_pos_seq=target_pos_seq,
			trn_lang=trn_lang,
			tgt_lang=tgt_lang
		)
		
		props[m.name] = RETURN_RESULTS_MAP.get(return_results, lambda x: x.mean)(m)
	
	return props
