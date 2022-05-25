import os
import re
import sys
import gzip
import json

from tqdm import tqdm
from typing import *
from inspect import signature, getmembers
from itertools import cycle
from statistics import mean
from collections import defaultdict

# language-specific regexes to match neg word(s)
NEG_REGEXES = {
	'en': re.compile('not'),
	'de': re.compile('(nicht|kein)'),
	'tu': re.compile('(m(i|ı|u|ü)y|m(adı|edi)|m(aya|eye))'),
}

# language-specific lowercase function to deal with Turkish i's
LOWERCASE = defaultdict(lambda: lambda s: s.lower())
LOWERCASE.update({
	'tu': lambda s: s.replace('İ', 'i').replace('I', 'ı').lower(),
})

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
def replace_negation_exact_match(
	pred_sentence: str, 
	gold_sentence: str, 
	trn_lang: str, 
	tgt_lang: str
) -> bool:
	'''Do the sentences match when the neg words are replaced with identical values?'''
	for lang in [trn_lang, tgt_lang]:
		pred_sentence 	= NEG_REGEXES[lang].sub('[NEG]', pred_sentence)
		gold_sentence 	= NEG_REGEXES[lang].sub('[NEG]', gold_sentence)
	
	# ensure spaces around neg to account for language-specific behavior
	# e.g., English "cannot" instead of "can not", Turkish negation is inside the word
	pred_sentence 	= re.sub(r'(?<!\s)(\[NEG\])', ' \\1', pred_sentence)
	pred_sentence 	= re.sub(r'(\[NEG\])(?!\s)', '\\1 ', pred_sentence)
	
	gold_sentence 	= re.sub(r'(?<!\s)(\[NEG\])', ' \\1', gold_sentence)
	gold_sentence 	= re.sub(r'(\[NEG\])(?!\s)', '\\1 ', gold_sentence)
	
	return pred_sentence == gold_sentence

@metric
def trn_lang_negation_in_prediction(
	pred_sentence: str, 
	trn_lang: str
) -> re.Match:
	'''Is the neg word from the training language in the sentence?'''
	return bool(NEG_REGEXES[trn_lang].search(LOWERCASE[trn_lang](pred_sentence)))

@metric
def tgt_lang_negation_in_prediction(
	pred_sentence: str,
	tgt_lang: str
) -> re.Match:
	'''Is the neg word from the target language in the sentence?'''
	return bool(NEG_REGEXES[tgt_lang].search(LOWERCASE[tgt_lang](pred_sentence)))

"""
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
	return len(NEG_REGEXES[trn_lang].findall(LOWERCASE[trn_lang](pred_sentence))) == 1

@metric
def one_tgt_lang_negation(
	pred_sentence: str,
	tgt_lang: str
) -> bool:
	'''Is there exactly 1 target language negation in the sentence?'''
	return len(NEG_REGEXES[tgt_lang].findall(LOWERCASE[tgt_lang](pred_sentence))) == 1

@metric
def one_negation(
	pred_sentence: str,
	trn_lang: str,
	tgt_lang: str
) -> bool:
	'''Is there exactly one negation in the sentence?'''
	for lang in [trn_lang, tgt_lang]:
		pred_sentence 	= NEG_REGEXES[lang].sub('[NEG]', pred_sentence)
	
	return len(re.findall(r'\[NEG\]', pred_sentence)) == 1

@metric
def zero_negation(
	pred_sentence: str,
	trn_lang: str,
	tgt_lang: str
) -> bool:
	'''Is there any negation in the sentence?'''
	for lang in [trn_lang, tgt_lang]:
		pred_sentence = NEG_REGEXES[lang].sub('[NEG]', pred_sentence)
	
	return len(re.findall(r'\[NEG\]', pred_sentence)) == 0

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
	
	def format_lines(lines: List[str]) -> List[str]:
		'''
		Format lines for comparison purposes.
		Remove extra whitespace and add a space before punctuation to facilitate word-level comparisons.
		
			params:
				lines (list[str]): a list of strings to format
		'''
		lines = [line.strip() for line in lines]
		lines = [re.sub(r'(?<!\s)([\?\.,])', ' \\1', line) for line in lines]
		lines = [re.sub(r'\s+', ' ', line) for line in lines]
		
		return lines
	
	with open(pred_file, 'r', encoding='utf-8') as pred_f:
		pred_lines	= pred_f.readlines()
	
	open_fn 		= gzip.open if gold_file.endswith('.gz') else open
	
	with open_fn(gold_file, 'rt', encoding='utf-8') as gold_f:
		gold_lines 	= gold_f.readlines()
	
	gold_file 		= re.sub(r'\.gz$', '', gold_file)
	
	pred_lines 		= format_lines(pred_lines)
	
	if gold_file.endswith('.json'):
		gold_jsons 	= [json.loads(gold_line) for gold_line in gold_lines]
		gold_lines 	= [gold_json['translation']['tgt'] for gold_json in gold_jsons]
		src_lines 	= [gold_json['translation']['src'] for gold_json in gold_jsons]
		src_lines 	= format_lines(src_lines)
	else:
		gold_lines 	= [gold_line.strip().split('\t')[1] for gold_line in gold_lines]
		src_lines 	= None
	
	gold_lines		= format_lines(gold_lines)
	
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
			trn_lang=trn_lang,
			tgt_lang=tgt_lang
		)
		
		props[m.name] = RETURN_RESULTS_MAP.get(return_results, lambda x: x.mean)(m)
	
	return props
