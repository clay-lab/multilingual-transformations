import re
import json

from typing import *
from itertools import cycle
from inspect import signature
from statistics import mean

# language-specific regexes to match neg word(s)
NEG_REGEXES = {
	'en': re.compile('not'),
	'de': re.compile('nicht'),
	'tu': re.compile('(m(i|ı|u|ü)y)|(m(adı|edi))|(m(aya|eye))'),
}

# language-specific lowercase functions
# needed to deal with Turkish i's
LOWERCASE = {
	'en': lambda s: s.lower(),
	'de': lambda s: s.lower(),
	'tu': lambda s: s.replace('İ', 'i').replace('I', 'ı').lower(),
}

def metric(fun: Callable) -> Callable:
	'''
	Decorator to simplify the definition of vectorized metric 
	functions that report mean accuracy on some measure.
	
		params:
			fun (Callable)			: a function that returns a value to be interpreted as a boolean
							
		returns:
			metric_fun (Callable)	: a function that returns the mean of applying the original fun to each tuple
									  of zipped arguments passed to it, with length 1 arguments repeated for each call.
									  note that arguments unused by the function will be ignored to facilitate the construction
									  of identical calls.
	'''
	def wrapper(*args: Tuple, **kwargs: Dict) -> float:
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
		# we also omit Nones from the mean
		trues = [
			bool(res) for res in 
				[
					fun(
						*each_step_args[:len(args)], 
						**dict(zip(
							kwargs.keys(), 
							each_step_args[len(args):]
						))
					)
				 	for each_step_args in zip(*args, *kwargs.values())
				]
			if res is not None
		]
		
		# if everything is none, return none; otherwise we can get the mean
		prop_true = mean(trues) if trues else None
		
		return prop_true
	
	return_fun = wrapper
	
	# make some attributes of the returned function reflect its original definition for clarity, ...
	return_fun.__name__ = fun.__name__
	
	# , ... except for the return type, which should match the new type 
	# (the original type would be too misleading)
	sig 		= signature(fun)
	return_type = signature(wrapper).return_annotation
	sig 		= sig.replace(return_annotation=return_type)
	
	return_fun.__signature__ = sig
	
	return return_fun

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
	'''Do the passed sentences match when converted to lowercase?'''
	return LOWERCASE[tgt_lang](pred_sentence) == LOWERCASE[tgt_lang](gold_sentence)

@metric
def replace_negation_exact_match(
	pred_sentence: str, 
	gold_sentence: str, 
	src_lang: str, 
	tgt_lang: str
) -> bool:
	'''Do the sentences match exactly when the neg words are replaced with identical values?'''
	for lang in [src_lang, tgt_lang]:
		pred_sentence 	= NEG_REGEXES[lang].sub('[NEG]', pred_sentence)
		gold_sentence 	= NEG_REGEXES[lang].sub('[NEG]', gold_sentence)
	
	# ensure spaces around neg to account for language-specific behavior
	# e.g., English "cannot" instead of "can not", Turkish negation is inside the word
	pred_sentence 	= re.sub(r'(?<!\s)(\[NEG\])', ' \\1', pred_sentence)
	pred_sentence 	= re.sub(r'(\[NEG\])(?!\s)', '\\1 ', pred_sentence)
	
	gold_sentence 	= re.sub(r'(?<!\s)(\[NEG\])', ' \\1', gold_sentence)
	gold_sentence 	= re.sub(r'(\[NEG\])(?!\s)', '\\1 ', gold_sentence)
	
	return exact_match(pred_sentence, gold_sentence)

@metric
def src_lang_negation_in_prediction(
	pred_sentence: str, 
	src_lang: str
) -> re.Match:
	'''Is the neg word from the source language in the sentence?'''
	return NEG_REGEXES[src_lang].search(LOWERCASE[src_lang](pred_sentence))

@metric
def tgt_lang_negation_in_prediction(
	pred_sentence: str,
	tgt_lang: str
) -> re.Match:
	'''Is the neg word from the target language in the sentence?'''
	return NEG_REGEXES[tgt_lang].search(LOWERCASE[tgt_lang](pred_sentence))

@metric
def first_word_match(
	pred_sentence: str, 
	gold_sentence: str
) -> bool:
	'''Does the first word of each sentence match?'''
	pred_words = pred_sentence.split()
	gold_words = gold_sentence.split()
	return pred_words[0] == gold_words[0]

@metric
def second_word_match(
	pred_sentence: str, 
	gold_sentence: str
) -> Union[bool,'NoneType']:
	'''Do the second words of each sentence match? If any sentence has only one word, returns None.'''
	
	pred_words = pred_sentence.split()
	gold_words = gold_sentence.split()
	
	if len(pred_words) > 1 and len(gold_words) > 1:
		return pred_words[1] == gold_words[1]

def compute_metrics(
	metrics: List[Callable], 
	pred_file: str, 
	gold_file: str,
) -> Counter:
	'''
	Compute metrics on a prediction file and a gold file.
	
		params:
			metrics (List[Callable]): a list of metrics functions to run on the passed files.
									  (these are defined above in this file).
			pred_file (str)			: a file containing sentences predicted by the model.
			gold_file (str)			: a file containing the target sentences.
									  the pred_file and gold_file should have corresponding 
									  sentences in the same order.
		
		returns:
			props (Dict[str,float])	: a dictionary mapping the name of each metric to the
									  proportion of sentences that pass that metric.
	'''
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
	
	with open(pred_file, 'r') as pred_f, open(gold_file, 'r') as gold_f:
		pred_lines 	= pred_f.readlines()
		gold_lines 	= gold_f.readlines()
	
	pred_lines = format_lines(pred_lines)
	
	if gold_file.endswith('.json'):
		gold_jsons 	= [json.loads(gold_line) for gold_line in gold_lines]
		gold_lines 	= [gold_json['translation']['tgt'] for gold_json in gold_jsons]
		src_lines 	= [gold_json['translation']['src'] for gold_json in gold_jsons]
		src_lines 	= format_lines(src_lines)
	else:
		gold_lines 	= [gold_line.strip().split('\t')[1] for gold_line in gold_lines]
		src_lines 	= None
	
	gold_lines		= format_lines(gold_lines)
	
	src_lang 		= None # placeholder
	tgt_lang 		= None # placeholder
	
	props = {}
	for m in metrics:
		props[metric.__name__] 	= m(
										pred_sentence=pred_lines,
										gold_sentence=gold_lines,
										src_sentence=src_lines,
										src_lang=src_lang,
										tgt_lang=tgt_lang
								)
	
	return props