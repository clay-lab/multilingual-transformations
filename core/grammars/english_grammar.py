# code adapted from Bob Frank's grammars.py
from nltk import Tree, PCFG
from nltk import nonterminals

import random
from typing import *
from .generator import generate, format_tree_string
from .generator import create_dataset_json, combine_dataset_jsons
"""
	Create some nonterminals

	S, Sentence: May have preceding AdvP
	S1, Sentence 1: May have following AdvP
	S2, Sentence 2:  Sentence
	AdvP, Adjunct sentence

	NPSgNom, Nominative singular noun phrase (NP)
	NPPlNom, Nominative plural NP
	
	MP, Modal phrase
	
	VP, Verb Phrase
	IVP, Intransitive verb phrase
	TVP, Transitive verb phrase
	
	RelP, Relative clause with masculine pronoun
	
	NPAcc, Accusative noun phrase (NP)
	NPSgAcc, Accusative singular NP
	NPPlAcc, Accusative plural NP
	
	DetSg, Singular determiner
	DetPl, Plural determiner
	
	NSgNom, Nominative singular noun
	NPlNom, Nominative plural noun
	
	NSgAcc, Accusative singular noun
	NPlAcc, Accusative plural noun
	
	PN, Name
	
	M, Modal
	
	IV, Intransitive verb
	TV, Transitive verb
	
	RP, Relative pronoun
	
	Adv, adverbial complementizer
	
	Neg, negation
	
	NTand, and
	
	comma, comma (used to offset AdvPs)
"""

S, S1, S2, AdvP, NPSgNom, NPPlNom, MP, VP, IVP, TVP, RelP, NPAcc, NPSgAcc, NPPlAcc = nonterminals(
	'S, S1, S2, AdvP, NPSgNom, NPPlNom, MP, VP, IVP, TVP, RelP, NPAcc, NPSgAcc, NPPlAcc'
)

not_grammar = PCFG.fromstring("""
	S -> AdvP comma S [0.1] | S1 [0.9]
	S1 -> S1 comma AdvP [0.1] | S2 [0.9]
	S2 -> NPNom MP [1.0]
	AdvP -> Adv NPNom MP [0.9] | AdvP comma Conj AdvP [0.1]
	
	NPNom -> NPSgNom [0.4] | NPPlNom [0.4] | PN [0.2]
	
	NPSgNom -> DetSg NSgNom [1.0]
	NPPlNom -> DetPl NPlNom [1.0]
	
	MP -> M VP [1.0]
	
	VP -> IV [0.5] | TV NPAcc [0.5]
	
	RelP -> RP NPNom M TV [1.0]
	
	NPAcc -> NPSgAcc [0.4] | NPPlAcc [0.4] | NPSgAcc RelP [0.1] | NPPlAcc RelP [0.1]
	
	NPSgAcc -> DetSg NSgAcc [1.0]
	NPPlAcc -> DetPl NPlAcc [1.0]
	
	DetSg -> 'the' [0.5] | 'a' [0.5]
	DetPl -> 'the' [0.4] | 'some' [0.4] | '' [0.2] | 'any' [0.0]
	
	NSgNom -> 'student' [0.25] | 'professor' [0.25] | 'wizard' [0.25] | 'witch' [0.25]
	NSgAcc -> 'cake' [0.1] | 'pancake' [0.1] | 'strudel' [0.1] | 'donut' [0.1] | 'candy' [0.1] | 'baklava' [0.1] | 'cookie' [0.1] | 'waffle' [0.1] | 'pastry' [0.1] | 'croissant' [0.1]
	
	NPlNom -> 'students' [0.25] | 'professors' [0.25] | 'wizards' [0.25] | 'witches' [0.25] 
	NPlAcc -> 'cakes' [0.1] | 'pancakes' [0.1] | 'strudel' [0.1] | 'donuts' [0.1] | 'candy' [0.1] | 'baklava' [0.1] | 'cookies' [0.1] | 'waffles' [0.1] | 'pastries' [0.1] | 'croissants' [0.1]
	
	PN -> 'John' [0.1] | 'Frank' [0.1] | 'Zachary' [0.1] | 'Luke' [0.1] | 'Ben' [0.1] | 'Sue' [0.1] | 'Beth' [0.1] | 'Lily' [0.1] | 'Julia' [0.1] | 'Laura' [0.1]
	
	M -> 'can' [0.25] | 'may' [0.25] | 'must' [0.25] | 'should' [0.25]
	
	IV -> 'drink' [0.1] | 'celebrate' [0.1] | 'wiggle' [0.1] | 'laugh' [0.1] | 'smile' [0.1] | 'giggle' [0.1] | 'jump' [0.1] | 'run' [0.1] | 'walk' [0.1] | 'swim' [0.1]
	
	TV -> 'prepare' [0.1] | 'make' [0.1] | 'eat' [0.1] | 'decorate' [0.1] | 'paint' [0.1] | 'chew' [0.1] | 'devour' [0.1] | 'assemble' [0.1] | 'create' [0.1] | 'hide' [0.1]
	
	RP -> 'that' [0.5] | '' [0.5]
	
	Conj -> 'and' [1.0]
	
	Adv -> 'because' [0.5] | 'since' [0.5]
	
	comma -> ',' [1.0]
""")

setattr(not_grammar, 'lang', 'en')

def negation(grammar: PCFG) -> Tuple:
	pos_tree = generate(grammar)
	neg_tree = negate(pos_tree)
	return pos_tree, 'neg', neg_tree

def affirmation(grammar: PCFG) -> Tuple:
	pos_tree = generate(grammar)
	return pos_tree, 'pos', pos_tree

def neg_or_pos(grammar: PCFG, neg_p: float = 0.5) -> Tuple:
	
	return negation(grammar) if random.random() < neg_p else affirmation(grammar)

def negate(t: Tree) -> Tree:
	# Make a deep copy so we don't mess up the original tree
	t_copy = t.copy(deep=True)
	
	# Get the main clause, which is S2
	main_clause = next(
		t_copy.subtrees(
			filter = lambda x: x.label() == S2
		)
	)
	
	# Get the main clause MP
	main_clause_mp = next(
		main_clause.subtrees(
			filter = lambda x: x.label() == MP
		)
	)
	
	# Get the modal within the main clause MP
	main_clause_m = next(
		main_clause_mp.subtrees(
			filter = lambda x: x.label().symbol() == 'M'
		)
	)
	
	# Negate it
	if main_clause_m[0] != 'can':
		main_clause_m[0] += ' not'
	else:
		main_clause_m[0] += 'not'
	
	# convert any some to any in the scope of negation
	try:
		main_clause_obj_det = next(
			main_clause_mp.subtrees(
				filter = lambda x: x.label().symbol() == 'DetPl'
			)
		)
		
		if main_clause_obj_det[0] == 'some':
			main_clause_obj_det[0] = 'any'
	except StopIteration:
		pass
	
	return t_copy

def test_file(grammar: PCFG = not_grammar, n: int = 10, filename: str = 'test.txt'):
	"""
	Create a small test file with n pairs of formatted positive and negative sentences
	"""
	s = [negation(grammar) for _ in range(n)]
	s = [(format_tree_string(t[0]), t[1], format_tree_string(t[2])) for t in s]
	with open(filename, 'w') as out:
		for pair in s:
			out.write(' '.join(pair) + '\n\n')