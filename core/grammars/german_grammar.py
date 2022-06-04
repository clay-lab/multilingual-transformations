# code adapted from Bob Frank's grammars.py
from nltk import Tree, PCFG
from nltk import nonterminals

import random
from typing import *
from .generator import generate
from .generator import create_dataset_json, combine_dataset_jsons
"""
	Create some nonterminals

	S, Sentence: May have preceding AdvP
	S1, Sentence 1: May have following AdvP
	S2, Sentence 2:  Sentence
	AdvP, Adjunct sentence

	NPSgNom, Nominative singular noun phrase (NP)
	NPPlNom, Nominative plural NP
	NPMascSgNom, Nominative singular masculine NP
	NPFemSgNom, Nominative singular feminine NP
	NPMascPlNom, Nominative plural masculine NP
	NPFemPlNom, Nominative plural feminine NP

	MPSg, Singular modal phrase
	MPPl, Plural modal phrase
	MPEmbSg, Singular modal phrase (embedded clause)
	MPEmbPl, Plural modal phrase (embedded clause)

	VP, Verb Phrase
	IVP, Intransitive verb phrase
	TVP, Transitive verb phrase
	TVPMasc, Transitive verb phrase with relative clause on masculine object
	TVPFem, Transitive verb phrase with relative clause on feminine object
	TVPNeut, Transitive verb phrase with relative clause on neuter object

	RelPMasc, Relative clause with masculine pronoun
	RelPFem, Relative clause with feminine pronoun
	RelPNeut, Relative clause with neuter pronoun

	NPAcc, Accusative noun phrase (NP)
	NPMascSgAcc, Accusative singular masculine NP
	NPFemSgAcc, Accusative singular feminine NP
	NPNeutSgAcc, Accusative singular neuter NP
	NPMascPlAcc, Accusative plural masculine NP
	NPFemPlAcc, Accusative plural feminine NP
	NPNeutPlAcc, Accusative plural neuter NP

	DetMascNom, Nominative masculine singular determiner
	DetFemNom, Nominative feminine singular determiner
	DetNeutNom, Nominative neuter singular determiner
	DetPlNom, Nominative plural determiner

	DetMascAcc, Accusative masculine singular determiner
	DetFemAcc, Accusative feminine singular determiner
	DetNeutAcc, Accusative neuter singular determiner
	DetPlAcc, Accusative plural determiner

	NMascSgNom, Nominative masculine singular noun
	NMascPlNom, Nominative masculine plural noun
	NFemSgNom, Nominative feminine singular noun
	NFemPlNom, Nominative feminine plural noun

	NMascSgAcc, Accusative masculine singular noun
	NMascPlAcc, Accusative masculine plural noun
	NFemSgAcc, Accusative feminine singular noun
	NFemPlAcc, Accusative feminine plural noun
	NNeutSgAcc, Accusative neuter singular noun
	NNeutPlAcc, Accusative neuter plural noun

	PN, Name

	PronMascSg, Nominative masculine singular pronoun
	PronFemSg, Nominative feminine singular pronoun

	M3SgPres, 3rd person singular present tense modal
	M3PlPres, 3rd person plural present tense modal

	IV, Intransitive verb
	TV, Transitive verb

	RPMascAcc, Accusative masculine singular relative pronoun
	RPFemAcc, Accusative feminine singular relative pronoun
	RPNeutAcc, Accusative neuter singular relative pronoun
	RPPlAcc, Accusative plural relative pronoun

	Adv, adverbial complementizer

	Neg, negation

	NTand, and

	comma, comma (used to offset AdvPs and RelPs in German)
"""

S, SInv, S1, SInv1, S2, SInv2, AdvP, \
NPSgNom, NPPlNom, \
NPMascSgNom, NPFemSgNom, NPMascPlNom, NPFemPlNom, \
MPSg, MPPl, MPSgInv, MPPlInv, MPEmbSg, MPEmbPl, \
VP, IVP, TVP, TVPMasc, TVPFem, TVPNeut, \
RelPMasc, RelPFem, RelPNeut, \
NPAcc, NPMascSgAcc, NPFemSgAcc, NPNeutSgAcc, NPMascPlAcc, NPFemPlAcc, NPNeutPlAcc = nonterminals(
	'S, SInv, S1, SInv1, S2, SInv2, AdvP, NPSgNom, NPPlNom, NPmascSgNom, ' +
	'NPFemSgNom, NPMascPlNom, NPFemPlNom, MPSg, MPPl, MPSgInv, MPPlInv, MPEmbSg, MPEmbPl, ' +
	'VP, IVP, TVP, TVPMasc, TVPFem, TVPNeut, RelPMasc, RelPFem, RelPNeut, ' +
	'NPAcc, NPMascSgAcc, NPFemSgAcc, NPNeutSgAcc, NPMascPlAcc, NPFemPlAcc, NPNeutPlAcc'
)

nicht_grammar = PCFG.fromstring("""
	S -> AdvP comma SInv1 [0.1] | S1 [0.9]
	SInv -> AdvP comma SInv1 [0.1] | SInv1 [0.9]
	S1 -> S2 comma AdvP [0.1] | S2 [0.9]
	SInv1 -> SInv2 comma AdvP [0.1] | SInv2 [0.9]
	S2 -> NPSgNom MPSg [0.5] | NPPlNom MPPl [0.5]
	SInv2 -> MPSgInv [0.5] | MPPlInv [0.5]
	AdvP -> Adv NPSgNom MPEmbSg [0.45] | Adv NPPlNom MPEmbPl [0.45] | AdvP comma NTand AdvP [0.1]
	
	NPSgNom -> NPMascSgNom [0.4] | NPFemSgNom [0.4] | PN [0.2]
	NPPlNom -> NPMascPlNom [0.5] | NPFemPlNom [0.5]
	
	NPMascSgNom -> DetMascNom NMascSgNom [1.0]
	NPFemSgNom -> DetFemNom NFemSgNom [1.0]
	NPMascPlNom -> DetPlNom NMascPlNom [1.0]
	NPFemPlNom -> DetPlNom NFemPlNom [1.0]
	
	MPSg -> M3SgPres VP [1.0]
	MPPl -> M3PlPres VP [1.0]
	MPSgInv -> M3SgPres NPSgNom VP [1.0]
	MPPlInv -> M3PlPres NPPlNom VP [1.0]
	MPEmbSg -> IVP M3SgPres [0.8] | TVPMasc M3SgPres comma RelPMasc [0.05] | TVPFem M3SgPres comma RelPFem [0.05] | TVPNeut M3SgPres comma RelPNeut [0.05] | TVPPl M3SgPres comma RelPNeut [0.05]
	MPEmbPl -> IVP M3PlPres [0.8] | TVPMasc M3PlPres comma RelPMasc [0.05] | TVPFem M3PlPres comma RelPFem [0.05] | TVPNeut M3PlPres comma RelPNeut [0.05] | TVPPl M3PlPres comma RelPNeut [0.05]
	
	VP -> IVP [0.425] | TVP [0.425] | TVPMasc comma RelPMasc [0.05] | TVPFem comma RelPFem [0.05] | TVPNeut comma RelPNeut [0.05]
	IVP -> IV [1.0]
	TVP -> NPAcc TV [1.0]
	TVPMasc -> NPMascSgAcc TV [1.0]
	TVPFem -> NPFemSgAcc TV [1.0]
	TVPNeut -> NPNeutSgAcc TV [1.0]
	TVPPl -> NPMascPlAcc [0.34] | NPFemPlAcc [0.33] | NPNeutPlAcc [0.33]
	
	RelPMasc -> RPMascAcc NPSgNom TV M3SgPres [0.5] | RPPlAcc NPPlNom TV M3PlPres [0.5]
	RelPFem -> RPFemAcc NPSgNom TV M3SgPres [0.5] | RPPlAcc NPPlNom TV M3PlPres [0.5]
	RelPNeut -> RPNeutAcc NPSgNom TV M3SgPres [0.5] | RPPlAcc NPPlNom TV M3PlPres [0.5]
	
	NPAcc -> NPMascSgAcc [0.17] | NPFemSgAcc [0.17] | NPNeutSgAcc [0.17] | NPMascPlAcc [0.17] | NPFemPlAcc [0.16] | NPNeutPlAcc [0.16]
	
	NPMascSgAcc -> DetMascAcc NMascSgAcc [1.0]
	NPFemSgAcc -> DetFemAcc NFemSgAcc [1.0]
	NPNeutSgAcc -> DetNeutAcc NNeutSgAcc [1.0]
	
	NPMascPlAcc -> DetPlAcc NMascPlAcc [1.0]
	NPFemPlAcc -> DetPlAcc NFemPlAcc [1.0]
	NPNeutPlAcc -> DetPlAcc NNeutPlAcc [1.0]
	
	DetMascNom -> 'der' [0.5] | 'ein' [0.5]
	DetFemNom ->  'die' [0.5] | 'eine' [0.5]
	DetNeutNom -> 'das' [0.5] | 'ein' [0.5]
	DetPlNom -> 'die' [0.5] | 'viele' [0.5]
	
	DetMascAcc -> 'den' [0.5] | 'einen' [0.5]
	DetFemAcc -> 'die' [0.5] | 'eine' [0.5]
	DetNeutAcc -> 'das' [0.5] | 'ein' [0.5]
	DetPlAcc -> 'die' [0.5] | 'viele' [0.5]
	
	NMascSgNom -> 'Student' [0.35] | 'Professor' [0.35] | 'Zauberer' [0.3]
	NMascSgAcc -> 'Kuchen' [0.25] | 'Pfannkuchen' [0.25] | 'Strudel' [0.25] | 'Krapfen' [0.25]
	
	NMascPlNom -> 'Studenten' [0.35] | 'Professoren' [0.35] | 'Zauberer' [0.3]
	NMascPlAcc -> 'Kuchen' [0.25] | 'Pfannkuchen' [0.25] | 'Strudel' [0.25] | 'Krapfen' [0.25]
	
	NFemSgNom -> 'Hexe' [1.0]
	NFemSgAcc -> 'Zuckerstange' [0.5] | 'Baklava' [0.5]
	
	NFemPlNom -> 'Hexen' [1.0]
	NFemPlAcc -> 'Zuckerstangen' [0.5] | 'Baklavas' [0.5]
	
	NNeutSgAcc -> 'Plätzchen' [0.25] | 'Soufflé' [0.25] | 'Eclair' [0.25] | 'Croissant' [0.25]
	NNeutPlAcc -> 'Plätzchen' [0.25] | 'Soufflés' [0.25] | 'Eclairs' [0.25] | 'Croissants' [0.25]
	
	PN -> 'Jonas' [0.1] | 'Franz' [0.1] | 'Alex' [0.1] | 'Lukas' [0.1] | 'Ben' [0.1] | 'Anna' [0.1] | 'Angelika' [0.1] | 'Lola' [0.1] | 'Julia' [0.1] | 'Laura' [0.1]
	
	PronMascSg -> 'er' [1.0]
	PronFemSg -> 'sie' [1.0]
	
	M3SgPres -> 'kann' [0.2] | 'darf' [0.2] | 'muss' [0.3] | 'soll' [0.3]
	M3PlPres -> 'können' [0.2] | 'dürfen' [0.2] | 'müssen' [0.3] | 'sollen' [0.3]
	
	IV -> 'schlucken' [0.1] | 'feiern' [0.1] | 'wackeln' [0.1] | 'lachen' [0.1] | 'lächeln' [0.1] | 'kichern' [0.1] | 'springen' [0.1] | 'rennen' [0.1] | 'laufen' [0.1] | 'schwimmen' [0.1]
	
	TV -> 'zubereiten' [0.1] | 'machen' [0.1] | 'essen' [0.1] | 'bestreuen' [0.1] | 'malen' [0.1] | 'kauen' [0.1] | 'verschlingen' [0.1] | 'zusammenbauen' [0.1] | 'erschaffen' [0.1] | 'verstecken' [0.1]
	
	RPMascAcc -> 'den' [1.0]
	RPFemAcc -> 'die' [1.0]
	RPNeutAcc -> 'das' [1.0]
	RPPlAcc -> 'die' [1.0]
	
	Conj -> 'und' [1.0]
	
	Adv -> 'da' [0.5] | 'weil' [0.5]
	
	comma -> ',' [1.0]
	
""")

setattr(nicht_grammar, 'lang', 'de')
'''
nicht_grammar_no_indef = PCFG.fromstring("""
	S -> AdvP comma SInv1 [0.1] | S1 [0.9]
	SInv -> AdvP comma SInv1 [0.1] | SInv1 [0.9]
	S1 -> S2 comma AdvP [0.1] | S2 [0.9]
	SInv1 -> SInv2 comma AdvP [0.1] | SInv2 [0.9]
	S2 -> NPSgNom MPSg [0.5] | NPPlNom MPPl [0.5]
	SInv2 -> MPSgInv [0.5] | MPPlInv [0.5]
	AdvP -> Adv NPSgNom MPEmbSg [0.45] | Adv NPPlNom MPEmbPl [0.45] | AdvP comma NTand AdvP [0.1]
	
	NPSgNom -> NPMascSgNom [0.4] | NPFemSgNom [0.4] | PN [0.2]
	NPPlNom -> NPMascPlNom [0.5] | NPFemPlNom [0.5]
	
	NPMascSgNom -> DetMascNom NMascSgNom [1.0]
	NPFemSgNom -> DetFemNom NFemSgNom [1.0]
	NPMascPlNom -> DetPlNom NMascPlNom [1.0]
	NPFemPlNom -> DetPlNom NFemPlNom [1.0]
	
	MPSg -> M3SgPres VP [1.0]
	MPPl -> M3PlPres VP [1.0]
	MPSgInv -> M3SgPres NPSgNom VP [1.0]
	MPPlInv -> M3PlPres NPPlNom VP [1.0]
	MPEmbSg -> IVP M3SgPres [0.8] | TVPMasc M3SgPres comma RelPMasc [0.05] | TVPFem M3SgPres comma RelPFem [0.05] | TVPNeut M3SgPres comma RelPNeut [0.05] | TVPPl M3SgPres comma RelPNeut [0.05]
	MPEmbPl -> IVP M3PlPres [0.8] | TVPMasc M3PlPres comma RelPMasc [0.05] | TVPFem M3PlPres comma RelPFem [0.05] | TVPNeut M3PlPres comma RelPNeut [0.05] | TVPPl M3PlPres comma RelPNeut [0.05]
	
	VP -> IVP [0.425] | TVP [0.425] | TVPMasc comma RelPMasc [0.05] | TVPFem comma RelPFem [0.05] | TVPNeut comma RelPNeut [0.05]
	IVP -> IV [1.0]
	TVP -> NPAcc TV [1.0]
	TVPMasc -> NPMascSgAcc TV [1.0]
	TVPFem -> NPFemSgAcc TV [1.0]
	TVPNeut -> NPNeutSgAcc TV [1.0]
	TVPPl -> NPMascPlAcc [0.34] | NPFemPlAcc [0.33] | NPNeutPlAcc [0.33]
	
	RelPMasc -> RPMascAcc NPSgNom TV M3SgPres [0.5] | RPPlAcc NPPlNom TV M3PlPres [0.5]
	RelPFem -> RPFemAcc NPSgNom TV M3SgPres [0.5] | RPPlAcc NPPlNom TV M3PlPres [0.5]
	RelPNeut -> RPNeutAcc NPSgNom TV M3SgPres [0.5] | RPPlAcc NPPlNom TV M3PlPres [0.5]
	
	NPAcc -> NPMascSgAcc [0.17] | NPFemSgAcc [0.17] | NPNeutSgAcc [0.17] | NPMascPlAcc [0.17] | NPFemPlAcc [0.16] | NPNeutPlAcc [0.16]
	
	NPMascSgAcc -> DetMascAcc NMascSgAcc [1.0]
	NPFemSgAcc -> DetFemAcc NFemSgAcc [1.0]
	NPNeutSgAcc -> DetNeutAcc NNeutSgAcc [1.0]
	
	NPMascPlAcc -> DetPlAcc NMascPlAcc [1.0]
	NPFemPlAcc -> DetPlAcc NFemPlAcc [1.0]
	NPNeutPlAcc -> DetPlAcc NNeutPlAcc [1.0]
	
	DetMascNom -> 'der' [1.0]
	DetFemNom ->  'die' [1.0]
	DetNeutNom -> 'das' [1.0]
	DetPlNom -> 'die' [1.0]
	
	DetMascAcc -> 'den' [1.0]
	DetFemAcc -> 'die' [1.0]
	DetNeutAcc -> 'das' [1.0]
	DetPlAcc -> 'die' [1.0]
	
	NMascSgNom -> 'Student' [0.35] | 'Professor' [0.35] | 'Zauberer' [0.3]
	NMascSgAcc -> 'Kuchen' [0.25] | 'Pfannkuchen' [0.25] | 'Strudel' [0.25] | 'Krapfen' [0.25]
	
	NMascPlNom -> 'Studenten' [0.35] | 'Professoren' [0.35] | 'Zauberer' [0.3]
	NMascPlAcc -> 'Kuchen' [0.25] | 'Pfannkuchen' [0.25] | 'Strudel' [0.25] | 'Krapfen' [0.25]
	
	NFemSgNom -> 'Hexe' [1.0]
	NFemSgAcc -> 'Zuckerstange' [0.5] | 'Baklava' [0.5]
	
	NFemPlNom -> 'Hexen' [1.0]
	NFemPlAcc -> 'Zuckerstangen' [0.5] | 'Baklavas' [0.5]
	
	NNeutSgAcc -> 'Plätzchen' [0.25] | 'Soufflé' [0.25] | 'Eclair' [0.25] | 'Croissant' [0.25]
	NNeutPlAcc -> 'Plätzchen' [0.25] | 'Soufflés' [0.25] | 'Eclairs' [0.25] | 'Croissants' [0.25]
	
	PN -> 'Jonas' [0.1] | 'Franz' [0.1] | 'Alex' [0.1] | 'Lukas' [0.1] | 'Ben' [0.1] | 'Anna' [0.1] | 'Angelika' [0.1] | 'Lola' [0.1] | 'Julia' [0.1] | 'Laura' [0.1]
	
	PronMascSg -> 'er' [1.0]
	PronFemSg -> 'sie' [1.0]
	
	M3SgPres -> 'kann' [0.2] | 'darf' [0.2] | 'muss' [0.3] | 'soll' [0.3]
	M3PlPres -> 'können' [0.2] | 'dürfen' [0.2] | 'müssen' [0.3] | 'sollen' [0.3]
	
	IV -> 'schlucken' [0.1] | 'feiern' [0.1] | 'wackeln' [0.1] | 'lachen' [0.1] | 'lächeln' [0.1] | 'kichern' [0.1] | 'springen' [0.1] | 'rennen' [0.1] | 'laufen' [0.1] | 'schwimmen' [0.1]
	
	TV -> 'zubereiten' [0.1] | 'machen' [0.1] | 'essen' [0.1] | 'bestreuen' [0.1] | 'malen' [0.1] | 'kauen' [0.1] | 'verschlingen' [0.1] | 'zusammenbauen' [0.1] | 'erschaffen' [0.1] | 'verstecken' [0.1]
	
	RPMascAcc -> 'den' [1.0]
	RPFemAcc -> 'die' [1.0]
	RPNeutAcc -> 'das' [1.0]
	RPPlAcc -> 'die' [1.0]
	
	NTand -> 'und' [1.0]
	
	Adv -> 'da' [0.5] | 'weil' [0.5]
	
	comma -> ',' [1.0]
	
""")

setattr(nicht_grammar_no_indef, 'lang', 'de')
'''

def negation(grammar: PCFG) -> Tuple:
	source = generate(grammar)
	target = negate(source)
	return source, 'neg', target

def affirmation(grammar: PCFG) -> Tuple:
	source = generate(grammar)
	return source, 'pos', source
	
def neg_or_pos(grammar: PCFG, neg_p: float = 0.5) -> Tuple:
	
	return negation(grammar) if random.random() < neg_p else affirmation(grammar)
	
def negate(t: Tree) -> Tree:
	# Make a deep copy so we don't mess up the original tree
	t_copy = t.copy(deep = True)
	
	# Get the main clause, which is S2 or SInv2
	main_clause = next(
		t_copy.subtrees(
			filter = lambda x: x.label() in [S2, SInv2]
		)	
	)
	
	# see if the subject is indefinite and use 'kein'
	main_clause_subj = next(
		main_clause.subtrees(
			filter = lambda x: x.label() in [NPSgNom, NPPlNom]
		)	
	)
	
	# we use this to catch the StopIteration when there is no determiner
	try:
		main_clause_subj_det = next(
			main_clause_subj.subtrees(
				filter = lambda x: x.label().symbol() in ['DetMascNom', 'DetFemNom', 'DetNeutNom', 'DetPlNom']
			)
		)
		
		if main_clause_subj_det and 'ein' in main_clause_subj_det[0]:
			main_clause_subj_det[0] = 'k' + main_clause_subj_det[0]
			return t_copy
	except StopIteration:
		pass
	
	# if the subject is not indefinite,
	# Get the main clause MP, which is MPSg or MPPl
	# Embedded clauses have the labels MPEmbSg or MPEmbPl
	main_clause_mp = next(
		main_clause.subtrees(
			filter = lambda x: x.label() in [MPSg, MPPl, MPSgInv, MPPlInv]
		)
	)
	
	# Get the main clause VP within the MP
	main_clause_vp = next(
		main_clause_mp.subtrees(
			filter = lambda x: x.label() == VP
		)
	)
	
	# Get the main VP within the main clause VP (to exclude TVs in relative clauses)
	main_clause_itvp = next(
		main_clause_vp.subtrees(
			filter = lambda x: x.label() in [IVP, TVP, TVPMasc, TVPFem, TVPNeut]
		)
	)
	
	# If there is an indefinite object, we do negation with 'kein'
	try: # use try here to catch a StopIteration that occurs when there is no object NP
		main_clause_indef_obj_det = next(
			main_clause_itvp.subtrees(
				filter = lambda x: x.label().symbol() in ['DetMascAcc', 'DetFemAcc', 'DetNeutAcc', 'DetPlAcc']
			)
		)
		
		if main_clause_indef_obj_det and 'ein' in main_clause_indef_obj_det[0]:
			main_clause_indef_obj_det[0] = 'k' + main_clause_indef_obj_det[0]
			return t_copy
		elif main_clause_indef_obj_det and 'viele' == main_clause_indef_obj_det[0]:
			main_clause_indef_obj_det[0] = 'nicht ' + main_clause_indef_obj_det[0]
			return t_copy
	# if we fail above, it's because there is no object with a determiner	
	except StopIteration:
		pass
	
	# Get the verb within the main VP within the main clause VP
	main_clause_v = next(
		main_clause_itvp.subtrees(
			filter = lambda x: x.label().symbol() in ['IV', 'TV']
		)
	)
	
	# Negate it
	main_clause_v[0] = 'nicht ' + main_clause_v[0]
	
	return t_copy

def test_file(grammar: PCFG = nicht_grammar, n: int = 10, filename: str = 'test.txt'):
	"""
	Create a small test file with n pairs of formatted positive and negative sentences
	"""
	s = [negation(grammar) for _ in range(n)]
	with open(filename, 'w') as out:
		for pair in s:
			out.write(' '.join(pair) + '\n\n')