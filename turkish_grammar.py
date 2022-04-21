import sys
#!{sys.executable} -m pip install nltk

import re
import random

from nltk import PCFG, Tree
from typing import *
from generator import generate, create_dataset_json, combine_dataset_jsons

turkish_grammar = PCFG.fromstring("""
    S -> VP Person [.33]
    S -> NP_nom1 VP Person1 [.23]
    S -> NP_nom2 VP Person2 [.22]
    S -> NP_nom3 VP Person3 [.22]
    VP -> NP_acc V_trans [.5] | V_intrans [.5]
    NP_nom1 -> 'ben' [1]
    NP_nom2 -> 'sen' [1]
    NP_nom3 -> 'o' [.4] | N [.6]
    NP_acc -> N_obj '-i' [.8] | PP N_obj '-i' [.2]
    PP -> N_place P [1]
    P -> '-de-ki' [1]
    N -> 'kedi' [.1] | 'köpek' [.1] | 'veteriner' [.1] | 'memur' [.1] | 'kraliçe' [.1] | 'başkan' [.1] | 'koyun' [.1] | 'yönetmen' [.1] | 'işçi' [.1] | ' balina' [.1]
    N_obj -> ' yemek' [.1] | ' ekmek' [.1] | ' goril' [.1] | ' ayı' [.1] | ' eşek' [.1] | ' ördek' [.1] | ' oklukirpi' [.2] | ' gergeden' [.1] | ' maymun' [.1]
    N_place -> ' masa' [.1] | ' sandalye' [.1] | ' yer' [.1] | ' kitap' [.1] | ' ev' [.1] | ' sokak' [.1] | ' oda' [.1] | ' köşe' [.1] | ' tavan' [.05] | ' kanape' [.05] | ' atölye' [.05] | ' deniz' [.05]
    V_trans -> V_stem_trans Tense [1]
    V_stem_trans -> ' iste' [.05] | ' sev' [.05] | ' ara' [.05]| ' gör' [.1] | ' siparış et' [.05] |' güven' [.05] | ' bul' [.05]| ' emret' [.05] |' tut'[.05]| ' kır'[.05]| ' affet'[.05]| ' özgür bırak' [.05]| ' tercih et'[.05]|' gözlemle' [.05]| ' sakla' [.05]| ' çal' [.05] | ' söv' [.05] | ' kurtar' [.05]| ' uyar' [.05]
    Tense -> '-iyor' [.5] | '-di' [.25] | '-ecek' [.25]
    Person -> Person1 [.2] | Person2 [.2] | Person3 [.6]
    Person1 -> '-im' [1]
    Person2 -> '-sin' [1]
    Person3 -> '-' [1]
    V_intrans -> V_stem_intrans Tense [1]
    V_stem_intrans -> ' dinlen' [.05]| ' git' [.05] | ' öğren' [.05] |  ' izin ver' [.05]|' bekle' [.05]  | ' imdat iste' [.05] | ' özür dile'[.05] | ' oy ver' [.05] | ' gül' [.05] |' şikayet et'[.05]| ' övün'[.05]| ' şaşır'[.05] | ' acele et'[.05]| ' hata yap'[.05]| ' otur'[.05]| ' dur'[.05] | ' bağır' [.05]| ' not al'[.05] | ' yüz'[.05]| ' düşün'[.05]
""")

def vh(expression: str) -> str:
    expression = re.sub('(?<=u.)-i', 'u', expression)
    expression = re.sub('(?<=a.)-i', 'ı', expression)
    expression = re.sub('(?<=ı.)-i', 'ı', expression)
    expression = re.sub('(?<=o.)-i', 'u', expression)
    expression = re.sub('(?<=ü.)-i', 'ü', expression)
    expression = re.sub('(?<=ö.)-i', 'ü', expression)
    # expression = re.sub('(?<=e.)-i', 'u', expression)
    
    # expression = re.sub('(?<=s.)n', 'd', expression)
    expression = re.sub('i-i', 'iyi', expression)
    # expression = re.sub('a-iy', 'ıy', expression)
    
    expression = re.sub('a-i', 'ayı', expression)
    expression = re.sub('ı-i', 'ıyı', expression)
    
    expression = re.sub('i-l', 'iyl', expression)
    expression = re.sub('e-l', 'eyl', expression)
    expression = re.sub('ü-l', 'üyl', expression)
    expression = re.sub('ö-l', 'öyl', expression)
    
    expression = re.sub('ı-le', 'iyla', expression)
    expression = re.sub('a-le', 'ayla', expression)
    expression = re.sub('u-le', 'uyla', expression)
    expression = re.sub('o-le', 'oyla', expression)
    
    expression = re.sub('(?<=ü.)-di', 'dü', expression)
    expression = re.sub('(?<=ö.)-di', 'dü', expression)
    expression = re.sub('(?<=a.)-di', 'dı', expression)
    expression = re.sub('(?<=ı.)-di', 'dı', expression)
    expression = re.sub('(?<=u.)-di', 'du', expression)
    expression = re.sub('(?<=i.)-di', 'di', expression)
    expression = re.sub('(?<=e.)-di', 'di', expression)
    expression = re.sub('(?<=o.)-di', 'du', expression)
    
    expression = re.sub('(?<=ü.)-iyor', 'üyor', expression)
    expression = re.sub('(?<=ö.)-iyor', 'üyor', expression)
    expression = re.sub('(?<=a.)-iyor', 'ıyor', expression)
    expression = re.sub('(?<=ı.)-iyor', 'ıyor', expression)
    expression = re.sub('(?<=u.)-iyor', 'uyor', expression)
    expression = re.sub('(?<=i.)-iyor', 'iyor', expression)
    expression = re.sub('(?<=e.)-iyor', 'iyor', expression)
    expression = re.sub('(?<=o.)-iyor', 'uyor', expression)
    
    expression = re.sub('a-di', 'adı', expression)
    expression = re.sub('a-iyor', 'ıyor', expression)
    
    expression = re.sub('e-di', 'edi', expression)
    expression = re.sub('e-iyor', 'iyor', expression)
    
    expression = re.sub('e-ecek', 'eyecek', expression)
    expression = re.sub('a-ecek', 'ayacak', expression)
    
    expression = re.sub('(?<=ü.)-ecek', 'ecek', expression)
    expression = re.sub('(?<=ö.)-ecek', 'ecek', expression)
    expression = re.sub('(?<=a.)-ecek', 'acak', expression)
    expression = re.sub('(?<=ı.)-ecek', 'acak', expression)
    expression = re.sub('(?<=u.)-ecek', 'acak', expression)
    expression = re.sub('(?<=i.)-ecek', 'ecek', expression)
    expression = re.sub('(?<=e.)-ecek', 'ecek', expression)
    expression = re.sub('(?<=o.)-ecek', 'acak', expression)
    
    expression = re.sub('k-i', 'ği', expression)
    # expression = re.sub('di-sin', 'din', expression)
    expression = re.sub('ör-i', 'örü', expression)
    # expression = re.sub('ör-di', 'ördü', expression)
    # expression = re.sub('a-di', 'adı', expression)
    expression = re.sub('t-d', 'tt', expression)
    expression = re.sub('k-d', 'kt', expression)
    
    expression = re.sub('td', 'tt', expression)
    expression = re.sub('kd', 'kt', expression)
    
    expression = re.sub('ap-de', 'apta', expression)
    
    expression = re.sub('t-i', 'di', expression)
    expression = re.sub('a-de-ki', 'adaki', expression)
    expression = re.sub('ei', 'i', expression)
    expression = re.sub('aı', 'ı', expression)
    expression = re.sub('aı', 'ı', expression)
    expression = re.sub('gite', 'gide', expression)
    expression = re.sub('ete', 'ede', expression)
    
    return expression

def vh_n(expression: str) -> str:
    expression = re.sub('(?<=u.)-i', 'u', expression)
    expression = re.sub('(?<=a.)-i', 'ı', expression)
    expression = re.sub('(?<=ı.)-i', 'ı', expression)
    expression = re.sub('(?<=o.)-i', 'u', expression)
    expression = re.sub('(?<=ü.)-i', 'ü', expression)
    expression = re.sub('(?<=ö.)-i', 'ü', expression)
    # expression = re.sub('(?<=e.)-i', 'u', expression)
    
    # expression = re.sub('(?<=s.)n', 'd', expression)
    expression = re.sub('i-i', 'iyi', expression)
    # expression = re.sub('a-iy', 'ıy', expression)
    
    expression = re.sub('a-i', 'ayı', expression)
    expression = re.sub('ı-i', 'ıyı', expression)
    
    expression = re.sub('i-l', 'iyl', expression)
    expression = re.sub('e-l', 'eyl', expression)
    expression = re.sub('ü-l', 'üyl', expression)
    expression = re.sub('ö-l', 'öyl', expression)
    
    expression = re.sub('ı-le', 'iyla', expression)
    expression = re.sub('a-le', 'ayla', expression)
    expression = re.sub('u-le', 'uyla', expression)
    expression = re.sub('o-le', 'oyla', expression)
    
    expression = re.sub('(?<=ü.)-m--di', 'medi', expression)
    expression = re.sub('(?<=ö.)-m--di', 'medi', expression)
    expression = re.sub('(?<=a.)-m--di', 'madı', expression)
    expression = re.sub('(?<=ı.)-m--di', 'madı', expression)
    expression = re.sub('(?<=u.)-m--di', 'madı', expression)
    expression = re.sub('(?<=i.)-m--di', 'medi', expression)
    expression = re.sub('(?<=e.)-m--di', 'medi', expression)
    expression = re.sub('(?<=o.)-m--di', 'madı', expression)
    
    expression = re.sub('(?<=ü.)-m--iyor', 'müyor', expression)
    expression = re.sub('(?<=ö.)-m--iyor', 'müyor', expression)
    expression = re.sub('(?<=a.)-m--iyor', 'mıyor', expression)
    expression = re.sub('(?<=ı.)-m--iyor', 'mıyor', expression)
    expression = re.sub('(?<=u.)-m--iyor', 'muyor', expression)
    expression = re.sub('(?<=i.)-m--iyor', 'miyor', expression)
    expression = re.sub('(?<=e.)-m--iyor', 'miyor', expression)
    expression = re.sub('(?<=o.)-m--iyor', 'muyor', expression)
    
    expression = re.sub('a-m--di', 'amadı', expression)
    expression = re.sub('a-m--iyor', 'amıyor', expression)
    
    expression = re.sub('e-m--di', 'emedi', expression)
    expression = re.sub('e-m--iyor', 'emiyor', expression)
    
    expression = re.sub('e-m--ecek', 'emeyecek', expression)
    expression = re.sub('a-m--ecek', 'amayacak', expression)
    
    expression = re.sub('(?<=ü.)-m--ecek', 'meyecek', expression)
    expression = re.sub('(?<=ö.)-m--ecek', 'meyecek', expression)
    expression = re.sub('(?<=a.)-m--ecek', 'mayacak', expression)
    expression = re.sub('(?<=ı.)-m--ecek', 'mayacak', expression)
    expression = re.sub('(?<=u.)-m--ecek', 'mayacak', expression)
    expression = re.sub('(?<=i.)-m--ecek', 'meyecek', expression)
    expression = re.sub('(?<=e.)-m--ecek', 'meyecek', expression)
    expression = re.sub('(?<=o.)-m--ecek', 'mayacak', expression)
    
    expression = re.sub('k-i', 'ği', expression)
    # expression = re.sub('di-sin', 'din', expression)
    expression = re.sub('ör-i', 'örü', expression)
    # expression = re.sub('ör-di', 'ördü', expression)
    # expression = re.sub('a-di', 'adı', expression)
    expression = re.sub('t-d', 'tt', expression)
    expression = re.sub('k-d', 'kt', expression)
    
    expression = re.sub('td', 'tt', expression)
    expression = re.sub('kd', 'kt', expression)
    
    expression = re.sub('ap-de', 'apta', expression)
    
    expression = re.sub('t-i', 'di', expression)
    expression = re.sub('a-de-ki', 'adaki', expression)
    expression = re.sub('ei', 'i', expression)
    expression = re.sub('aı', 'ı', expression)
    expression = re.sub('aı', 'ı', expression)
    expression = re.sub('gite', 'gide', expression)
    expression = re.sub('ete', 'ede', expression)
    
    return expression

def vh2(expression: str) -> str:
    expression = re.sub('-', '', expression)
    expression = re.sub('p-d', 'pt', expression)
    expression = re.sub('sd', 'st', expression)
    expression = re.sub('etiyor', 'ediyor', expression)
    expression = re.sub('etece', 'edece', expression)
    # expression = re.sub('a-ecek', 'ayacak', expression)
    expression = re.sub('itiyor', 'idiyor', expression)
    expression = re.sub('^\s', '', expression)
    
    return expression

def vowelharmony(expression: str) -> str:
    # expression = re.sub('di-im', 'di-m', expression)
    # expression = re.sub('di-sin', 'di-n', expression)
    expression = re.sub('(?<=ü.)-di-sin', 'dün', expression)
    expression = re.sub('(?<=ö.)-di-sin', 'dün', expression)
    expression = re.sub('(?<=a.)-di-sin', 'dın', expression)
    expression = re.sub('(?<=ı.)-di-sin', 'dın', expression)
    expression = re.sub('(?<=u.)-di-sin', 'dun', expression)
    expression = re.sub('(?<=i.)-di-sin', 'din', expression)
    expression = re.sub('(?<=e.)-di-sin', 'din', expression)
    expression = re.sub('(?<=o.)-di-sin', 'dun', expression)
    expression = re.sub('(?<=ü.)-di-im', 'düm', expression)
    expression = re.sub('(?<=ö.)-di-im', 'düm', expression)
    expression = re.sub('(?<=a.)-di-im', 'dım', expression)
    expression = re.sub('(?<=ı.)-di-im', 'dım', expression)
    expression = re.sub('(?<=u.)-di-im', 'dum', expression)
    expression = re.sub('(?<=i.)-di-im', 'dim', expression)
    expression = re.sub('(?<=e.)-di-im', 'dim', expression)
    expression = re.sub('(?<=o.)-di-im', 'dum', expression)
    
    expression = re.sub('(?<=ü.)-iyor-sin', 'üyorsun', expression)
    expression = re.sub('(?<=ö.)-iyor-sin', 'üyorsun', expression)
    expression = re.sub('(?<=a.)-iyor-sin', 'ıyorsun', expression)
    expression = re.sub('(?<=ı.)-iyor-sin', 'ıyorsun', expression)
    expression = re.sub('(?<=u.)-iyor-sin', 'uyorsun', expression)
    expression = re.sub('(?<=i.)-iyor-sin', 'iyorsun', expression)
    expression = re.sub('(?<=e.)-iyor-sin', 'iyorsun', expression)
    expression = re.sub('(?<=o.)-iyor-sin', 'uyorsun', expression)
    expression = re.sub('(?<=ü.)-iyor-im', 'üyorum', expression)
    expression = re.sub('(?<=ö.)-iyor-im', 'üyorum', expression)
    expression = re.sub('(?<=a.)-iyor-im', 'ıyorum', expression)
    expression = re.sub('(?<=ı.)-iyor-im', 'ıyorum', expression)
    expression = re.sub('(?<=u.)-iyor-im', 'uyorum', expression)
    expression = re.sub('(?<=i.)-iyor-im', 'iyorum', expression)
    expression = re.sub('(?<=e.)-iyor-im', 'iyorum', expression)
    expression = re.sub('(?<=o.)-iyor-im', 'uyorum', expression)
    
    expression = re.sub('(?<=ü.)-ecek-sin', 'eceksin', expression)
    expression = re.sub('(?<=ö.)-ecek-sin', 'eceksin', expression)
    expression = re.sub('(?<=a.)-ecek-sin', 'acaksın', expression)
    expression = re.sub('(?<=ı.)-ecek-sin', 'acaksın', expression)
    expression = re.sub('(?<=u.)-ecek-sin', 'acaksın', expression)
    expression = re.sub('(?<=i.)-ecek-sin', 'eceksin', expression)
    expression = re.sub('(?<=e.)-ecek-sin', 'eceksin', expression)
    expression = re.sub('(?<=o.)-ecek-sin', 'acaksın', expression)
    expression = re.sub('(?<=ü.)-ecek-im', 'eceğim', expression)
    expression = re.sub('(?<=ö.)-ecek-im', 'eceğim', expression)
    expression = re.sub('(?<=a.)-ecek-im', 'acağım', expression)
    expression = re.sub('(?<=ı.)-ecek-im', 'acağım', expression)
    expression = re.sub('(?<=u.)-ecek-im', 'acağım', expression)
    expression = re.sub('(?<=i.)-ecek-im', 'eceğim', expression)
    expression = re.sub('(?<=e.)-ecek-im', 'eceğim', expression)
    expression = re.sub('(?<=o.)-ecek-im', 'acağım', expression)
    
    expression = re.sub('a-di-im', 'adım', expression)
    expression = re.sub('a-di-sin', 'adın', expression)
    expression = re.sub('a-iyor-sin', 'ıyorsun', expression)
    expression = re.sub('a-iyor-im', 'ıyorum', expression)
    
    expression = re.sub('e-di-im', 'edim', expression)
    expression = re.sub('e-di-sin', 'edin', expression)
    expression = re.sub('e-iyor-sin', 'iyorsun', expression)
    expression = re.sub('e-iyor-im', 'iyorum', expression)
    
    expression = re.sub('a-ecek-sin', 'ayacaksın', expression)
    expression = re.sub('e-ecek-sin', 'eyeceksin', expression)
    
    expression = re.sub('a-ecek-im', 'ayacağım', expression)
    expression = re.sub('e-ecek-im', 'eyeceğim', expression)
    
    # expression = re.sub('e-ec', 'eyec', expression)
    # expression = re.sub('a-ecek', 'ayacak', expression)
    
    expression = re.sub('or-im', 'orum', expression)
    expression = re.sub('or-sin', 'orsun', expression)
    # expression = re.sub('un-iy', 'unu', expression)
    expression = re.sub('un-di', 'undu', expression)
    
    return vh2((vh(expression)))

def vowelharmony_n(expression: str) -> str:
    # expression = re.sub('di-im', 'di-m', expression)
    # expression = re.sub('di-sin', 'di-n', expression)
    expression = re.sub('(?<=ü.)-m--di-sin', 'medin', expression)
    expression = re.sub('(?<=ö.)-m--di-sin', 'medin', expression)
    expression = re.sub('(?<=a.)-m--di-sin', 'madın', expression)
    expression = re.sub('(?<=ı.)-m--di-sin', 'madın', expression)
    expression = re.sub('(?<=u.)-m--di-sin', 'madın', expression)
    expression = re.sub('(?<=i.)-m--di-sin', 'medin', expression)
    expression = re.sub('(?<=e.)-m--di-sin', 'medin', expression)
    expression = re.sub('(?<=o.)-m--di-sin', 'madın', expression)
    expression = re.sub('(?<=ü.)-m--di-im', 'medim', expression)
    expression = re.sub('(?<=ö.)-m--di-im', 'medim', expression)
    expression = re.sub('(?<=a.)-m--di-im', 'madım', expression)
    expression = re.sub('(?<=ı.)-m--di-im', 'madım', expression)
    expression = re.sub('(?<=u.)-m--di-im', 'madım', expression)
    expression = re.sub('(?<=i.)-m--di-im', 'medim', expression)
    expression = re.sub('(?<=e.)-m--di-im', 'medim', expression)
    expression = re.sub('(?<=o.)-m--di-im', 'madım', expression)
    
    expression = re.sub('(?<=ü.)-m--iyor-sin', 'müyorsun', expression)
    expression = re.sub('(?<=ö.)-m--iyor-sin', 'müyorsun', expression)
    expression = re.sub('(?<=a.)-m--iyor-sin', 'mıyorsun', expression)
    expression = re.sub('(?<=ı.)-m--iyor-sin', 'mıyorsun', expression)
    expression = re.sub('(?<=u.)-m--iyor-sin', 'muyorsun', expression)
    expression = re.sub('(?<=i.)-m--iyor-sin', 'miyorsun', expression)
    expression = re.sub('(?<=e.)-m--iyor-sin', 'miyorsun', expression)
    expression = re.sub('(?<=o.)-m--iyor-sin', 'muyorsun', expression)
    expression = re.sub('(?<=ü.)-m--iyor-im', 'müyorum', expression)
    expression = re.sub('(?<=ö.)-m--iyor-im', 'müyorum', expression)
    expression = re.sub('(?<=a.)-m--iyor-im', 'mıyorum', expression)
    expression = re.sub('(?<=ı.)-m--iyor-im', 'mıyorum', expression)
    expression = re.sub('(?<=u.)-m--iyor-im', 'muyorum', expression)
    expression = re.sub('(?<=i.)-m--iyor-im', 'miyorum', expression)
    expression = re.sub('(?<=e.)-m--iyor-im', 'miyorum', expression)
    expression = re.sub('(?<=o.)-m--iyor-im', 'muyorum', expression)
    
    expression = re.sub('(?<=ü.)-m--ecek-sin', 'meyeceksin', expression)
    expression = re.sub('(?<=ö.)-m--ecek-sin', 'meyeceksin', expression)
    expression = re.sub('(?<=a.)-m--ecek-sin', 'mayacaksın', expression)
    expression = re.sub('(?<=ı.)-m--ecek-sin', 'mayacaksın', expression)
    expression = re.sub('(?<=u.)-m--ecek-sin', 'mayacaksın', expression)
    expression = re.sub('(?<=i.)-m--ecek-sin', 'meyeceksin', expression)
    expression = re.sub('(?<=e.)-m--ecek-sin', 'meyeceksin', expression)
    expression = re.sub('(?<=o.)-m--ecek-sin', 'mayacaksın', expression)
    expression = re.sub('(?<=ü.)-m--ecek-im', 'meyeceğim', expression)
    expression = re.sub('(?<=ö.)-m--ecek-im', 'meyeceğim', expression)
    expression = re.sub('(?<=a.)-m--ecek-im', 'mayacağım', expression)
    expression = re.sub('(?<=ı.)-m--ecek-im', 'mayacağım', expression)
    expression = re.sub('(?<=u.)-m--ecek-im', 'mayacağım', expression)
    expression = re.sub('(?<=i.)-m--ecek-im', 'meyeceğim', expression)
    expression = re.sub('(?<=e.)-m--ecek-im', 'meyeceğim', expression)
    expression = re.sub('(?<=o.)-m--ecek-im', 'mayacağım', expression)
    
    expression = re.sub('a-m--di-im', 'amadım', expression)
    expression = re.sub('a-m--di-sin', 'amadın', expression)
    expression = re.sub('a-m--iyor-sin', 'amıyorsun', expression)
    expression = re.sub('a-m--iyor-im', 'amıyorum', expression)
    
    expression = re.sub('e-m--di-im', 'emedim', expression)
    expression = re.sub('e-m--di-sin', 'emedin', expression)
    expression = re.sub('e-m--iyor-sin', 'emiyorsun', expression)
    expression = re.sub('e-m--iyor-im', 'emiyorum', expression)
    
    expression = re.sub('a-m--ecek-sin', 'amayacaksın', expression)
    expression = re.sub('a-m--ecek-im', 'amayacağım', expression)
    
    expression = re.sub('e-m--ecek-sin', 'emeyeceksin', expression)
    expression = re.sub('e-m--ecek-im', 'emeyeceğim', expression)
    
    # expression = re.sub('e-m--ec', 'emeyec', expression)
    # expression = re.sub('a-m--ece', 'amayaca', expression)
    
    expression = re.sub('or-im', 'orum', expression)
    expression = re.sub('or-sin', 'orsun', expression)
    # expression = re.sub('un-iy', 'unu', expression)
    expression = re.sub('un-di', 'undu', expression)
    
    return vh2(vh_n(expression))

def vowelharmony_neg(expression: str) -> str:
    expression = re.sub('mdü', 'medi', expression)
    expression = re.sub('mdı', 'madı', expression)
    expression = re.sub('mdu', 'madı', expression)
    expression = re.sub('mdi', 'medi', expression)   
    expression = re.sub('mtü', 'medi', expression)
    expression = re.sub('mtı', 'madı', expression)
    expression = re.sub('mtu', 'madı', expression)
    expression = re.sub('mti', 'medi', expression)   
    
    expression = re.sub('mece', 'meyec', expression)
    expression = re.sub('mece', 'mayac', expression)  
    
    return expression

def vh_neg(expression: str) -> str:
    
    return vowelharmony_neg(vowelharmony(expression))

'''
def nodash(expression: str) -> str:
    expression = re.sub('-', '', expression)
    return expression
'''

def affirmation(grammar: PCFG) -> Tuple[str]:
    pos_tree = generate(grammar)
    pos = ''.join(pos_tree.leaves())
    source = vowelharmony(pos)
    source = (source[0].upper() if not source[0] == 'i' else 'İ') + source[1:] + '.'
    return source, 'pos', source

def negation(grammar: PCFG) -> Tuple[str]:
    pos_tree = generate(grammar)
    pos = ''.join(pos_tree.leaves())
    
    neg_tree = negate(pos_tree)
    neg = ''.join(neg_tree.leaves())
    
    source = vowelharmony(pos)
    source = (source[0].upper() if not source[0] == 'i' else 'İ') + source[1:] + '.'
    
    target = vowelharmony_n(neg)
    target = (target[0].upper() if not target[0] == 'i' else 'İ') + target[1:] + '.'
    
    return source, 'neg', target

def neg_or_pos(grammar: PCFG, neg_p: float = 0.5) -> Tuple[str]:
    
    return negation(grammar) if random.random() < neg_p else affirmation(grammar)

def negate(t: Tree) -> Tree:
    t = t.copy(deep=True)
    
    symbol = t[0].label().symbol()
    if symbol == 'VP':
        symbol2 = t[0,0].label().symbol()
        if symbol2 == 'NP_acc':
            verbstem = t[0,1,0,0]
            verbstem = verbstem + '-m-'
            t[0,1,0,0] = verbstem    
        else:
            verbstem = t[0,0,0,0]
            verbstem = verbstem + '-m-'
            t[0,0,0,0] = verbstem
    else:
        symbol3 = t[1,0].label().symbol()
        if symbol3 == 'NP_acc':
            verbstem = t[1,1,0,0]
            verbstem = verbstem + '-m-'
            t[1,1,0,0] = verbstem
        else:
            verbstem = t[1,0,0,0]
            verbstem = verbstem + '-m-'
            t[1,0,0,0] = verbstem
    return t