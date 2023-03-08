# coding=UTF-8
'''
Initial method designed in December 2012, School of Electrical Engineering, Univeristy of Belgrade
Python implementation added on 7 May 2015 at the University of Manchester, School of Computer Science
Updated on 11.16.2021 at Serbian AI Society
@author: Nikola Milosevic
'''
from typing import List

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

rules = {
    'ovnicxki': '',
    'ovnicxka': '',