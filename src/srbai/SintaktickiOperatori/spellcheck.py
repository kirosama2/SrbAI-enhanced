import os

import numpy as np
from collections import defaultdict
import time


class SpellCheck:
    def __init__(self, dictionary):
        if dictionary in ['sr-latin','sr-cirilic']:
            path = os.path.abspath(__file__).split(os.path.sep)
            path_len = len(path)
            i = 0
            new_path = ""
            for p in path:
                if i<path_len-2:
                    if p == 'SintaktickiOperatori':
                        continue
                    new_path = new_path + p + os.path.sep
                    i = i + 1
            if dictionary == 'sr-latin':
                dictionary = new_path + "Resursi"+os.path.sep+"Recnici"+os.path.sep+ "Serbian (Latin).dic"
                dictionary = dictionary.replace(os.path.sep+os.path.sep,os.path.sep)
            if dictionary == 'sr-cirilic':
                dictionary = new_path + "Resursi"+os.path.sep+"Recnici"+os.path.sep+"Serbian (Cyrilic).dic"
                dictionary = dictionary.replace(os.path.sep + os.path.sep, os.path.sep)

        self.trie = {}
        self.max_cost = 2
        self.load_dictionary(dictionary)


    def load_dictionary(self, dictionary):
        with open(dictionary, 'r') as f:
            for line in f:
                self.add_word(line.strip())

    def add_word(self, word):
        node = self.trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = word

    def search_trie(self, node, word, cost, results):
        if '$' in node:
            results.append((node['$'], cost))
        if not word:
            return
        if cost > self.max_cost:
            return
        for c in node:
            if c == '$':
                continue
            if c == word[0]:
                self.search_trie(node[c], word[1:], cost, results)
            else:
                self.search_trie(node[c], word[1:], cost + 1, results)
                self.search_trie(node[c], word, cost + 1, results)
                self.search_trie(node[c], word[1:], cost + 1, results)
    def lev_distance(self,s, t):
        m, n = len(s), len(t)
        d = np.zeros((m+1,