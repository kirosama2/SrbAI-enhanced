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
                dictionary = new_path + "Resursi"+os.path.sep+"Recnici"+os.path.sep+ "Serbian (La