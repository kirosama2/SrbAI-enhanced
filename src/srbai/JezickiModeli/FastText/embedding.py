from typing import List, Dict, Union

import numpy as np

from text_preprocessing import word_to_vector


def _iter_len(str_len: int, ngrams: int) -> int:
    return str_len - (ngrams - 1) if str_len > ngrams else 1


def make_ngram_list_from_word(word: str, ngram_size: int) -> List[str]:
    """
    Returns a list of ngrams as well as the word itself
    """
    return [
        word[i:i + ngram_size] for i in range(0, _iter_len(len(word), ngram_size))
    ]


def count_missing_ngrams(
    word_ngram_list: List[st