from typing import Tuple, Dict, Callable, List
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
import tqdm

from globals import ModelConfig
from text_preprocessing import word_to_vector, make_context_matrix


def relu(tensor: np.ndarray) -> np.ndarray:
    """
    Does the relu function.
    f(x) = max(0, x)
    """
    return tenso