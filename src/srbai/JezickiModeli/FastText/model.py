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
    return tensor * (tensor > 0.0)


def relu_derivative(tensor: np.ndarray) -> np.ndarray:
    """
    Does the derivative of relu
    """
    return 1.0 * (tensor > 0.0)


def tanh(tensor: np.ndarray) -> np.ndarray:
    """
    Does the hyperbolic tan elementwise on tensor
    """
    return np.tanh(tensor)


def tanh_derivative(tensor: np.ndarray) -> np.ndarray:
    """
    Does the derivative of tanh
    """
    return 1 - tanh(tensor) ** 2


def identity(tensor: np.ndarray) -> np.ndarray:
    """
    Does the y = x function on a tensor
    """
    return tensor


def identity_derivative(tensor: np.ndarray) -> np.ndarray:
    """
    Does the derivative of the identity function
    """
    return np.ones_like(tensor)


def softmax(tensor: np.ndarray) -> np.ndarray:
    """
    Does the safe softmax function with numerical stability
    """
    mxs = np.max(tensor, axis=1).reshape(-1, 1)
    exps = np.exp(tensor - mxs)

    sums = np.sum(exps, axis=1).reshape(-1, 1)

    return exps / sums


def make_network_hyperparameters(
    input_size: int, vector_space_size: int, output_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes random weights and biases for the network
    Returns a pair of weights, input-hidden and hidden-output
    """
    input_2_hidden = np.random.uniform(-1, 1, size=(input_size, vector_space_size))
    hidden_2_output = np.random.uniform(-1, 1, size=(vector_space_size, output_size))
    return input_2_hidden, hidden_2_output


def forward_one_word(
    