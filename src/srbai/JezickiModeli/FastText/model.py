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
    word_vector: np.ndarray,
    network_hyperparams: Tuple[np.ndarray, np.ndarray],
    activation: Callable[[np.ndarray], np.ndarray],
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Does the forward pass for one word of the network.
    Network hyperparameters are in a tuple of input to hidden and hidden to output weights

    Returns a tuple with all the transfer and activation pairs (in order of layers),
    with the last layers' activation being the output
    """
    input_2_hidden_weights, hidden_2_output_weights = network_hyperparams

    input_2_hidden_transfer = word_vector @ input_2_hidden_weights
    input_2_hidden_activation = activation(input_2_hidden_transfer)

    hidden_2_output_transfer = input_2_hidden_activation @ hidden_2_output_weights
    hidden_2_output_activation = softmax(hidden_2_output_transfer)

    return (input_2_hidden_transfer, input_2_hidden_activation), (
        hidden_2_output_transfer,
        hidden_2_output_activation,
    )


def backward_one_word(
    word_vector: np.ndarray,
    context_matrix: np.ndarray,
    forward_results: Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ],
    network_hyperparams: Tuple[np.ndarray, np.ndarray],
    activation_derivative: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes the word vector it's working on currently, as well as its context.
    Also, the results for all steps in the forward pass and the weights for each layer

    Returns gradients for the network hyperparameters in the order of layers
    """
    _, hidden_2_output_weights = network_hyperparams
    (input_2_hidden_transfer, input_2_hidden_activation), (
        _,
        hidden_2_output_activation,
    ) = forward_results

    grad_hidden_2_output_transfer = (
        np.sum(hidden_2_output_activation - context_matrix, axis=0, keepdims=True)
        / context_matrix.shape[0]
    )
    grad_hidden_2_output_weights = (
        input_2_hidden_activation.T @ grad_hidden_2_output_transfer
    )

    grad_input_2_hidden_activation = (
        grad_hidden_2_output_transfer @ hidden_2_output_weights.T
    )
    grad_input_2_hidden_transfer = (
        grad_input_2_hidden_activation * activation_derivative(input_2_hidden_transfer)
    )
    grad_input_2_hidden_weights = word_vector.T @ grad_input_2_hidden_transfer

    return grad_input_2_hidden_weights, grad_hidden_2_output_weights


def loss_one_word(word_output: np.ndarray, context_matrix: np.ndarray) -> float:
    """
    Calculates log loss for a single word based on its output and context
    """
    loss = np.sum(-1 * (context_matrix * np.log(word_output)))
    return loss / context_matrix.shape[0]


def update_hyperparameter(
    hyperparameter: np.ndarray,
    grad_hyperparameter: np.nda