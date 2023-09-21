# %% (setup)
import os
import sys
import re
import time
import torch as t
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Iterable, Optional, Union, Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

Arr = np.ndarray
grad_tracking_enabled = True

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part5_backprop', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part5_backprop.tests as tests
from part5_backprop.utils import visualize, get_mnist
from plotly_utils import line

MAIN = __name__ == "__main__"


# %% log_back
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    '''
    return grad_out / x


if MAIN:
    tests.test_log_back(log_back)
# %% unbroadcast
def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    '''
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    '''
    out = broadcasted
    reduced_dims = tuple(range(len(broadcasted.shape) - len(original.shape)))
    out = out.sum(reduced_dims, keepdims=False)

    stretched_dims = tuple([i for i, b in enumerate(original.shape) if b == 1])
    out = out.sum(stretched_dims, keepdims=True)
    return out


if MAIN:
    tests.test_unbroadcast(unbroadcast)


# %% multiply_back
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    '''Backwards function for x * y wrt argument 0 aka x.'''
    if not isinstance(y, Arr):
        y = np.array(y)
    return unbroadcast(grad_out * y, x)

def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    '''Backwards function for x * y wrt argument 1 aka y.'''
    if not isinstance(x, Arr):
        x = np.array(x)
    return unbroadcast(grad_out * x, y)


if MAIN:
    tests.test_multiply_back(multiply_back0, multiply_back1)
    tests.test_multiply_back_float(multiply_back0, multiply_back1)


# %% forward_and_back
def forward_and_back(a: Arr, b: Arr, c: Arr) -> Tuple[Arr, Arr, Arr]:
    '''
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    '''
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)
    grad_out = np.array([1])
    dg_df = log_back(grad_out, g, f)
    dg_dd = multiply_back0(dg_df, f, d, e)
    dg_da = multiply_back0(dg_dd, d, a, b)
    dg_db = multiply_back1(dg_dd, d, a, b)
    dg_de = multiply_back1(dg_df, f, d, e)
    dg_dc = log_back(dg_de, e, c)
    return (dg_da, dg_db, dg_dc)


if MAIN:
    tests.test_forward_and_back(forward_and_back)


# %% === AUTOGRAD ===
# %% (Recipe)
@dataclass(frozen=True)
class Recipe:
    '''Extra information necessary to run backpropagation. You don't need to modify this.'''

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."

    args: tuple
    "The input arguments passed to func."
    "For instance, if func was np.sum then args would be a length-1 tuple containing the tensor to be summed."

    kwargs: Dict[str, Any]
    "Keyword arguments passed to func."
    "For instance, if func was np.sum then kwargs might contain 'dim' and 'keepdims'."

    parents: Dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."


# %% BackwardFuncLookup
class BackwardFuncLookup:
    def __init__(self) -> None:
        self.backward_fns = {}

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
        if not forward_fn in self.backward_fns:
            self.backward_fns[forward_fn] = {}
        self.backward_fns[forward_fn][arg_position] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        if not forward_fn in self.backward_fns or not arg_position in self.backward_fns[forward_fn]:
            return None
        return self.backward_fns[forward_fn]


if MAIN:
    BACK_FUNCS = BackwardFuncLookup()
    BACK_FUNCS.add_back_func(np.log, 0, log_back)
    BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
    BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)

    assert BACK_FUNCS.get_back_func(np.log, 0) == log_back
    assert BACK_FUNCS.get_back_func(np.multiply, 0) == multiply_back0
    assert BACK_FUNCS.get_back_func(np.multiply, 1) == multiply_back1

    print("Tests passed - BackwardFuncLookup class is working as expected!")


# %%
