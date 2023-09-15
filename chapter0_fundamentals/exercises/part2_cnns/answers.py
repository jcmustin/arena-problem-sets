# %% imports
import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"

# %% === EINOPS ===
arr = np.load(section_dir / "numbers.npy")

if MAIN:
  display_array_as_img(arr[0])
# %% einops 1
arr1 = einops.rearrange(arr, 'n rgb r c -> rgb r (n c)')
if MAIN:
  display_array_as_img(arr1)
# %% einops 2
arr2 = einops.repeat(arr[0], 'rgb r c -> rgb (2 r) c')
if MAIN:
  display_array_as_img(arr2)
# %% einops 3
arr3 = einops.repeat(arr[:2], 'n rgb r c -> rgb (n r) (2 c)')
if MAIN:
  display_array_as_img(arr3)
# %% einops 4
arr4 = einops.repeat(arr[0], 'rgb r c -> rgb (r 2) c')
if MAIN:
  display_array_as_img(arr4)
# %% einops 5
arr5 = einops.repeat(arr[0], 'rgb r c -> r (rgb c)')
if MAIN:
  display_array_as_img(arr5)
# %% einops 6
arr6 = einops.repeat(arr, '(c1 c2) rgb r c -> rgb (c1 r) (c2 c)', c2 = 3)
if MAIN:
  display_array_as_img(arr6)
# %% einops 7
arr7 = einops.reduce(arr, 'b c h w -> h (b w)', 'max').astype(int)
if MAIN:
  display_array_as_img(arr7)
# %% einops 8
arr8 = einops.reduce(arr.astype(float), 'b c h w -> h w', 'min').astype(int)
if MAIN:
  display_array_as_img(arr8)
# %% einops 9
arr9 = einops.rearrange(arr[1], 'c h w -> c w h')
if MAIN:
  display_array_as_img(arr9)
# %% einops 10
arr10 = einops.reduce(arr.astype(float), '(c1 c2) rgb (r a) (c b) -> rgb (c1 r) (c2 c)', 'mean', c2 = 3, a = 2, b=2)
if MAIN:
  display_array_as_img(arr10)

# %% einsum
def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, 'i i -> ')

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, 'i j, j -> i')

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, 'i j, j k -> i k')

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, 'i, i -> ')

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, 'i, j -> i j')


if MAIN:
    tests.test_einsum_trace(einsum_trace)
    tests.test_einsum_mv(einsum_mv)
    tests.test_einsum_mm(einsum_mm)
    tests.test_einsum_inner(einsum_inner)
    tests.test_einsum_outer(einsum_outer)


# %% === STRIDES === 
if MAIN:
    test_input = t.tensor(
        [[0, 1, 2, 3, 4], 
        [5, 6, 7, 8, 9], 
        [10, 11, 12, 13, 14], 
        [15, 16, 17, 18, 19]], dtype=t.float
    )
# %% basic strides
import torch as t
from collections import namedtuple


if MAIN:
    TestCase = namedtuple("TestCase", ["output", "size", "stride"])

    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]), 
            size=(4,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 2], [5, 7]]), 
            size=(2, 2),
            stride=(5, 2),
        ),

        TestCase(
            output=t.tensor([0, 1, 2, 3, 4]),
            size=(5,),
            stride=(1,),
        ),

        TestCase(
            output=t.tensor([0, 5, 10, 15]),
            size=(4,),
            stride=(5,),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [5, 6, 7]
            ]), 
            size=(2, 3),
            stride=(5, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [10, 11, 12]
            ]), 
            size=(2, 3),
            stride=(10, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 0, 0], 
                [11, 11, 11]
            ]), 
            size=(2, 3),
            stride=(11, 0),
        ),

        TestCase(
            output=t.tensor([0, 6, 12, 18]), 
            size=(4,),
            stride=(6,),
        ),
    ]

    for (i, test_case) in enumerate(test_cases):
        if (test_case.size is None) or (test_case.stride is None):
            print(f"Test {i} failed: attempt missing.")
        else:
            actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
            if (test_case.output != actual).any():
                print(f"Test {i} failed:")
                print(f"Expected: {test_case.output}")
                print(f"Actual: {actual}\n")
            else:
                print(f"Test {i} passed!\n")
# %% as_strided_trace
def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    row_stride, col_stride = mat.stride()
    return mat.as_strided(size=(mat.shape[0],), stride=(row_stride+col_stride,)).sum()


if MAIN:
    tests.test_trace(as_strided_trace)
# %% as_strided_mv
def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
  '''
  Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
  '''
  r, c = mat.shape
  vec_stride, = vec.stride()
  vec_expanded = vec.as_strided(stride=(0, vec_stride), size=(r, c))
  return (mat * vec_expanded).sum(dim=1)


if MAIN:
  tests.test_mv(as_strided_mv)
  tests.test_mv2(as_strided_mv)

# %% as_strided_mm
def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
  '''
  Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
  '''
  rA, cA = matA.shape
  cA, cB = matB.shape
  rA_stride, cA_stride = matA.stride()
  rB_stride, cB_stride = matB.stride()
  matA_expanded = matA.as_strided(stride=(cA_stride, rA_stride, 0), size=(cA, rA, cB))
  matB_expanded = matB.as_strided(stride=(rB_stride, 0, cB_stride), size=(cA, rA, cB))
  return (matA_expanded * matB_expanded).sum(dim=0)


if MAIN:
  tests.test_mm(as_strided_mm)
  tests.test_mm2(as_strided_mm)


# %% === CONVOLUTIONS ===
# %% conv1d_minimal_simple
def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
  '''
  Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

  Simplifications: batch = input channels = output channels = 1.

  x: shape (width,)
  weights: shape (kernel_width,)

  Returns: shape (output_width,)
  '''
  w = x.shape[0]
  kw = weights.shape[0]
  x_stride, = x.stride()
  kw_stride, = weights.stride()
  x_expanded = x.as_strided(stride=(x_stride, x_stride), size=(w-kw+1, kw))
  w_expanded = weights.as_strided(stride=(0, kw_stride), size=(w-kw+1, kw))
  return einops.reduce(x_expanded*w_expanded, 'ow kw -> ow', 'sum') # note: could've used einsum here


if MAIN:
  tests.test_conv1d_minimal_simple(conv1d_minimal_simple)
# %% conv1d_minimal
def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
  '''
  Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

  x: shape (batch, in_channels, width)
  weights: shape (out_channels, in_channels, kernel_width)

  Returns: shape (batch, out_channels, output_width)
  '''
  b, ic, xw = x.shape
  b_stride, xic_stride, xw_stride = x.stride()
  oc, ic, kw = weights.shape

  x_expanded = x.as_strided(stride=(b_stride, xic_stride, 0, xw_stride, xw_stride), size=(b, ic, oc, xw-kw+1, kw))

  return einops.einsum(x_expanded, weights, 'b ic oc ow kw, oc ic kw -> b oc ow')


if MAIN:
  tests.test_conv1d_minimal(conv1d_minimal)
# %% conv2d_minimal
def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    b, ic, h, w = x.shape
    b_stride, xic_stride, xh_stride, xw_stride = x.stride()
    oc, _, kh, kw = weights.shape

    x_strided = x.as_strided(stride=(b_stride, xic_stride, xh_stride, xw_stride, xh_stride, xw_stride), size=(b, ic, h-kh+1, w-kw+1, kh, kw))

    return einops.einsum(x_strided, weights, 'b ic oh ow kh kw, oc ic kh kw -> b oc oh ow')


if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)
# %% pad1d
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    b, ic, w = x.shape
    out = x.new_full(size=(b, ic, w+left+right), fill_value=pad_value)
    out[..., left:left+w] = x
    return out

if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)

# %% pad2d
def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    b, ic, h, w = x.shape
    out = x.new_full(size=(b, ic, h+top+bottom, w+left+right), fill_value=pad_value)
    out[:, :, top:top+h, left:left+w] = x
    return out


if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)
# %% conv1d
def conv1d(
    x: Float[Tensor, "b ic w"], 
    weights: Float[Tensor, "oc ic kw"], 
    stride: int = 1, 
    padding: int = 0
) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    x_padded = pad1d(x, left=padding, right=padding, pad_value=0)
    b, ic, w = x_padded.shape
    b_stride, ic_stride, w_stride = x_padded.stride()
    oc, _, kw = weights.shape

    x_strided=x_padded.as_strided(stride=(b_stride, ic_stride, w_stride*stride, 0, w_stride), size=(b, ic, (w-kw)//stride+1, oc, kw))
    return einops.einsum(x_strided, weights, 'b ic ow oc kw, oc ic kw -> b oc ow')


if MAIN:
    tests.test_conv1d(conv1d)


# %% force_pair
IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

# Examples of how this function can be used:


if MAIN:
    for v in [(1, 2), 2, (1, 2, 3)]:
        try:
            print(f"{v!r:9} -> {force_pair(v)!r}")
        except ValueError:
            print(f"{v!r:9} -> ValueError")


# %% conv2d
def conv2d(
    x: Float[Tensor, "b ic h w"], 
    weights: Float[Tensor, "oc ic kh kw"], 
    stride: IntOrPair = 1, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    h_stride, w_stride = force_pair(stride)
    h_padding, w_padding = force_pair(padding)
    x_padded = pad2d(x, w_padding, w_padding, h_padding, h_padding, 0)
    b, ic, h, w = x_padded.shape
    b_stride, ic_stride, xh_stride, xw_stride = x_padded.stride()
    oc, ic2, kh, kw = weights.shape
    assert ic == ic2, "in_channels for x and weights don't match up"

    oh = (h-kh)//h_stride + 1
    ow = (w-kw)//w_stride + 1
    x_strided = x_padded.as_strided(stride=(b_stride, ic_stride, xh_stride*h_stride, xw_stride*w_stride, 0, xh_stride, xw_stride), size=(b, ic, oh, ow, oc, kh, kw))

    return einops.einsum(x_strided, weights, 'b ic oh ow oc kh kw, oc ic kh kw -> b oc oh ow')

if MAIN:
    tests.test_conv2d(conv2d)


# %% maxpool2d
def maxpool2d(
    x: Float[Tensor, "b ic h w"], 
    kernel_size: IntOrPair, 
    stride: Optional[IntOrPair] = None, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
    '''
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    kh, kw = force_pair(kernel_size)
    h_stride, w_stride = force_pair(stride) if stride else (kh, kw)
    h_padding, w_padding = force_pair(padding)

    x_padded = pad2d(x, w_padding, w_padding, h_padding, h_padding, -float("inf"))
    b, ic, h, w = x_padded.shape
    b_stride, ic_stride, xh_stride, xw_stride = x_padded.stride()

    oh = (h-kh)//h_stride + 1
    ow = (w-kw)//w_stride + 1

    x_strided = x_padded.as_strided(stride=(b_stride, ic_stride, xh_stride*h_stride, xw_stride*w_stride, xh_stride, xw_stride), size=(b, ic, oh, ow, kh, kw))
    x_pooled = x_strided.amax((-1, -2))

    return x_pooled


if MAIN:
    tests.test_maxpool2d(maxpool2d)


# %% == CUSTOM MODULES
# %% MaxPool2d

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")


# %% Relu
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x.maximum(x.new_full(x.size(), 0)) # t.tensor(0) would have been enough, b.c. broadcasting


if MAIN:
    tests.test_relu(ReLU)


# %% Flatten
class Flatten(nn.Module):
  def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
    super().__init__()
    self.start_dim = start_dim
    self.end_dim = end_dim


  def forward(self, input: t.Tensor) -> t.Tensor:
    '''
    Flatten out dimensions from start_dim to end_dim, inclusive of both.
    '''
    stop = None if self.end_dim == -1 else self.end_dim+1
    dims = input.shape
    flattened_dims = t.Size([t.tensor(dims[self.start_dim:stop]).prod()])
    size = dims[:self.start_dim]+flattened_dims+dims[self.end_dim:][1:]

    stride = input.stride()
    new_stride = stride[:self.start_dim] + (stride[self.end_dim],) + stride[self.end_dim:][1:]
    return input.as_strided(size=size, stride=new_stride)

  def extra_repr(self) -> str:
    return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])


if MAIN:
  tests.test_flatten(Flatten)


# %% Linear
class Linear(nn.Module):
  def __init__(self, in_features: int, out_features: int, bias=True):
    '''
    A simple linear (technically, affine) transformation.

    The fields should be named `weight` and `bias` for compatibility with PyTorch.
    If `bias` is False, set `self.bias` to None.
    '''
    super().__init__()
    bounds = (-1/np.sqrt(in_features), 1/np.sqrt(in_features))

    params = t.zeros(out_features, in_features).uniform_(*bounds)

    self.weight = nn.Parameter(params)
    self.bias = nn.Parameter(t.ones(out_features)) if bias else None

  def forward(self, x: t.Tensor) -> t.Tensor:
    '''
    x: shape (*, in_features)
    Return: shape (*, out_features)
    '''
    prod = einops.einsum(self.weight, x, 'i k, j k -> j i')
    return prod if self.bias is None else prod + self.bias

  def extra_repr(self) -> str:
    return ", ".join([f"{key}={getattr(self, key)}" for key in ["in_features", "out_features", "bias"]])


if MAIN:
  tests.test_linear_forward(Linear)
  tests.test_linear_parameters(Linear)
  tests.test_linear_no_bias(Linear)


# %% Conv2d
class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        kh, kw = force_pair(kernel_size)
        sf = 1 / np.sqrt(in_channels)
        self.weight = nn.Parameter(sf * (2*t.rand((out_channels, in_channels, kh, kw))-1))
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["weight", "stride", "padding"]])


if MAIN:
    tests.test_conv2d_module(Conv2d)

# %%
