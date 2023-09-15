# %% (setup)
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import Tensor, optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional, Type
from jaxtyping import Float
from dataclasses import dataclass
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np
from IPython.display import display, HTML

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_optimization"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from plotly_utils import bar, imshow, plot_train_loss_and_test_accuracy_from_trainer
from part3_resnets.solutions import IMAGENET_TRANSFORM, ResNet34, get_resnet_for_feature_extraction
from part4_optimization.utils import plot_fn, plot_fn_with_points
import part4_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


# %% (pathological curvatures)
def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


if MAIN:
    plot_fn(pathological_curve_loss)


# %% opt_fn_with_sgd
def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    optimizer = t.optim.SGD([xy], lr=lr, momentum=momentum)
    contour = t.zeros(n_iters, 2)
    for i in range(n_iters):
        contour[i] = xy.detach()
        loss = fn(xy[0], xy[1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return contour


opt_fn_with_sgd(pathological_curve_loss, t.tensor([2.5, 2.5]).requires_grad_(True), lr=0.4, momentum=0.9)
# %% (plot_fn_with_points)
if MAIN:
    points = []

    optimizer_list = [
        (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
        (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'])

        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)


# %% SGD
class SGD:
    def __init__(
        self, 
        params: Iterable[t.nn.parameter.Parameter], 
        lr: float, 
        momentum: float = 0.0, 
        weight_decay: float = 0.0
    ):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        '''
        self.params = list(params) # turn params into a list (because it might be a generator)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.t = 1
        self.g = [t.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, param in enumerate(self.params):
            g_old = self.g[i]
            self.g[i] = param.grad
            if self.weight_decay != 0:
                self.g[i] += self.weight_decay * param
            if self.momentum != 0 and  self.t > 1:
                self.g[i] += self.momentum * g_old
            self.params[i] -= self.lr * self.g[i]
        self.t += 1

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"



if MAIN:
    tests.test_sgd(SGD)


# %% RMSprop
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

        '''
        self.params = list(params) # turn params into a list (because it might be a generator)
        self.v = [t.zeros_like(param) for param in self.params]
        self.buffer = [t.zeros_like(param) for param in self.params]
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for (i, param) in enumerate(self.params):
            grad = param.grad
            if self.lr != 0:
                grad += self.weight_decay * param
            new_v = self.alpha*self.v[i] + (1-self.alpha)*grad.pow(2)
            if self.momentum > 0:
                self.buffer[i] = self.momentum*self.buffer[i] + grad/(t.sqrt(new_v)+self.eps)
                param -= self.lr*self.buffer[i]
            else:
                param -= self.lr*grad/(t.sqrt(new_v)+self.eps)
            self.v[i] = new_v


    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"



if MAIN:
    tests.test_rmsprop(RMSprop)


# %% Adam
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        self.params = list(params) # turn params into a list (because it might be a generator)
        self.ms = [t.zeros_like(param) for param in self.params]
        self.vs = [t.zeros_like(param) for param in self.params]
        self.buffer = [t.zeros_like(param) for param in self.params]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for (i, (p, m, v)) in enumerate(zip(self.params, self.ms, self.vs)):
            grad = p.grad
            if self.weight_decay != 0:
                grad += self.weight_decay * p
            b1, b2 = self.betas
            self.ms[i] = b1*m + (1-b1)*grad
            self.vs[i] = b2*v + (1-b2)*grad.pow(2)
            m_hat = self.ms[i]/(1-b1**self.t)
            v_hat = self.vs[i]/(1-b2**self.t)
            p -= self.lr*m_hat/(t.sqrt(v_hat)+self.eps)
        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
    tests.test_adam(Adam)


# %% AdamW
class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        '''
        self.params = list(params) # turn params into a list (because it might be a generator)
        self.ms = [t.zeros_like(param) for param in self.params]
        self.vs = [t.zeros_like(param) for param in self.params]
        self.buffer = [t.zeros_like(param) for param in self.params]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        b1, b2 = self.betas
        for (i, (p, m, v)) in enumerate(zip(self.params, self.ms, self.vs)):
            grad = p.grad
            p -= self.lr * self.weight_decay * p
            self.ms[i] = b1*m + (1-b1)*grad
            self.vs[i] = b2*v + (1-b2)*grad.pow(2)
            m_hat = self.ms[i]/(1-b1**self.t)
            v_hat = self.vs[i]/(1-b2**self.t)
            p -= self.lr*m_hat/(v_hat.sqrt()+self.eps)
        self.t += 1

    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


if MAIN:
    tests.test_adamw(AdamW)


# %% implement opt_fn
def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_hyperparams: dict, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    optimizer = optimizer_class([xy], **optimizer_hyperparams)
    contour = t.zeros(n_iters, 2)
    for i in range(n_iters):
        contour[i] = xy.detach()
        loss = fn(xy[0], xy[1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return contour


if MAIN:
    points = []

    optimizer_list = [
        (SGD, {"lr": 0.03, "momentum": 0.99}),
        (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
        (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn(pathological_curve_loss, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params)
        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)


# %%
