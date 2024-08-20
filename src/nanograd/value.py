#!/usr/bin/env python3
# %%
from __future__ import annotations
from copy import deepcopy
from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
import random
from pathlib import Path
import json
import logging
from typing import Any, Collection
import math

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available and set the device
from typing import Self

type ValueLike = float | Value


# %%
@dataclass(unsafe_hash=True)
class Value:
    data: float = 0.0
    grad: float = 0.0
    _backward: Callable[[], None] = lambda: None
    _children: tuple[Self, ...] = ()
    _op: str = ""
    label: str = ""

    @staticmethod
    def to_value(v: Value | float) -> Value:
        return v if isinstance(v, Value) else Value(v)

    def __add__(self, other: Self | float) -> Self:
        other = Value.to_value(other)
        out = Value(
            self.data + other.data,
            _children=(self, other),
            _op="+",
            label=f"{self.label} + {other.label}",
        )

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Self | float) -> Self:
        other = Value.to_value(other)
        out = Value(
            self.data * other.data,
            _children=(self, other),
            _op="*",
            label=f"{self.label} * {other.label}",
        )

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float | Self) -> Self:
        other = Value.to_value(other)
        out = Value(
            self.data**other.data,
            _children=(self,),
            _op="**",
            label=f"{self.label}^{other.label}",
        )

        def _backward() -> None:
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Self | float) -> Self:
        other = Value.to_value(other)
        return self + other

    def __rmul__(self, other: Self | float) -> Self:
        other = Value.to_value(other)
        return self * other

    def __sub__(self, other: Self | float) -> Self:
        other = Value.to_value(other)
        return self + (-other)

    def __rsub__(self, other: Self | float) -> Self:
        other = Value.to_value(other)
        return -self + other

    def __inv__(self) -> Self:
        return self**-1

    def __truediv__(self, other: Self | float) -> Self:
        other = Value.to_value(other)
        return self * other.__inv__()

    def __rtruediv__(self, other: Self | float) -> Self:
        other = Value.to_value(other)
        return self / other

    def __neg__(self) -> Self:
        return -1 * self

    def relu(self) -> Self:
        out = Value(
            max(self.data, 0), _children=(self,), _op="relu", label=f"relu {self.label}"
        )

        def _backward():
            self.grad += self.data * (1 if self.data > 0 else 0)

        out._backward = _backward
        return out

    def with_label_(self, label: str) -> Self:
        self.label = label
        return self

    def with_label(self, label: str) -> Self:
        out = deepcopy(self)
        out.label = label
        return out

    def backward(self) -> None:
        topsorted: list[Self] = []
        seen: set[Self] = set()

        def topsort_(v: Self) -> list[Self]:
            """mutatin  g chainable topsort"""
            if v not in seen:
                seen.add(v)
                for ch in v._children:
                    topsort_(ch)
                topsorted.append(v)
            return topsorted

        topsort_(self)

        # topsort
        self.grad = 1.0
        for node in reversed(topsorted):
            node._backward()


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(self, in_dim: int = 2) -> None:
        idxs = range(in_dim)
        rand = lambda: Value(np.random.uniform(-1, 1))
        self.W: list[Value] = [rand() for _ in idxs]
        self.b: Value = rand()

    def parameters(self) -> list[Value]:
        return self.W + [self.b]

    def __call__(self, xs: list[Value]) -> Value:
        return sum((w * x for w, x in zip(self.W, xs)), start=self.b)


# TODO(alok): here in backward, the gradient is put on the data, but it should be on the FUNCTIONS themselves. watch 'you only linearize once'


class Layer(Module):
    def __init__(self, in_dim: int, out_dim: int):
        O = range(out_dim)

        self.neurons = [Neuron(in_dim=in_dim) for _ in O]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        out: list[Value] = []
        for n in self.neurons:
            for p in n.parameters():
                out.append(p)  # TODO would it break with extend instead of append?
        return out


class Relu(Module):
    def __call__(_self, xs: list[Value]) -> list[Value]:
        return [x.relu() for x in xs]


# this is too much code without testing

a = Value(2.0)
b = Value(3.0)
c = a + b

c.backward()
print(a.grad, b.grad, c.grad)


class MLP(Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        self.layers = [Layer(in_dim, hidden_dim)]
        for _ in range(NUM_HIDDEN := 5):
            self.layers.extend([Relu(), Layer(hidden_dim, hidden_dim)])
        self.layers.extend([Relu(), Layer(hidden_dim, out_dim)])

    def __call__(self, xs):
        for l in self.layers:
            xs = l(xs)
        return xs

    def parameters(self) -> list[Value]:
        out: list[Value] = []
        for l in self.layers:
            for p in l.parameters():  # TODO check if easier way
                out.append(p)
        return out

    def step(self, lr: float = 3e-4) -> None:
        for p in self.parameters():
            p.data -= p.grad * lr


def deep_map[T](func: Callable[[Value], Value], data):
    """
    Recursively applies a function to all non-collection elements in nested iterables.

    Args:
        func (Callable[[T], T]): The function to apply to non-collection elements.
        data (Any): The data structure to traverse.

    Returns:
        Any: A new data structure with the function applied to all non-collection elements.

    Examples:
        >>> deep_map(lambda x: x * 2, [1, [2, 3, [4, 5]], 6])
        [2, [4, 6, [8, 10]], 12]
        >>> deep_map(str.upper, ['a', ['b', 'c'], 'd'])
        ['A', ['B', 'C'], 'D']
    """

    if isinstance(data, Value):
        return func(data)
    elif isinstance(data, Collection) and not isinstance(data, dict):
        return type(data)(deep_map(func, item) for item in data)
    elif isinstance(data, dict):
        return {key: deep_map(func, value) for key, value in data.items()}
    else:
        return func(data)


# Test the function
test_data = [1, [2, 3, [4, 5]], 6]
result = deep_map(lambda x: x * 2, test_data)
print(f"Test result: {result}")

Xs: list[list[Value]] = deep_map(
    Value, np.random.uniform(-1.0, 1.0, size=(4, 4)).tolist()
)
ys = np.random.rand(4).tolist()


def mse(Xs: list[list[Value]], ys: list[float], model: MLP) -> Value:
    preds = [model(x) for x in Xs]

    return sum((p[0] - y) ** 2 for p, y in zip(preds, ys)) / len(Xs)


# %%
model: MLP = MLP(in_dim=4, out_dim=1, hidden_dim=4)
for epoch in range(EPOCHS := 1000):
    model.zero_grad()
    loss = mse(Xs, ys, model)
    loss.backward()
    model.step()
    if epoch % 10 == 0:
        print(loss)
