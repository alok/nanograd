#!/usr/bin/env python3
# %%

import sys
from copy import deepcopy
from graphviz import Digraph
import matplotlib.pyplot as plt
import os
from functools import partial, lru_cache, wraps
from pathlib import Path
import itertools
import time
from dataclasses import dataclass
from typing import Callable, Any, NamedTuple, Self
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from jaxtyping import Float

# %%
# data
# model
# optimizer (gradient descent)
# debugging


class Value:
    def __init__(
        self,
        data: float,
        grad: Callable[[None], float] = lambda: 0.0,
        _children=(),
        _op: str = "",
        label: str = "",
    ):
        self.data: float = data
        self.grad: Callable[[None], float] = grad
        print(f"grad: {self.grad}")
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self._op}), label={self.label}, prev={self._prev}"

    def with_label(self, label: str) -> Self:
        out = deepcopy(self)
        out.label = label
        return out

    def __add__(self, other) -> Self:
        return Value(
            self.data + other.data,
            grad=lambda: self.grad() + other.grad(),
            _children=(self, other),
            _op="+",
            label=f"{self.label}+{other.label}",
        )

    def __neg__(self) -> Self:
        return Value(
            -self.data,
            lambda: -self.grad(),
            _children=(self,),
            _op="negate",
        )

    def __sub__(self, other) -> Self:
        return Value(
            self.data - other.data,
            grad=lambda: self.grad() - other.grad(),
            _children=(self, other),
            _op="-",
        )

    def __inv__(self) -> Self:
        return Value(
            1 / self.data,
            lambda: -self.grad() / self.data**2,
            _children=(self,),
            _op="/",
        )

    def __mul__(self, other) -> Self:
        return Value(
            self.data * other.data,
            grad=lambda: self.grad() * other.data + self.data * other.grad(),
            _children=(self, other),
            _op="*",
        )

    def __truediv__(self, other) -> Self:
        return self * other.__inv__()


# net: weights, input -> output

h = 1e-8


def finite_diff(f, x, ε: float = h):
    """compute derivative by centered finite difference approximation"""
    return (f((1 + ε) * x) - f((1 - ε) * x)) / (2 * ε)


def f(x):
    return 3 * x**2 - 4 * x + 5


def f_d(x):
    return 6 * x - 4


xs = np.arange(-5, 5, 0.25)
ys = f(xs)

fig, ax = plt.subplots()
ax.plot(xs, ys)
# %%
a = 2.0
b = -3.0
c = 10.0
d1 = a * b + c
d2 = (a + h) * b + c
print((d2 - d1) / h)
# %%

d1 = a * b + c
d2 = a * b + (c + h)
print((d2 - d1) / h)

x = [1, 2]


# def net(weights):
#     return lambda input: weights @ relu(weights @ relu(weights @ input))


# new_weights = old_weights + learning_rate * grad_wrt_weights

# def loss(guess, true_output)-> float:
#     return mean((guess - true_output) ** 2)


# %%
a = Value(2.0, label="a")
b = Value(-3, label="b")
c = Value(10, label="c")
# %%
d = (a * b + c).with_label("d")
type Node = Value
type Edge = tuple[Value, Value]


# %%
def trace(root: Value):
    nodes: set[Node] = set()
    edges: set[Edge] = set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def trace_func(
    root: Value, nodes: set[Node] | None = None, edges: set[Edge] | None = None
):
    if nodes is None:
        nodes = set()
    if edges is None:
        edges = set()

    if root not in nodes:
        nodes.add(root)
        for c in root._prev:
            edges.add((c, root))
            nodes, edges = trace_func(c, nodes, edges)
    return nodes, edges


trace(d)
trace_func(d)

# %%


def draw_dot(root):
    dg = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        grad_component = f" | {n.grad():.4f}" if n.grad() != 0.0 else ""
        label_component = f"{n.label}{' | ' if n.label else ''}"
        dg.node(
            name=uid,
            label=f"{label_component}{n.data:.4f}{grad_component}",
            shape="record",
        )
        if n._op:
            new_name = uid + n._op
            dg.node(name=new_name, label=n._op)  # connect
            dg.edge(new_name, uid)

    for n1, n2 in edges:
        # connect to op node
        dg.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dg


e = (a * b).with_label("e")
e
draw_dot(d)
draw_dot(e)

