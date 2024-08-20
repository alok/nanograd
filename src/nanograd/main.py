#!/usr/bin/env python3
# %%

from sklearn.datasets import make_moons,make_blobs
import itertools
from copy import deepcopy
from graphviz import Digraph
import matplotlib.pyplot as plt
import os
from functools import partial, lru_cache, wraps
from pathlib import Path
import itertools
import time
from dataclasses import dataclass
from typing import Callable, Any, Iterable, NamedTuple, Self, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from jaxtyping import Float
import random

# %%
# data
# model
# optimizer (gradient descent)
# debugging


class Value:
    """
    A class representing a node in a computational graph, supporting automatic differentiation.

    Attributes:
        data (float): The value stored in this node.
        grad (float): The gradient of this node with respect to the final output.
        _backward (Callable[[], None]): A function to compute the gradient of this node. Not to be confused with the `backward` method, which is the preferred entry point for computing gradients.
        _prev (set[Self]): The set of child nodes in the computational graph.
        _op (str): The operation that produced this node.
        label (str): A label for this node, useful for debugging and visualization.

    Examples:
        >>> x = Value(2.0, label="x")
        >>> y = Value(3.0, label="y")
        >>> z = x * y
        >>> z.data
        6.0
        >>> z.backward()
        >>> x.grad
        3.0
        >>> y.grad
        2.0
    """

    def __init__(
        self,
        data: float,
        grad: float = 0.0,
        _children: tuple[Self, ...] = (),
        _backward: Callable[[], None] = lambda: None,
        _op: str = "",
        label: str = "",
    ):
        self.data: float = data
        self.grad: float = grad
        self._backward = _backward
        self._prev: set[Self] = set(_children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self._op}), label={self.label}, prev={self._prev}"

    def with_label(self, label: str) -> Self:
        """
        MUTABLY create a new Value with the same data but a different label.

        Args:
            label (str): The new label for the Value.

        Returns:
            Self: A new Value instance with the updated label.

        Examples:
            >>> x = Value(2.0, label="x")
            >>> y = x.with_label("y")
            >>> y.data == x.data
            True
            >>> y.label
            'y'
        """
        self.label = label
        return self

    def __add__(self, other: Self | float) -> Self:
        """
        Add this Value to another Value or a float.

        Args:
            other (Self | float): The Value or float to add.

        Returns:
            Self: A new Value representing the sum.

        Examples:
            >>> x = Value(2.0, label="x")
            >>> y = Value(3.0, label="y")
            >>> z = x + y
            >>> z.data
            5.0
            >>> z.label
            'x + y'
        """
        other = other if isinstance(other, Value) else Value(other)
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
        """
        Multiply this Value by another Value or a float.

        Args:
            other (Self | float): The Value or float to multiply by.

        Returns:
            Self: A new Value representing the product.

        Examples:
            >>> x = Value(2.0, label="x")
            >>> y = Value(3.0, label="y")
            >>> z = x * y
            >>> z.data
            6.0
            >>> z.label
            'x * y'
        """
        other = other if isinstance(other, Value) else Value(other)
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

    def __pow__(self, other: Self | float | int) -> Self:
        """
        Raise this Value to a power.

        Args:
            other (Self | float | int): The exponent.

        Returns:
            Self: A new Value representing the result of the power operation.

        Examples:
            >>> x = Value(2.0, label="x")
            >>> y = x ** 3
            >>> y.data
            8.0
            >>> y.label
            'x ** 3'
        """
        if not isinstance(other, (int, float)):
            raise ValueError("other must be an int or float")
        if isinstance(other, int):
            other = float(other)
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data**other.data,
            _children=(self,),
            _op="**",
            label=f"{self.label} ** {other.label}",
        )

        def _backward() -> None:
            self.grad += other.data * self.data ** (other.data - 1) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> Self:
        return self + other

    def __neg__(self) -> Self:
        return (self * -1).with_label(f"-{self.label}")

    def __sub__(self, other: Self | float) -> Self:
        other = other if isinstance(other, Value) else Value(other)
        out = (self + (-other)).with_label(f"{self.label} - {other.label}")
        return out

    def __rsub__(self, other: float) -> Self:
        return (other - self).with_label(f"{other} - {self.label}")

    def __inv__(self) -> Self:
        return (self**-1).with_label(f"1/{self.label}⁻¹")

    def __rmul__(self, other: float) -> Self:
        return self * other

    def __truediv__(self, other: Self | float) -> Self:
        other = other if isinstance(other, Value) else Value(other)
        return (self * other**-1).with_label(f"{self.label} / {other.label}")

    def __rtruediv__(self, other: float) -> Self:
        return Value(other) / self

    def exp(self) -> Self:
        """
        Compute the exponential of this Value.

        Returns:
            Self: A new Value representing exp(self).

        Examples:
            >>> x = Value(1.0, label="x")
            >>> y = x.exp()
            >>> abs(y.data - 2.718281828) < 1e-6
            True
            >>> y.label
            'exp(x)'
        """
        e = math.exp(self.data)
        out = Value(
            data=e,
            _children=(self,),
            _op="exp",
            label=f"exp({self.label})",
        )

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Self:
        """
        Compute the hyperbolic tangent of this Value.

        Returns:
            Self: A new Value representing tanh(self).

        Examples:
            >>> x = Value(0.0, label="x")
            >>> y = x.tanh()
            >>> y.data
            0.0
            >>> y.label
            'tanh(x)'
        """
        t = math.tanh(self.data)
        out = Value(
            data=t,
            _children=(self,),
            _op="tanh",
            label=f"tanh({self.label})",
        )

        def _backward() -> None:
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Perform backpropagation starting from this node.

        This method computes the gradients of all nodes in the computational graph
        with respect to the final output (assumed to be this node).

        Examples:
            >>> x = Value(2.0, label="x")
            >>> y = x * x
            >>> y.backward()
            >>> x.grad
            4.0
        """
        topsorted: list[Self] = []
        seen: set[Self] = set()

        def topo(v: Self) -> None:
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    topo(c)
                topsorted.append(v)
                
        topo(self)

        self.grad = 1.0  # start the grad chain here
        print(f'topsorted: {topsorted}')
        for node in reversed(topsorted):
            node._backward()

    def backward_preorder(self) -> None:
        """
        Perform backpropagation in pre-order traversal.

        This method is an alternative to the standard backward method,
        traversing the graph in pre-order instead of post-order.

        Examples:
            >>> x = Value(2.0, label="x")
            >>> y = x * x
            >>> y.backward_preorder()
            >>> x.grad
            4.0
        """
        topsorted: list[Self] = []
        seen: set[Self] = set()

        def topo(v: Self) -> None:
            if v not in seen:
                seen.add(v)
                topsorted.append(v)
                for c in v._prev:
                    topo(c)

        topo(self)
        self.grad = 1.0
        for node in topsorted:
            node.backward()


# net: weights, input -> output

h = 1e-8


# %%
x = Value(2.0, label="x")
y = x * x
y.grad
y.backward()
y.grad
x.grad
# %%


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
        grad_component = f" | {n.grad:.4f}" if n.grad != 0.0 else ""
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


x = Value(2.0, label="x")
y = Value(3.0, label="y")
z = x * y
z.data  # 6.0
z.backward()
x.grad  # 3.0
y.grad  # 2.0

# %%


e = (a * b).with_label("e")

draw_dot(d)
draw_dot(e)
3.0 * a
# %%
e.grad
# %%
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")
n = (x1 * w1 + x2 * w2 + b).with_label("n")
n.backward()
draw_dot(n)
# %%


class Module:
    def parameters(self) -> Iterable[Value]: ...
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    def __init__(self, in_dim: int, activation_fn: Callable[[Value], Value] = lambda x: x) -> None:
        self.w = [Value(random.uniform(-1,1)) for _ in range(in_dim)]
        self.b: Value = Value(random.uniform(-1, 1))
        self.activation_fn = activation_fn

    def __call__(self, x: list[float]) -> Value:
        pre_act = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), start=self.b)
        out = self.activation_fn(pre_act)
        return out

    def parameters(self):
        return self.w + [self.b]
    

xs = [2.0, -3.0]
n = Neuron(2)
n(xs)
# %%


class Layer(Module):
    def __init__(self, in_dim: int = 1, out_dim: int = 1, activation_fn: Callable[[Value], Value] = Value.tanh) -> None:
        self.neurons = [Neuron(in_dim, activation_fn=activation_fn if i != out_dim - 1 else lambda x: x) for i in range(out_dim)]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __call__(self, x: list[Value]) -> list[Value]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

# %%

#%%
class MLP(Module):
    def __init__(self, in_dims: int, hidden_dims: list[int], out_dims: int) -> None:
        sizes = [in_dims] + hidden_dims + [out_dims]
        self.layers = [
            Layer(in_dim, out_dim) for in_dim, out_dim in itertools.pairwise(sizes)
        ]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x: list[Value]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x


class SGDOptimizer(Module):
    def __init__(self, params: Iterable[Value], lr: float = 0.01) -> None:
        self.params = params
        self.lr = lr

    def parameters(self):
        return self.params

    def step(self) -> None:
        for p in self.parameters():
            data_before = p.data
            p.data -= self.lr * p.grad  # minus to decrease the loss
            data_after = p.data
            if p.grad != 0.0:
                assert data_before != data_after


xs = [[2.0, -3.0], [3.0, -4.0], [4.0, -5.0]]
xs = [[Value(x_i, label=f"{row},{col},{x_i}") for col, x_i in enumerate(row)] for row in xs]

ys = [1.0, -1.0, -2.0]
net = MLP(2, [3, 3], 1)

for epoch in range(5):
    all(v.grad == 0.0 for v in net.parameters())
    preds = [net(x) for x in xs]
    loss = sum((pred_i - y_i) ** 2 for pred_i, y_i in zip(preds, ys))
    net.zero_grad()
    # for w in net.parameters():
    #     w.grad = 0.0
    loss.backward()
    print(f"loss.grad: {loss.grad}")
    for w in net.parameters():
        w.data -= 0.1 * w.grad
    print(net.parameters())
    print(loss.data)


# %%


X,y = make_moons(n_samples=100, noise=0.1)  
y = y*2-1

plt.figure(figsize=(5,5))
plt.scatter(X[:,0],X[:,1],c=y,s=20,cmap='jet')