#!/usr/bin/env python3
#%%
from __future__ import annotations
from torch import Tensor, vmap, autograd, nn
import gensim.downloader
import torch
from typing import (
    Callable,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    overload,
    Union,
    Literal,
    Any,
    cast,
    overload,
    TYPE_CHECKING,
)
from torch.autograd import Function

vocab_size = 10
embedding_dim = 3
embedding = nn.Embedding(vocab_size, embedding_dim)


type Context = Literal[12_888]

model = gensim.downloader.load("glove-wiki-gigaword-300")

king = model["king"]
queen = model["queen"]

print(king)
print(queen)

torch.softmax([-.8,-5.1,.5,3.4,-2.2,2.4])
# %%

# attention

