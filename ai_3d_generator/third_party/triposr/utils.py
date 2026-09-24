# Copyright (c) 2024 Tripo AI & Stability AI. MIT License (see LICENSE).
# Modified 2026: dataclass configs instead of omegaconf; only the helpers the
# add-on needs are kept.
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    @dataclass
    class Config:
        pass

    cfg: Config

    def __init__(self, cfg: Optional[dict] = None, *args, **kwargs) -> None:
        super().__init__()
        self.cfg = self.Config(**(cfg or {}))
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        raise NotImplementedError


def scale_tensor(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "softplus":
        return lambda x: F.softplus(x)
    try:
        return getattr(F, name)
    except AttributeError:
        raise ValueError(f"Unknown activation function: {name}")


def chunk_batch(func: Callable, chunk_size: int, x: torch.Tensor, check=None):
    """Simplified: func(x) returns a dict of tensors; x is chunked on dim 0.
    check() is called between chunks (used for cancellation)."""
    if chunk_size <= 0:
        return func(x)
    out = {}
    for i in range(0, max(1, x.shape[0]), chunk_size):
        if check is not None:
            check()
        for k, v in func(x[i:i + chunk_size]).items():
            out.setdefault(k, []).append(v.detach())
    return {k: torch.cat(v, dim=0) for k, v in out.items()}
