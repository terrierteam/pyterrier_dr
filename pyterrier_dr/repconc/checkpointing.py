"""Checkpointing for RepCONC training.

Mirrors :mod:`pyterrier_dr.jpq.checkpointing`: a single ``torch.save`` dict
containing the (learnable) PQ state, the biencoder's torch module state
dicts (discovered by attribute name, deduplicated by identity), the optimizer
state, RNG state and trainer metadata. The biencoder is an external object
(not an ``nn.Module`` owned by the model), so its modules are saved
individually by name.
"""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _torch_rng_state() -> dict[str, Any]:
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _torch_rng_load(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["torch_cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])


def _save_checkpoint(path: str, model, optimizer, step: int, best_metric: float, trainer_self) -> None:
    ckpt = {
        "step": step,
        "best_metric": best_metric,
        "pq_state_dict": model.pq.state_dict(),
        "encoder_state_dicts": {
            name: module.state_dict() for name, module in model.encoder_modules()
        },
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "rng_state": _torch_rng_state(),
        "trainer_meta": {
            "M": trainer_self.M,
            "nbits": trainer_self.nbits,
            "d": trainer_self.d,
            "device": str(trainer_self.device),
            "sk_epsilon": trainer_self.sk_epsilon,
            "sk_iters": trainer_self.sk_iters,
        },
    }
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _load_checkpoint(path: str, model, optimizer=None, trainer_self=None) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.pq.load_state_dict(ckpt["pq_state_dict"])
    encoder_states = ckpt.get("encoder_state_dicts", {})
    for name, module in model.encoder_modules():
        if name in encoder_states:
            module.load_state_dict(encoder_states[name])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("rng_state"):
        _torch_rng_load(ckpt["rng_state"])
    return ckpt.get("step", 0), ckpt.get("best_metric", float("-inf"))
