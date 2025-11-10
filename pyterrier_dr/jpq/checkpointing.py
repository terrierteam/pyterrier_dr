import os, random
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
    random.setstate(state["pyhon_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["torch_cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    

def _save_checkpoint(path: str, model, optimizer, epoch: int, step: int, best_metric: float) -> None:
    ckpt = {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": _torch_rng_state(),
    }
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _load_checkpoint(path: str, model, optimizer) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "rng_state" in ckpt and ckpt["rng_state"]:
        _torch_rng_load(ckpt["rng_state"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("best_metric", float("-inf"))

