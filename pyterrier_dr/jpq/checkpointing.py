import json
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
    

def _save_checkpoint(path: str, model, optimizer, epoch: int, step: int, best_metric: float, trainer_self) -> None:
    ckpt = {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": _torch_rng_state(),
        "trainer_meta": {
            "M": trainer_self.M,
            "nbits": trainer_self.nbits,
            "device": str(trainer_self.device),
            "pq_impl": trainer_self.pq_impl,
        },
    }

    # --- PQ centroids learned during quantization
    if hasattr(trainer_self, "pq") and trainer_self.pq is not None:
        ckpt["pq_centroids"] = trainer_self.pq.centroids

    # --- Learned JPQ sub-embeddings from the passage encoder
    if hasattr(model, "passage") and hasattr(model.passage, "sub_embeddings"):
        subembs = [w.weight.detach().cpu() for w in model.passage.sub_embeddings]
        ckpt["jpq_sub_embeddings"] = torch.stack(subembs).numpy()

    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _export_pq(dest: str, ckpt: dict):
    os.makedirs(dest, exist_ok=True)
    meta = ckpt.get("trainer_meta", {})
    meta["epoch"] = ckpt.get("epoch")
    meta["best_metric"] = ckpt.get("best_metric")

    np.save(os.path.join(dest, "pq_centroids.npy"), ckpt.get("pq_centroids")) # type: ignore
    np.save(os.path.join(dest, "jpq_sub_embeddings.npy"), ckpt.get("jpq_sub_embeddings")) # type: ignore
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _load_checkpoint(path: str, model, optimizer) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "rng_state" in ckpt and ckpt["rng_state"]:
        _torch_rng_load(ckpt["rng_state"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("best_metric", float("-inf"))

