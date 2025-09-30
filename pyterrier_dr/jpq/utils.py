from contextlib import contextmanager
import time
import numpy as np
import os

@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[TIMER] {name}: {dt/60:.2f} min ({dt:.1f} s)")

def l2_normalize_np(arr: np.ndarray) -> np.ndarray:
    X = np.array(arr, dtype=np.float32, copy=True)
    X /= (np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32) + 1e-12)
    return X

def bytes_to_gb(nbytes: int) -> float:
    return nbytes / (1024**3)

def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total