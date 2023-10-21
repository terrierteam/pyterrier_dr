from enum import Enum
import torch


class SimFn(Enum):
    dot = 'dot'
    cos = 'cos'


class Variants(type):
    def __getattr__(cls, name):
        if name in cls.VARIANTS:
            def wrapped(*args, **kwargs):
                return cls(cls.VARIANTS[name], *args, **kwargs)
            return wrapped

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)


def infer_device(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def package_available(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def faiss_available():
    return package_available('faiss')


def assert_faiss():
    assert faiss_available(), "faiss required; install using instructions here: <https://github.com/facebookresearch/faiss/blob/main/INSTALL.md>"


def scann_available():
    if not package_available('scann'):
        return False
    import scann
    # version 1.0.0 is requiredl detect via scann.scann_ops_pybind.builder
    return hasattr(scann.scann_ops_pybind, 'builder')


def assert_scann():
    assert scann_available(), "scann==1.0.0 required; install from wheel here: <https://github.com/google-research/google-research/blob/master/scann/docs/releases.md#scann-wheel-archive>"


def voyager_available():
    return package_available('voyager')


def assert_voyager():
    assert voyager_available(), "voyager required; install with `pip install voyager`"
