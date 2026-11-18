from enum import Enum
import torch


class SimFn(Enum):
    dot = 'dot'
    cos = 'cos'


class Variants(type):
    def __getattr__(cls, name):
        if name in cls.VARIANTS:
            @staticmethod
            def wrapped(*args, **kwargs):
                return cls(cls.VARIANTS[name], *args, **kwargs)
            wrapped = wrapped.__get__(cls)
            wrapped.__doc__ = f"Model: ``{cls.VARIANTS[name]}`` `[link] <https://huggingface.co/{cls.VARIANTS[name]}>`__"
            if name == next(iter(cls.VARIANTS)):
                wrapped.__doc__ = '*(default)* ' + wrapped.__doc__
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


def ensure_faiss_omp_compat():
    """Work around the OpenMP runtime conflict between torch and faiss-wheels on macOS.

    Both ``torch`` and the ``faiss-wheels`` package bundle their own
    ``libomp.dylib``. With two OpenMP runtimes in one process, whichever
    runtime first enters an OpenMP *parallel region* "wins": the other
    runtime then segfaults (or deadlocks) on its first parallel region.
    A faiss configured with a single OpenMP thread never enters a parallel
    region, so the safe configuration is to initialise torch's runtime
    first, and then run faiss single-threaded (verified stable for every
    import order on macOS/ARM; multithreaded faiss in any order was
    observed to crash or hang).

    This workaround sets ``KMP_DUPLICATE_LIB_OK=TRUE`` (unless the user has
    already configured it) so that loading the second libomp does not abort,
    and configures faiss to use a single OpenMP thread, so faiss never
    enters a parallel region and cannot crash as the second runtime.

    **Precondition:** torch must be imported *before* faiss (i.e. before the
    first ``import faiss``), so that torch's libomp is the first runtime
    loaded; importing faiss before torch makes the first torch parallel
    region fatal, which no in-process workaround can fix.
    ``pyterrier_dr`` itself guarantees this order (its package ``__init__``
    imports torch before any faiss import), and this function must be
    called *before the first faiss parallel operation* (train/add/search).

    It is a no-op off macOS, and is safe to call repeatedly (the work
    happens once). The cost on macOS is that faiss's OpenMP parallelism is
    disabled; if you require multithreaded faiss, do not import torch in
    the same process (or set ``KMP_DUPLICATE_LIB_OK=TRUE`` in the
    environment and accept the risk).
    """
    import os
    import sys
    if sys.platform != "darwin":
        return
    if getattr(ensure_faiss_omp_compat, "_initialized", False):
        return
    ensure_faiss_omp_compat._initialized = True
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import faiss
    faiss.omp_set_num_threads(1)


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


def flatnav_available():
    return package_available('flatnav')

def assert_flatnav():
    assert flatnav_available(), "flatnav required; install with instructions here: https://github.com/BlaiseMuhirwa/flatnav"


def kannolo_available():
    return package_available('kannolo')

def assert_kannolo():
    assert kannolo_available(), "kannolo required; install with `pip install kannolo`"  

