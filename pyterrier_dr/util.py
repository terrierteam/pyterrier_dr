from enum import Enum

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

def infer_device(device):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)
