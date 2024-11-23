import numpy as np
import torch
import pyterrier as pt
from transformers import AutoConfig
from .biencoder import BiEncoder, BiQueryEncoder
from .util import Variants
from tqdm import tqdm

def _sbert_encode(self, texts, batch_size=None):
    show_progress = False
    if isinstance(texts, tqdm):
        texts.disable = True
        show_progress = True
    texts = list(texts)
    if len(texts) == 0:
        return np.empty(shape=(0, 0))
    return self.model.encode(texts, batch_size=batch_size or self.batch_size, show_progress_bar=show_progress)


class SBertBiEncoder(BiEncoder):
    def __init__(self, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name).to(self.device).eval()
        self.config = AutoConfig.from_pretrained(model_name)

    encode_queries = _sbert_encode
    encode_docs = _sbert_encode

    def __repr__(self):
        return f'SBertBiEncoder({repr(self.model_name)})'


class _SBertBiEncoder(SBertBiEncoder, metaclass=Variants):
    VARIANTS: dict = None
    def __init__(self, model_name=None, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(model_name or next(iter(self.VARIANTS.values())), batch_size=batch_size, text_field=text_field, verbose=verbose, device=device)

    def __repr__(self):
        inv_variants = {v: k for k, v in self.VARIANTS.items()}
        if self.model_name in inv_variants:
            return f'{self.__class__.__name__}.{inv_variants[self.model_name]}()'
        return super().__repr__()


class Ance(_SBertBiEncoder):
    VARIANTS = {
        'firstp': 'sentence-transformers/msmarco-roberta-base-ance-firstp',
    }


class GTR(_SBertBiEncoder):
    VARIANTS = {
        'base': 'sentence-transformers/gtr-t5-base',
        'large': 'sentence-transformers/gtr-t5-large',
        'xl': 'sentence-transformers/gtr-t5-xl',
        'xxl': 'sentence-transformers/gtr-t5-xxl',
    }

class Query2Query(pt.Transformer):
    DEFAULT_MODEL_NAME = 'neeva/query2query'
    def __init__(self, model_name=DEFAULT_MODEL_NAME, batch_size=32, verbose=False, device=None):
        self.model_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name).to(self.device).eval()
        self.batch_size = batch_size
        self.verbose = verbose

    encode = _sbert_encode
    transform = BiQueryEncoder.transform
    __repr__ = _SBertBiEncoder.__repr__
