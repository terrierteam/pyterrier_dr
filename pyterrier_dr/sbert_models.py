import numpy as np
import torch
import pyterrier as pt
from transformers import AutoConfig
from functools import partialmethod
from .biencoder import BiEncoder, BiQueryEncoder
from .util import Variants
from tqdm import tqdm

def _sbert_encode(self, texts, batch_size=None, prompt=None, normalize_embeddings=False):
    show_progress = False
    if isinstance(texts, tqdm):
        texts.disable = True
        show_progress = True
    texts = list(texts)
    if prompt is not None:
        texts = [prompt + t for t in texts]
    if len(texts) == 0:
        return np.empty(shape=(0, 0))
    return self.model.encode(texts, 
                             batch_size=batch_size or self.batch_size, 
                             show_progress_bar=show_progress,
                             normalize_embeddings=normalize_embeddings
                             )


class SBertBiEncoder(BiEncoder):
    def __init__(self, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
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
    """Dense encoder for ANCE (Approximate nearest neighbor Negative Contrastive Learning).

    See :class:`~pyterrier_dr.BiEncoder` for usage information.

    .. cite.dblp:: conf/iclr/XiongXLTLBAO21

    .. automethod:: firstp()
    """
    VARIANTS = {
        'firstp': 'sentence-transformers/msmarco-roberta-base-ance-firstp',
    }

class E5(_SBertBiEncoder):
    """Dense encoder for E5 (EmbEddings from bidirEctional Encoder rEpresentations).

    See :class:`~pyterrier_dr.BiEncoder` for usage information.

    .. cite.dblp:: journals/corr/abs-2212-03533

    .. automethod:: base()
    .. automethod:: small()
    .. automethod:: large()
    """

    encode_queries = partialmethod(_sbert_encode, prompt='query: ', normalize_embeddings=True)
    encode_docs = partialmethod(_sbert_encode, prompt='passage: ', normalize_embeddings=True)

    VARIANTS = {
        'base' : 'intfloat/e5-base-v2',
        'small': 'intfloat/e5-small-v2', 
        'large': 'intfloat/e5-large-v2',
    }

class GTR(_SBertBiEncoder):
    """Dense encoder for GTR (Generalizable T5-based dense Retrievers)

    See :class:`~pyterrier_dr.BiEncoder` for usage information.

    .. cite.dblp:: conf/emnlp/Ni0LDAMZLHCY22

    .. automethod:: base()
    .. automethod:: large()
    .. automethod:: xl()
    .. automethod:: xxl()
    """
    VARIANTS = {
        'base': 'sentence-transformers/gtr-t5-base',
        'large': 'sentence-transformers/gtr-t5-large',
        'xl': 'sentence-transformers/gtr-t5-xl',
        'xxl': 'sentence-transformers/gtr-t5-xxl',
    }

class Query2Query(pt.Transformer, metaclass=Variants):
    """Dense query encoder model for query similarity.

    Note that this encoder only provides a :meth:`~pyterrier_dr.BiEncoder.query_encoder` (no document encoder or scorer).

    .. cite:: query2query
        :citation: Bathwal and Samdani. State-of-the-art Query2Query Similarity. 2022.
        :link: https://web.archive.org/web/20220923212754/https://neeva.com/blog/state-of-the-art-query2query-similarity

    .. automethod:: base()
    """
    def __init__(self, model_name=None, batch_size=32, verbose=False, device=None):
        self.model_name = model_name or next(iter(self.VARIANTS.values()))
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name).to(self.device).eval()
        self.batch_size = batch_size
        self.verbose = verbose

    VARIANTS = {
        'base': 'neeva/query2query',
    }

    encode = _sbert_encode
    transform = BiQueryEncoder.transform
    __repr__ = _SBertBiEncoder.__repr__
