from typing import List, Optional, Union
import json
import os
import random
import numpy as np
import torch
import pyterrier as pt
from transformers import AutoConfig
from .biencoder import BiEncoder
from tqdm import tqdm
import pyterrier_alpha as pta


class CDE(BiEncoder):
    def __init__(self, model_name='jxm/cde-small-v1', cache: Optional['CDECache'] = None, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, trust_remote_code=True).to(self.device).eval()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.cache = cache or CDECache(cde=self)

    def encode_context(self, texts: List[str], batch_size=None):
        show_progress = False
        if isinstance(texts, tqdm):
            texts.disable = True
            show_progress = True
        texts = list(texts)
        if len(texts) == 0:
            return np.empty(shape=(0, 0))
        return self.model.encode(
            texts,
            prompt_name="document",
            batch_size=batch_size or self.batch_size,
            show_progress=show_progress,
        )

    def encode_queries(self, texts: List[str], batch_size=None):
        show_progress = False
        if isinstance(texts, tqdm):
            texts.disable = True
            show_progress = True
        texts = list(texts)
        if len(texts) == 0:
            return np.empty(shape=(0, 0))
        result = self.model.encode(
            texts,
            prompt_name='query',
            dataset_embeddings=self.cache.context(),
            show_progress=show_progress,
        )
        # sentence transformers doesn't norm?
        result = result / np.linalg.norm(result, ord=2, axis=1, keepdims=True)
        return result

    def encode_docs(self, texts: List[str], batch_size=None):
        show_progress = False
        if isinstance(texts, tqdm):
            texts.disable = True
            show_progress = True
        texts = list(texts)
        if len(texts) == 0:
            return np.empty(shape=(0, 0))
        result = self.model.encode(
            texts,
            prompt_name='document',
            dataset_embeddings=self.cache.context(),
            show_progress=show_progress,
        )
        # sentence transformers doesn't norm?
        result = result / np.linalg.norm(result, ord=2, axis=1, keepdims=True)
        return result

    def __repr__(self):
        return f'CDE({self.model_name!r})'


def _sample_iter(it, k, rng=None):
    if rng is None:
        rng = random.Random(42)
    elif not isinstance(rng, random.Random):
        rng = random.Random(rng)
    result = []
    for i, item in enumerate(it):
        if i < k:
            result.append(item)
        elif (e := rng.randint(0, i)) < k:
            result[e] = item
    return result


class CDECache(pta.Artifact, pt.Indexer):
    ARTIFACT_TYPE = 'cde_cache'
    ARTIFACT_FORMAT = 'np_pickle'

    def __init__(self, path: Optional[str] = None, cde: Optional[Union[CDE, str]] = None, rng = None):
        super().__init__(path or '__x_ignore__')
        self._context = None
        if cde is None:
            if self.built():
                with open(self.path/'pt_meta.json') as fin:
                    model_name = json.load(fin)['cde_model']
                cde = CDE(model_name, cache=self)
            else:
                cde = CDE(cache=self)
        elif isinstance(cde, str):
            cde = CDE(cde, cache=self)
        self.cde = cde
        self.rng = rng

    def index(self, inp):
        assert not self.built()
        context_docs_size = self.cde.config.transductive_corpus_size
        sample = _sample_iter(inp, k=context_docs_size, rng=self.rng)
        self._context = self.cde.encode_context([d[self.cde.text_field] for d in sample])
        if str(self.path) != '__x_ignore__':
            with pta.ArtifactBuilder(self) as builder:
                builder.metadata['cde_model'] = self.cde.model_name
                builder.metadata['shape'] = list(self._context.shape)
                self._context.tofile(builder.path / 'sample.np')

    def __repr__(self):
        if str(self.path) == '__x_ignore__':
            return 'CDECache()'
        return f'CDECache({self.path!r})'

    def built(self):
        if str(self.path) == '__x_ignore__':
            return self._context is not None
        return os.path.exists(self.path)

    def context(self):
        if self._context is None:
            if not self.built():
                raise RuntimeError('{self!r} is not built. Run cde.cache.index() over your corpus first to build the context cache.')
            ctxt = np.fromfile(self.path/'sample.np', dtype=np.float32)
            with open(self.path/'pt_meta.json') as fin:
                shape = json.load(fin)['shape']
            self._context = ctxt.reshape(shape)
        return self._context
