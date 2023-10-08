from more_itertools import chunked
import numpy as np
import torch
from torch import nn
import pyterrier as pt
import pandas as pd
from . import SimFn


class BiEncoder(pt.Transformer):
    def __init__(self, batch_size=32, text_field='text', verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose

    def encode_queries(self, texts, batch_size=None) -> np.array:
        raise NotImplementedError()

    def encode_docs(self, texts, batch_size=None) -> np.array:
        raise NotImplementedError()

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        columns = set(inp.columns)
        modes = [
            (['qid', 'query', self.text_field], self.scorer),
            (['qid', 'query_vec', self.text_field], self.scorer),
            (['qid', 'query', 'doc_vec'], self.scorer),
            (['qid', 'query_vec', 'doc_vec'], self.scorer),
            (['query'], self.query_encoder),
            ([self.text_field], self.doc_encoder),
        ]
        for fields, fn in modes:
            if all(f in columns for f in fields):
                return fn()(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in modes:
            message += f'\n - {fn.__doc__.strip()}: {fields}'
        raise RuntimeError(message)

    def query_encoder(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Query encoding
        """
        return BiQueryEncoder(self, verbose=verbose, batch_size=batch_size)

    def doc_encoder(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Doc encoding
        """
        return BiDocEncoder(self, verbose=verbose, batch_size=batch_size)

    def scorer(self, verbose=None, batch_size=None, sim_fn=None) -> pt.Transformer:
        """
        Scoring (re-ranking)
        """
        return BiScorer(self, verbose=verbose, batch_size=batch_size, sim_fn=sim_fn)

    @property
    def sim_fn(self) -> SimFn:
        """
        The similarity function to use between embeddings for this model
        """
        if hasattr(self, 'config') and hasattr(self.config, 'sim_fn'):
            return SimFn(self.config.sim_fn)
        return SimFn.dot # default


class BiQueryEncoder(pt.Transformer):
    def __init__(self, bi_encoder_model: BiEncoder, verbose=None, batch_size=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_encoder_model.encode_queries(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert all(c in inp.columns for c in ['query'])
        it = inp['query'].values
        it, inv = np.unique(it, return_inverse=True)
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        enc = self.encode(it)
        return inp.assign(query_vec=[enc[i] for i in inv])

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.query_encoder()'


class BiDocEncoder(pt.Transformer):
    def __init__(self, bi_encoder_model: BiEncoder, verbose=None, batch_size=None, text_field=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size
        self.text_field = text_field if text_field is not None else bi_encoder_model.text_field

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_encoder_model.encode_docs(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert all(c in inp.columns for c in [self.text_field])
        it = inp[self.text_field]
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Docs', unit='doc')
        return inp.assign(doc_vec=list(self.encode(it)))

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.doc_encoder()'


class BiScorer(pt.Transformer):
    def __init__(self, bi_encoder_model: BiEncoder, verbose=None, batch_size=None, text_field=None, sim_fn=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size
        self.text_field = text_field if text_field is not None else bi_encoder_model.text_field
        self.sim_fn = sim_fn if sim_fn is not None else bi_encoder_model.sim_fn

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert 'query_vec' in inp.columns or 'query' in inp.columns
        assert 'doc_vec' in inp.columns or self.text_field in inp.columns
        if 'query_vec' in inp.columns:
            query_vec = inp['query_vec']
        else:
            query_vec = self.bi_encoder_model.query_encoder(batch_size=self.batch_size, verbose=self.verbose)(inp)['query_vec']
        if 'doc_vec' in inp.columns:
            doc_vec = inp['doc_vec']
        else:
            doc_vec = self.bi_encoder_model.doc_encoder(batch_size=self.batch_size, verbose=self.verbose)(inp)['doc_vec']
        if self.sim_fn == SimFn.dot:
            scores = (query_vec * doc_vec).apply(np.sum)
        else:
            raise ValueError(f'{self.sim_fn} not yet supported by BiScorer')
        outp = inp.assign(score=scores)
        return pt.model.add_ranks(outp)

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.scorer()'
