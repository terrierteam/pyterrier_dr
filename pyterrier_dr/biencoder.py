from more_itertools import chunked
import numpy as np
import torch
from torch import nn
import pyterrier as pt


class BiEncoder(pt.Transformer):
    def __init__(self, batch_size=32, text_field='text', verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose

    def encode_queries(self, texts, batch_size=None):
        raise NotImplementedError()

    def encode_docs(self, texts, batch_size=None):
        raise NotImplementedError()

    def transform(self, inp):
        columns = set(inp.columns)
        modes = [
            (['qid', 'query', self.text_field], self.scorer),
            (['query'], self.query_encoder),
            ([self.text_field], self.doc_encoder),
        ]
        for fields, fn in modes:
            if all(f in columns for f in fields):
                return fn()(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in modes:
            f += f'\n - {fn.__doc__.strip()}: {columns}\n'
        raise RuntimeError(message)

    def query_encoder(self, verbose=None, batch_size=None):
        """
        Query encoding
        """
        return BiQueryEncoder(self, verbose=verbose, batch_size=batch_size)

    def doc_encoder(self, verbose=None, batch_size=None):
        """
        Doc encoding
        """
        return BiDocEncoder(self, verbose=verbose, batch_size=batch_size)

    def scorer(self, verbose=None, batch_size=None):
        """
        Scoring (re-ranking)
        """
        return BiScorer(self, verbose=verbose, batch_size=batch_size)


class BiQueryEncoder(pt.transformer.TransformerBase):
    def __init__(self, bi_encoder_model, verbose=None, batch_size=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size

    def encode(self, texts):
        return self.bi_encoder_model.encode_queries(texts, batch_size=self.batch_size)

    def transform(self, inp):
        assert all(c in inp.columns for c in ['query'])
        it = inp['query']
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        return inp.assign(query_vec=list(self.encode(it)))

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.query_encoder()'


class BiDocEncoder(pt.transformer.TransformerBase):
    def __init__(self, bi_encoder_model, verbose=None, batch_size=None, text_field=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size
        self.text_field = text_field if text_field is not None else bi_encoder_model.text_field

    def encode(self, texts):
        return self.bi_encoder_model.encode_docs(texts, batch_size=self.batch_size)

    def transform(self, inp):
        assert all(c in inp.columns for c in [self.text_field])
        it = inp[self.text_field]
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Docs', unit='doc')
        return inp.assign(doc_vec=list(self.encode(it))])

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.doc_encoder()'


class BiScorer(pt.transformer.TransformerBase):
    def __init__(self, bi_encoder_model, verbose=None, batch_size=None, text_field=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size
        self.text_field = text_field if text_field is not None else bi_encoder_model.text_field

    def transform(self, inp):
        assert all(c in inp.columns for c in ['qid', 'query', self.text_field])
        return pt.apply.by_query(self._transform_byquery, add_ranks=True, verbose=self.verbose)(inp)

    def _transform_byquery(self, query_df):
        query_rep = self.bi_encoder_model.encode_queries([query_df['query'].iloc[0]], batch_size=self.batch_size)
        doc_reps = self.bi_encoder_model.encode_docs(query_df[self.text_field], batch_size=self.batch_size)
        scores = (query_rep * doc_reps).sum(axis=1)
        query_df['score'] = scores
        return query_df

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.scorer()'
