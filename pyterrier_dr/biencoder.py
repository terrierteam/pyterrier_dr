from typing import List, Optional
from abc import abstractmethod
import numpy as np
import pyterrier as pt
import pandas as pd
import pyterrier_alpha as pta
from . import SimFn


class BiEncoder(pt.Transformer):
    """Represents a single-vector dense bi-encoder.

    A ``BiEncoder`` encodes the text of a query or document into a dense vector.

    This class functions as a transformer factory:
     - Query encoding using :meth:`query_encoder`
     - Document encoding using :meth:`doc_encoder`
     - Text scoring (re-reranking) using :meth:`text_scorer`

    It can also be used as a transformer directly. It infers which transformer to use
    based on columns present in the input frame.

    Note that in most cases, you will want to use a ``BiEncoder`` as part of a pipeline
    with a :class:`~pyterrier_dr.FlexIndex` to perform dense indexing and retrival.
    """
    def __init__(self, *, batch_size=32, text_field='text', verbose=False):
        """
        Args:
            batch_size: The default batch size to use for query/document encoding
            text_field: The field in the input dataframe that contains the document text
            verbose: Whether to show progress bars
        """
        super().__init__()
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        with pta.validate.any(inp) as v:
            v.columns(includes=['query', self.text_field], mode='scorer')
            v.columns(includes=['query_vec', self.text_field], mode='scorer')
            v.columns(includes=['query', 'doc_vec'], mode='scorer')
            v.columns(includes=['query_vec', 'doc_vec'], mode='scorer')
            v.columns(includes=['query'], mode='query_encoder')
            v.columns(includes=[self.text_field], mode='doc_encoder')

        if v.mode == 'scorer':
            return self.scorer()(inp)
        elif v.mode == 'query_encoder':
            return self.query_encoder()(inp)
        elif v.mode == 'doc_encoder':
            return self.doc_encoder()(inp)

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

    def text_scorer(self, verbose=None, batch_size=None, sim_fn=None) -> pt.Transformer:
        """
        Text Scoring (re-ranking)
        """
        return BiScorer(self, verbose=verbose, batch_size=batch_size, sim_fn=sim_fn)

    def scorer(self, verbose=None, batch_size=None, sim_fn=None) -> pt.Transformer:
        return self.text_scorer(verbose=verbose, batch_size=batch_size, sim_fn=sim_fn)

    @property
    def sim_fn(self) -> SimFn:
        """
        The similarity function to use between embeddings for this model
        """
        if hasattr(self, 'config') and hasattr(self.config, 'sim_fn'):
            return SimFn(self.config.sim_fn)
        return SimFn.dot # default

    @abstractmethod
    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None) -> np.array:
        """Abstract method to encode a list of query texts into dense vectors.

        This function is used by the transformer returned by :meth:`query_encoder`.

        Args:
            texts: A list of query texts
            batch_size: The batch size to use for encoding

        Returns:
            np.array: A numpy array of shape (n_queries, n_dims)
        """
        raise NotImplementedError()

    @abstractmethod
    def encode_docs(self, texts: List[str], batch_size: Optional[int] = None) -> np.array:
        """Abstract method to encode a list of document texts into dense vectors.

        This function is used by the transformer returned by :meth:`doc_encoder`.

        Args:
            texts: A list of document texts
            batch_size: The batch size to use for encoding

        Returns:
            np.array: A numpy array of shape (n_docs, n_dims)
        """
        raise NotImplementedError()


class BiQueryEncoder(pt.Transformer):
    def __init__(self, bi_encoder_model: BiEncoder, verbose=None, batch_size=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_encoder_model.encode_queries(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=['query'])
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
        pta.validate.columns(inp, includes=[self.text_field])
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
        with pta.validate.any(inp) as v:
            v.columns(includes=['query_vec', 'doc_vec'])
            v.columns(includes=['query', 'doc_vec'])
            v.columns(includes=['query_vec', self.text_field])
            v.columns(includes=['query', self.text_field])
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
