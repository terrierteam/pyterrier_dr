from pyterrier.transformer import Transformer
from .biencoder import BiEncoder
from tqdm import tqdm
import pyterrier as pt
import pandas as pd
import numpy as np
import torch

class BGEM3Factory(BiEncoder):
    def __init__(self, model_name='BAAI/bge-m3', batch_size=32, max_length=8192, text_field='text', verbose=False, device=None, use_fp16=False):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16, device=self.device)


    def __repr__(self):
        return f'BGEM3Factory({repr(self.model_name)})'
    
    def dense_encoder(self, verbose=None, batch_size=None) -> pt.Transformer:
        '''
        Encoding using Single-Vector Dense Embeddings
        '''
        return BGEM3DenseEncoder(self, verbose=verbose, batch_size=batch_size)
    
    def sparse_encoder(self, verbose=None, batch_size=None, token_mode=None) -> pt.Transformer:
        '''
        Encoding using Sparse Embedding (Lexical Weights)
        '''
        return BGEM3SparseEncoder(self, verbose=verbose, batch_size=batch_size, token_mode=token_mode)
    
    def multivec_encoder(self, verbose=None, batch_size=None) -> pt.Transformer:
        '''
        Encoding using Multi-Vector Embeddings (ColBERT)
        '''
        return BGEM3MultiVecEncoder(self, verbose=verbose, batch_size=batch_size)

class BGEM3DenseEncoder(pt.Transformer):
    def __init__(self, bge_factory: BGEM3Factory, verbose=None, batch_size=None, max_length=None):
        self.bge_factory = bge_factory
        self.verbose = verbose if verbose is not None else bge_factory.verbose
        self.batch_size = batch_size if batch_size is not None else bge_factory.batch_size
        self.max_length = max_length if max_length is not None else bge_factory.max_length

    def encode(self, texts) -> np.array:
        return self.bge_factory.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length,
                             return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        if 'query' in inp.columns:
            # assert all(c in inp.columns for c in ['query'])
            it = inp['query'].values
            it, inv = np.unique(it, return_inverse=True)
            if self.verbose:
                it = pt.tqdm(it, desc='Encoding Single Vector Embeddings', unit='query')
            return inp.assign(query_vec=[self.encode(it)[i] for i in inv])
        else:
            assert all(c in inp.columns for c in [self.bge_factory.text_field])
            it = inp[self.bge_factory.text_field]
            if self.verbose:
                it = pt.tqdm(it, desc='Encoding Single Vector Embeddings', unit='doc')
            return inp.assign(doc_vec=list(self.encode(it)))

    def __repr__(self):
        return f'{repr(self.bge_factory)}.dense_encoder()'

class BGEM3SparseEncoder(pt.Transformer):
    def __init__(self, bge_factory: BGEM3Factory, verbose=None, batch_size=None, max_length=None, token_mode=None):
        self.bge_factory = bge_factory
        self.verbose = verbose if verbose is not None else bge_factory.verbose
        self.batch_size = batch_size if batch_size is not None else bge_factory.batch_size
        self.max_length = max_length if max_length is not None else bge_factory.max_length
        self.token_mode = token_mode # if set to True will convert token ids to tokens text

    def encode(self, texts) -> np.array:
        lexical_weights = self.bge_factory.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length,
                             return_dense=False, return_sparse=True, return_colbert_vecs=False)['lexical_weights']
        if self.token_mode:
            return self.bge_factory.model.convert_id_to_token(lexical_weights)
        else:
            return lexical_weights

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        if 'query' in inp.columns:
            # assert all(c in inp.columns for c in ['query'])
            it = inp['query'].values
            it, inv = np.unique(it, return_inverse=True)
            if self.verbose:
                it = pt.tqdm(it, desc='Encoding Sparse Embeddings', unit='query')
            return inp.assign(query_vec=[self.encode(it)[i] for i in inv])
        else:
            assert all(c in inp.columns for c in [self.bge_factory.text_field])
            it = inp[self.bge_factory.text_field]
            if self.verbose:
                it = pt.tqdm(it, desc='Encoding Sparse Embeddings', unit='doc')
            return inp.assign(doc_vec=list(self.encode(it)))

    def __repr__(self):
        return f'{repr(self.bge_factory)}.sparse_encoder()'

class BGEM3MultiVecEncoder(pt.Transformer):
    def __init__(self, bge_factory: BGEM3Factory, verbose=None, batch_size=None, max_length=None):
        self.bge_factory = bge_factory
        self.verbose = verbose if verbose is not None else bge_factory.verbose
        self.batch_size = batch_size if batch_size is not None else bge_factory.batch_size
        self.max_length = max_length if max_length is not None else bge_factory.max_length

    def encode(self, texts) -> np.array:
        return self.bge_factory.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length,
                             return_dense=False, return_sparse=False, return_colbert_vecs=True)['colbert_vecs']

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        if 'query' in inp.columns:
            # assert all(c in inp.columns for c in ['query'])
            it = inp['query'].values
            it, inv = np.unique(it, return_inverse=True)
            if self.verbose:
                it = pt.tqdm(it, desc='Encoding Multi-Vector Embeddings', unit='query')
            return inp.assign(query_vec=[self.encode(it)[i] for i in inv])
        else:
            assert all(c in inp.columns for c in [self.bge_factory.text_field])
            it = inp[self.bge_factory.text_field]
            if self.verbose:
                it = pt.tqdm(it, desc='Encoding Multi-Vector Embeddings', unit='doc')
            return inp.assign(doc_vec=list(self.encode(it)))

    def __repr__(self):
        return f'{repr(self.bge_factory)}.multivec_encoder()'