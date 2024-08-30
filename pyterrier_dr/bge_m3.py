from pyterrier.transformer import Transformer
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import pyterrier as pt
import pandas as pd
import numpy as np
import torch
from .biencoder import BiEncoder


class BGEM3Factory(BiEncoder):
    def __init__(self, model_name='BAAI/bge-m3', batch_size=32, max_length=8192, text_field='text', verbose=False, device=None, use_fp16=False):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16, device=self.device)


    def __repr__(self):
        return f'BGEM3Factory({repr(self.model_name)})'
    
    def encoder(self, verbose=None, batch_size=None, max_length=None) -> pt.Transformer:
        '''
        Encoding using Single-Vector Dense Embeddings
        '''
        return BGEM3Encoder(self, verbose=verbose, batch_size=batch_size, max_length=max_length)

class BGEM3Encoder(pt.Transformer):
    def __init__(self, bge_factory: BGEM3Factory, verbose=None, batch_size=None, max_length=None):
        self.bge_factory = bge_factory
        self.verbose = verbose if verbose is not None else bge_factory.verbose
        self.batch_size = batch_size if batch_size is not None else bge_factory.batch_size
        self.max_length = max_length if max_length is not None else bge_factory.max_length

    def encode(self, texts):
        return self.bge_factory.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length,
                             return_dense=True, return_sparse=True, return_colbert_vecs=True)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        if 'query' in inp.columns:
            # assert all(c in inp.columns for c in ['query'])
            it = inp['query'].values
            it, inv = np.unique(it, return_inverse=True)
            if self.verbose:
                it = pt.tqdm(it, desc='Encoding Single Vector Embeddings', unit='query')
            bgem3_results = self.encode(it)
            query_vec = [bgem3_results['dense_vecs'][i] for i in inv]
            query_sparse = [bgem3_results['lexical_weights'][i] for i in inv]
            query_multivecs = [bgem3_results['colbert_vecs'][i] for i in inv]
            # add to the input dataframe
            inp = inp.assign(query_vec=query_vec,
                             query_sparse=query_sparse,
                             query_multivecs=query_multivecs
                             )
        # check if the input dataframe contains the field(s) specified in the text_field
        if all(c in inp.columns for c in [self.bge_factory.text_field]):
            it = inp[self.bge_factory.text_field]
            if self.verbose:
                it = pt.tqdm(it, desc='Encoding Single Vector Embeddings', unit='doc')
            bgem3_results = self.encode(it)
            doc_vec = bgem3_results['dense_vecs']
            doc_sparse = bgem3_results['lexical_weights']
            doc_multivecs = bgem3_results['colbert_vecs']
            # add to the input dataframe
            inp = inp.assign(doc_vec=list(doc_vec), 
                             doc_sparse=list(doc_sparse), 
                             doc_multivecs=list(doc_multivecs)
                             )
        return inp

    def __repr__(self):
        return f'{repr(self.bge_factory)}.encoder()'
