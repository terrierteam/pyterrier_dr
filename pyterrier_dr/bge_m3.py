import pyterrier as pt
import pandas as pd
import numpy as np
import torch
import pyterrier_alpha as pta
from .biencoder import BiEncoder

class BGEM3(BiEncoder):
    def __init__(self, model_name='BAAI/bge-m3', batch_size=32, max_length=8192, text_field='text', verbose=False, device=None, use_fp16=False):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError("BGE-M3 requires the FlagEmbedding package. You can install it using 'pip install pyterrier-dr[bgem3]'")
        
        self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16, device=self.device)


    def __repr__(self):
        return f'BGEM3({repr(self.model_name)})'
    
    def encode_queries(self, texts, batch_size=None):
        return self.model.encode(list(texts), batch_size=batch_size, max_length=self.max_length,
                                  return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']

    def encode_docs(self, texts, batch_size=None):
        return self.model.encode(list(texts), batch_size=batch_size, max_length=self.max_length,
                                    return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']

    # Only does dense (single_vec) encoding
    def query_encoder(self, verbose=None, batch_size=None):
        return BGEM3QueryEncoder(self, verbose=verbose, batch_size=batch_size)
    def doc_encoder(self, verbose=None, batch_size=None):
        return BGEM3DocEncoder(self, verbose=verbose, batch_size=batch_size)
    
    # Does all three BGE-M3 encodings: dense, sparse and colbert(multivec)
    def query_multi_encoder(self, verbose=None, batch_size=None, return_dense=True, return_sparse=True, return_colbert_vecs=True):
        return BGEM3QueryEncoder(self, verbose=verbose, batch_size=batch_size, return_dense=return_dense, return_sparse=return_sparse, return_colbert_vecs=return_colbert_vecs)
    def doc_multi_encoder(self, verbose=None, batch_size=None, return_dense=True, return_sparse=True, return_colbert_vecs=True):
        return BGEM3DocEncoder(self, verbose=verbose, batch_size=batch_size, return_dense=return_dense, return_sparse=return_sparse, return_colbert_vecs=return_colbert_vecs)

class BGEM3QueryEncoder(pt.Transformer):
    def __init__(self, bge_factory: BGEM3, verbose=None, batch_size=None, max_length=None, return_dense=True, return_sparse=False, return_colbert_vecs=False):
        self.bge_factory = bge_factory
        self.verbose = verbose if verbose is not None else bge_factory.verbose
        self.batch_size = batch_size if batch_size is not None else bge_factory.batch_size
        self.max_length = max_length if max_length is not None else bge_factory.max_length

        self.dense = return_dense
        self.sparse = return_sparse
        self.multivecs = return_colbert_vecs
    
    def encode(self, texts):
        return self.bge_factory.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length,
                             return_dense=self.dense, return_sparse=self.sparse, return_colbert_vecs=self.multivecs)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.columns(inp, includes=['query'])

        # check if inp is empty
        if len(inp) == 0:
            if self.dense:
                inp = inp.assign(query_vec=[])
            if self.sparse:
                inp = inp.assign(query_toks=[])
            if self.multivecs:
                inp = inp.assign(query_embs=[])
            return inp

        it = inp['query'].values
        it, inv = np.unique(it, return_inverse=True)
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        bgem3_results = self.encode(it)

        if self.dense:
            inp = inp.assign(query_vec=[bgem3_results['dense_vecs'][i] for i in inv])
        if self.sparse:
            # for sparse convert ids to the actual tokens
            query_toks = self.bge_factory.model.convert_id_to_token(bgem3_results['lexical_weights'])
            inp = inp.assign(query_toks=query_toks)
        if self.multivecs:
            inp = inp.assign(query_embs=[bgem3_results['colbert_vecs'][i] for i in inv])
        return inp
    
    def __repr__(self):
        return f'{repr(self.bge_factory)}.query_encoder()'

class BGEM3DocEncoder(pt.Transformer):
    def __init__(self, bge_factory: BGEM3, verbose=None, batch_size=None, max_length=None, return_dense=True, return_sparse=False, return_colbert_vecs=False):
        self.bge_factory = bge_factory
        self.verbose = verbose if verbose is not None else bge_factory.verbose
        self.batch_size = batch_size if batch_size is not None else bge_factory.batch_size
        self.max_length = max_length if max_length is not None else bge_factory.max_length

        self.dense = return_dense
        self.sparse = return_sparse
        self.multivecs = return_colbert_vecs

    def encode(self, texts):
        return self.bge_factory.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length,
                             return_dense=self.dense, return_sparse=self.sparse, return_colbert_vecs=self.multivecs)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        # check if the input dataframe contains the field(s) specified in the text_field
        pta.validate.columns(inp, includes=[self.bge_factory.text_field])
        # check if inp is empty
        if len(inp) == 0:
            if self.dense:
                inp = inp.assign(doc_vec=[])
            if self.sparse:
                inp = inp.assign(toks=[])
            if self.multivecs:
                inp = inp.assign(doc_embs=[])
            return inp

        it = inp[self.bge_factory.text_field]
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Documents', unit='doc')
        bgem3_results = self.encode(it)

        if self.dense:
            inp = inp.assign(doc_vec=list(bgem3_results['dense_vecs']))
        if self.sparse:
            toks = bgem3_results['lexical_weights']
            # for sparse convert ids to the actual tokens
            toks = self.bge_factory.model.convert_id_to_token(toks)
            inp = inp.assign(toks=toks)
        if self.multivecs:
            inp = inp.assign(doc_embs=list(bgem3_results['colbert_vecs']))
        return inp

    def __repr__(self):
        return f'{repr(self.bge_factory)}.doc_encoder()'
