from more_itertools import chunked
import numpy as np
import torch
from torch import nn
import pyterrier as pt
from transformers import RobertaConfig, AutoTokenizer, AutoModel, AdamW
from sentence_transformers import SentenceTransformer


class Query2Query(pt.Transformer):
    def __init__(self, model_name='neeva/query2query', batch_size=32, verbose=False, device=None):
        self.model_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = SentenceTransformer('neeva/query2query').to(self.device).eval()
        self.batch_size = batch_size
        self.verbose = verbose

    def transform(self, inp):
        assert all(c in inp.columns for c in ['query'])
        results = []
        with torch.no_grad():
            it = iter(inp['query'])
            if self.verbose:
                it = logger.pbar(it, desc='query2query', unit='q', total=len(inp))
            for chunk in chunked(it, self.batch_size):
                chunk = list(chunk)
                res = self.model.encode(chunk)
                results += list(res)
        return inp.assign(query_vec=results)
