from typing import Optional, Union, List
import torch
from . import BiEncoder
from transformers import AutoTokenizer
from more_itertools import chunked
import numpy as np

# see https://huggingface.co/facebook/dragon-plus-query-encoder

class Dragon(BiEncoder):

    def __init__(self, *args, 
                 query_encoder = 'facebook/dragon-plus-query-encoder', 
                 doc_encoder = 'facebook/dragon-plus-context-encoder', 
                 batch_size=32,
                 device: Optional[Union[str, torch.device]] = None,
                 **kwargs):
        self.query_encoder_name = query_encoder
        self.query_encoder = None
        self.doc_encoder_name = doc_encoder
        self.doc_encoder = None
        self.tokenizer = AutoTokenizer.from_pretrained(query_encoder)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)

    def _query_model(self):
        if self.query_encoder is not None:
            return self.query_encoder
        from transformers import AutoModel
        self.query_encoder = AutoModel.from_pretrained(self.query_encoder_name)
        if self.device is not None:
            self.query_encoder = self.query_encoder.to(self.device)
        return self.query_encoder
    
    def _doc_model(self):
        if self.doc_encoder is not None:
            return self.doc_encoder
        from transformers import AutoModel
        self.doc_encoder = AutoModel.from_pretrained(self.doc_encoder_name)
        if self.device is not None:
            self.doc_encoder = self.doc_encoder.to(self.device)
        return self.doc_encoder

    def encode_queries_torch(self, texts: List[str], batch_size: Optional[int] = None) -> torch.Tensor:
        results = []
        query_encoder = self._query_model()
        for chunk in chunked(texts, batch_size or self.batch_size):
            inps = self.tokenizer(list(chunk),  max_length=192, return_tensors='pt', padding="longest", truncation=True)
            inps = inps.to(self.device)
            res = query_encoder(**inps).last_hidden_state[:, 0, :]
            results.append(res)
        if not results:
            return torch.zeros(shape=(0, 0))
        return torch.stack(results, axis=0)
    
    def encode_docs(self, texts: List[str], batch_size: Optional[int] = None) -> np.array:
        results = []
        doc_encoder = self._doc_model()
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk),  max_length=512, return_tensors='pt', padding="longest", truncation=True)
                inps = inps.to(self.device)
                res = doc_encoder(**inps).last_hidden_state[:, 0, :]
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)