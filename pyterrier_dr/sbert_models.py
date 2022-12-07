from more_itertools import chunked
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from .biencoder import BiEncoder


class SbertModel(BiEncoder):
    def __init__(self, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = SentenceTransformer(model_name).to(self.device).eval()

    def encode_queries(self, texts, batch_size=None):
        return self.model.encode(texts, batch_size=batch_size or self.batch_size)

    def encode_docs(self, texts, batch_size=None):
        return self.model.encode(texts, batch_size=batch_size or self.batch_size)

    def __repr__(self):
        return f'SbertModel({repr(self.model_name)})'


class Ance(SbertModel):
    DEFAULT_MODEL_NAME = 'sentence-transformers/msmarco-roberta-base-ance-firstp'
    def __init__(self, model_name=DEFAULT_MODEL_NAME, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(model_name, batch_size=32, text_field='text', verbose=False, device=None)

    def __repr__(self):
        if self.model_name == self.DEFAULT_MODEL_NAME:
            return 'ANCE()'
        return super().__repr__()
