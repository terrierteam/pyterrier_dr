from more_itertools import chunked
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .biencoder import BiEncoder


class HgfModel(BiEncoder):
    def __init__(self, model, tokenizer, config, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size, text_field, verbose)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    def encode_queries(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state[:, 0] # [CLS] embedding
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state[:, 0]
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

    @classmethod
    def from_pretrained(cls, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        res = cls(model, tokenizer, config, batch_size=batch_size, text_field=text_field, verbose=verbose, device=device)
        res.model_name = model_name
        return res

    def __repr__(self):
        if hasattr(self, 'model_name'):
            return f'HgfModel({repr(self.model_name)})'
        return 'HgfModel()'


class TasB(HgfModel):
    DEFAULT_MODEL_NAME = 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco'
    def __init__(self, model_name=DEFAULT_MODEL_NAME, batch_size=32, text_field='text', verbose=False, device=None):
        self.model_name = model_name
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(model, tokenizer, config, batch_size=32, text_field='text', verbose=False, device=None)

    def __repr__(self):
        if self.model_name != self.DEFAULT_MODEL_NAME:
            return f'TAS_B({repr(self.model_name)})'
        return 'TasB()'
