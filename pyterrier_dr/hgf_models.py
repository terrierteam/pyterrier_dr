from more_itertools import chunked
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .biencoder import BiEncoder
from .util import Variants


class HgfBiEncoder(BiEncoder):
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
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state[:, 0]
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
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
            return f'HgfBiEncoder({repr(self.model_name)})'
        return 'HgfBiEncoder()'


class _HgfBiEncoder(HgfBiEncoder, metaclass=Variants):
    VARIANTS: dict = None
    def __init__(self, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        self.model_name = model_name
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(model, tokenizer, config, batch_size=batch_size, text_field=text_field, verbose=verbose, device=device)

    def __repr__(self):
        inv_variants = {v: k for k, v in self.VARIANTS.items()}
        if self.model_name in inv_variants:
            return f'{self.__class__.__name__}.{inv_variants[self.model_name]}()'
        return super().__repr__()


class TasB(_HgfBiEncoder):
    VARIANTS = {
        'dot': 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco',
    }


class RetroMAE(_HgfBiEncoder):
    VARIANTS = {
        #'wiki_bookscorpus': 'Shitao/RetroMAE', # only pre-trained
        #'msmarco': 'Shitao/RetroMAE_MSMARCO', # only pre-trained
        'msmarco_finetune': 'Shitao/RetroMAE_MSMARCO_finetune',
        'msmarco_distill': 'Shitao/RetroMAE_MSMARCO_distill',
        'wiki_bookscorpus_beir': 'Shitao/RetroMAE_BEIR',
    }
