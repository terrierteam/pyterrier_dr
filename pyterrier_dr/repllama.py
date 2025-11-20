from more_itertools import chunked
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .biencoder import BiEncoder
from .util import Variants

def _get_model(peft_model_name):
    from peft import PeftModel, PeftConfig
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

class _RepLLamaBiEncoderBase(BiEncoder):
    def __init__(self, model: str, tokenizer, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = _get_model(model).to(self.device).eval()
        self.tokenizer = tokenizer

    def encode_queries_torch(self, texts, batch_size=None):
        results = []
        for chunk in chunked(texts, batch_size or self.batch_size):
            inps = self.tokenizer([f'query: {query}</s>' for query in chunk], return_tensors='pt')
            inps = {k: v.to(self.device) for k, v in inps.items()}
            res = self.model(**inps).last_hidden_state[:, -1] # last
            res = torch.nn.functional.normalize(res, p=2, dim=0)
            results.append(res)
        if not results:
            return torch.empty((0, 0))
        return torch.cat(results, dim=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                # TODO what about titles, as per original model
                inps = self.tokenizer([f'passage: {passage}</s>' for passage in chunk], return_tensors='pt')
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state[:, -1]
                res = torch.nn.functional.normalize(res, p=2, dim=0)
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    @classmethod
    def from_pretrained(cls, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        config = AutoConfig.from_pretrained(model_name)
        res = cls(model, tokenizer, config, batch_size=batch_size, text_field=text_field, verbose=verbose, device=device)
        res.model_name = model_name
        return res

    def __repr__(self):
        if hasattr(self, 'model_name'):
            return f'HgfBiEncoder({repr(self.model_name)})'
        return 'HgfBiEncoder()'


class _RepLLamaBiEncoder(_RepLLamaBiEncoderBase, metaclass=Variants):
    VARIANTS: dict = None
    def __init__(self, model_name=None, batch_size=32, text_field='text', verbose=False, device=None):
        self.model_name = model_name or next(iter(self.VARIANTS.values()))
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        super().__init__(model_name, tokenizer, batch_size=batch_size, text_field=text_field, verbose=verbose, device=device)

    def __repr__(self):
        inv_variants = {v: k for k, v in self.VARIANTS.items()}
        if self.model_name in inv_variants:
            return f'{self.__class__.__name__}.{inv_variants[self.model_name]}()'
        return super().__repr__()


class RepLLama(_RepLLamaBiEncoder):
    """
    pip requirements:
     - tiktoken
     - peft
    
    .. automethod:: v1_7b()
    """
    VARIANTS = {
        'v1_7b': 'castorini/repllama-v1-7b-lora-passage',
    }