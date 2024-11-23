from more_itertools import chunked
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from . import BiEncoder


class TctColBert(BiEncoder):
    def __init__(self, model_name='castorini/tct_colbert-msmarco', batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode_queries(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=36)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state
                res = res[:, 4:, :].mean(dim=1) # remove the first 4 tokens (representing [CLS] [ Q ]), and average
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer([f'[CLS] [D] {d}' for d in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state
                res = res[:, 4:, :] # remove the first 4 tokens (representing [CLS] [ D ])
                res = res * inps['attention_mask'][:, 4:].unsqueeze(2) # apply attention mask
                lens = inps['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1)
                lens[lens == 0] = 1 # avoid edge case of div0 errors
                res = res.sum(dim=1) / lens # average based on dim
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    def __repr__(self):
        return f'TctColBert({repr(self.model_name)})'
