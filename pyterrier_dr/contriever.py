from .tctcolbert_model import BiEncoder
import numpy as np
import torch
from more_itertools import chunked

class ContrieverModel(BiEncoder):
    def __init__(model_name="facebook/contriever-msmarco"):
        from contriever import Contriever
        super().__init__(Contriever.from_pretrained(model_name), tokenizer=model_name)

    def _encode(self, texts, max_length):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                inps = self.tokenizer([q for q in chunk], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                if self.cuda:
                    inps = {k: v.cuda() for k, v in inps.items()}
                res = self.model(**inps)
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

    def _encode_queries(self, texts):
        return self._encode(texts, max_length=36)
    
    def _encode_docs(self, texts):
        return self._encode(texts, max_length=200)
    
    