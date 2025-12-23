

from .biencoder import BiEncoder
import numpy as np
from more_itertools import chunked
import torch
class LionLlamaDense(BiEncoder):

    def __init__(self, model_name="hzeng/Lion-DS-1B-llama3-marco-mntp", batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size=batch_size, verbose=verbose, text_field=text_field)
        from transformers import AutoTokenizer 
        from ._lion import  LlamaBiDense
        self.model = LlamaBiDense.load_from_lora(model_name)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        print("model to", self.device)
        self.model = self.model.to(self.device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode_queries_torch(self, texts, batch_size=None):
        results = []
        for chunk in chunked(texts, batch_size or self.batch_size):
            inps = self.tokenizer(list(chunk),  max_length=192, return_tensors='pt', padding="longest", truncation=True)
            inps = inps.to(self.device)
            res = self.model.query_encode(**inps)
            results.append(res)
        if not results:
            return torch.zeros(shape=(0, 0))
        return torch.cat(results, dim=0).float()
    
    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), max_length=192, return_tensors='pt', padding='longest', truncation=True)
                inps = inps.to(self.device)
                res = self.model.doc_encode(**inps)
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)
    
    
