from more_itertools import chunked
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
from . import BiEncoder

"""
RepLLAMA model from:

@article{rankllama,
      title={Fine-Tuning LLaMA for Multi-Stage Text Retrieval}, 
      author={Xueguang Ma and Liang Wang and Nan Yang and Furu Wei and Jimmy Lin},
      year={2023},
      journal={arXiv:2310.08319},
}
"""

class RepLlama(BiEncoder):
    def __init__(self, model_name='castorini/repllama-v1-7b-lora-passage', tokenizer_name='meta-llama/Llama-2-7b-hf', batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
        # model loading code adapted from <https://huggingface.co/castorini/repllama-v1-7b-lora-passage>
        self.model_name = model_name
        config = PeftConfig.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_name)
        model = model.merge_and_unload()
        model.eval()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode_queries(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer([f'query: {q}</s>' for q in texts], return_tensors='pt', padding=True, truncation=True)
                inps = inps.to(self.device)
                res = self.model(**inps).last_hidden_state[:, -1]
                res = torch.nn.functional.normalize(res, p=2, dim=1)
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer([f'passage: {d}</s>' for d in texts], return_tensors='pt', padding=True, truncation=True)
                inps = inps.to(self.device)
                res = self.model(**inps).last_hidden_state[:, -1]
                res = torch.nn.functional.normalize(res, p=2, dim=1)
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

    def __repr__(self):
        return f'RepLlama({repr(self.model_name)})'
