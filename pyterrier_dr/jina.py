import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from .biencoder import BiEncoder
from tqdm import tqdm
from packaging.version import Version

class JinaEmbedder(BiEncoder):
    def __init__(self, model_name='jinaai/jina-embeddings-v4', batch_size=32, text_field='text', verbose=False, device=None, max_length=1024):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
        self.model_name = model_name
        
        if Version(transformers.__version__) >= Version("5.0.0"):
            raise RuntimeError(
                f"transformers {transformers.__version__} is not compatible with current implementation of {self.model_name}. Please downgrade: 'pip install 'transformers<5.0.0'")
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.max_length = max_length

        self.model = SentenceTransformer(model_name, trust_remote_code=True, tokenizer_kwargs={"model_max_length": self.max_length}).to(self.device).eval()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    def encode_queries(self, texts, batch_size=None, prompt="query",):
        show_progress = False
        if isinstance(texts, tqdm):
            texts.disable = True
            show_progress = True
        texts = list(texts)

        if len(texts) == 0:
            return np.empty(shape=(0, 0))
        
        # current jina implimentation will attempt to load urls as PIL Images
        # go through every text and if it starts with http add another space at the beginning to avoid jina's weird url tokenization
        texts = [f" {text}" if text.startswith("http") else text for text in texts]
        
        return self.model.encode(sentences=texts, 
                                batch_size=batch_size or self.batch_size, 
                                show_progress_bar=show_progress,
                                task="retrieval",
                                prompt_name=prompt,
                                )

    def encode_docs(self, texts, batch_size=None, prompt="passage"):
        show_progress = False
        if isinstance(texts, tqdm):
            texts.disable = True
            show_progress = True
        texts = list(texts)

        if len(texts) == 0:
            return np.empty(shape=(0, 0))
        
        # current jina implimentation will attempt to load urls as PIL Images
        # go through every text and if it starts with http add another space at the beginning to avoid jina's weird url tokenization
        texts = [f" {text}" if text.startswith("http") else text for text in texts]

        return self.model.encode(sentences=texts, 
                                batch_size=batch_size or self.batch_size, 
                                show_progress_bar=show_progress,
                                task="retrieval",
                                prompt_name=prompt,
                                )
    def __repr__(self):
        return f'JinaEmbedder({repr(self.model_name)})'