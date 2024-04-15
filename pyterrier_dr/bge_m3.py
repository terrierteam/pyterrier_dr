import torch
from transformers import AutoConfig
from .biencoder import BiEncoder
from tqdm import tqdm

def _BGEM3_encode(self, texts, batch_size=None, max_length=8192, return_dense=True, return_sparse=False, return_colbert_vecs=False):
    # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
    show_progress = False
    if isinstance(texts, tqdm):
        texts.disable = True
    embeddings = self.model.encode(list(texts), batch_size=batch_size or self.batch_size, max_length=max_length,
                             return_dense=return_dense, return_sparse=return_sparse, return_colbert_vecs=return_colbert_vecs)
    # TODO: add feature to also return sparse and colbert vecs
    return embeddings['dense_vecs']


class BGEM3Encoder(BiEncoder):
    def __init__(self, model_name='BAAI/bge-m3', batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(self.model_name,  
                       use_fp16=True, device=self.device) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.model = model
        self.config = AutoConfig.from_pretrained(model_name)

    encode_queries = _BGEM3_encode
    encode_docs = _BGEM3_encode

    def __repr__(self):
        return f'BGEM3Encoder({repr(self.model_name)})'