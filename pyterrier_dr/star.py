from .biencoder import BiEncoder
import numpy as np
from more_itertools import chunked
from transformers import RobertaModel
from torch import nn
import torch
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

class RobertaDot(RobertaPreTrainedModel):
    def __init__(self, config):
        RobertaPreTrainedModel.__init__(self, config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.embeddingHead = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.apply(self._init_weights)               
    
    def forward(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = outputs1[0][:, 0]
        embeds = self.norm(self.embeddingHead(full_emb))
        return embeds

class STAR(BiEncoder):

    def __init__(self, model_name, batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size=batch_size, verbose=verbose, text_field=text_field)
        self.device = device or torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = RobertaDot.from_pretrained(model_name).to(self.device).eval()

        # uses a copy of the old tokenizer code, and therefore the old hgf API
        from ._star_tokenizer import RobertaTokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def encode_queries_torch(self, texts, batch_size=None):
        results = []
        for chunk in chunked(texts, batch_size or self.batch_size):
            inps = self.tokenizer.batch_encode_plus(list(chunk),  max_length=192, return_tensors='pt', pad_to_max_length=True, truncation=True)
            inps = {k: v.to(self.device) for k, v in inps.items()}
            res = self.model.forward(**inps)
            results.append(res)
        if not results:
            return torch.zeros(shape=(0, 0))
        return torch.cat(results, dim=0)
    
    def encode_docs(self, texts, batch_size=None):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer.batch_encode_plus(list(chunk), max_length=192, return_tensors='pt', pad_to_max_length=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model.forward(**inps)
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)
    
