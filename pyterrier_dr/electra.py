import numpy as np
import more_itertools
import torch
import pyterrier as pt
from transformers import AutoTokenizer, ElectraForSequenceClassification

class ElectraScorer(pt.Transformer):
    def __init__(self, model_name='crystina-z/monoELECTRA_LCE_nneg31', batch_size=16, text_field='text', verbose=True, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.tokeniser = AutoTokenizer.from_pretrained('google/electra-base-discriminator')
        self.model = ElectraForSequenceClassification.from_pretrained(model_name).eval().to(self.device)

    def transform(self, inp):
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='ELECTRA scoring')
        with torch.no_grad():
            for chunk in more_itertools.chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                inps = self.tokeniser(queries, texts, return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                scores.append(self.model(**inps).logits[:, 1].cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res
