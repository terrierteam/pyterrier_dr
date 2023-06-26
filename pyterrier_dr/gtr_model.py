from more_itertools import chunked
import numpy as np
import torch
from torch import nn
import ir_datasets
import pyterrier as pt
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import RobertaConfig, AutoTokenizer, AutoModel, AdamW


logger = ir_datasets.log.easy()


class GTRT5(pt.Transformer):
    def __init__(self, model_name='sentence-transformers/gtr-t5-large', batch_size=32, text_field='text', verbose=False, cuda=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        self.model = AutoModel.from_pretrained(model_name).eval()
        if self.cuda:
            self.model = self.model.cuda()
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose
        self._optimizer = None

    def transform(self, inp):
        columns = set(inp.columns)
        modes = [
            (['qid', 'query', 'docno', self.text_field], self._transform_R),
            (['qid', 'query'], self._transform_Q),
            (['docno', self.text_field], self._transform_D),
        ]
        for fields, fn in modes:
            if all(f in columns for f in fields):
                return fn(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in modes:
            f += f'\n - {fn.__doc__.strip()}: {columns}\n'
        raise RuntimeError(message)

    def _transform_D(self, inp):
        """
        Document vectorisation
        """
        res = self.model.encode(inp[self.text_field], convert_to_numpy=True, batch_size=self.batch_size)
        # Using inp.assign here (instead of pt.apply) ends up being much faster.
        inp = inp.assign(doc_vec=[res[i] for i in range(res.shape[0])])
        return inp
    
    def _transform_Q(self, inp):
        """
        Query vectorisation
        """
        return pt.apply.query_vec(lambda r: self.model.encode(r['query'], convert_to_numpy=True, batch_size=self.batch_size))(inp)

    def _transform_R(self, inp):
        """
        Result re-ranking
        """
        return pt.apply.by_query(self._transform_R_byquery, add_ranks=True, verbose=self.verbose)(inp)

    def _transform_R_byquery(self, query_df):
        texts = [query_df['query'].iloc[0]] + list(query_df[self.text_field])
        reps = self.model.encode(texts, convert_to_tensor=True, batch_size=self.batch_size)
        query_rep = reps[0:1] # keep batch dim; allows broadcasting
        doc_reps = reps[1:]
        if self.score_fn == 'cos':
            scores = util.cos_sim(query_rep, doc_reps)[0].cpu().numpy()
        elif self.score_fn == 'dot':
            scores = util.dot_score(query_rep, doc_reps)[0].cpu().numpy()
        else:
            raise ValueError('unknown score_fn')
        query_df['score'] = scores
        return query_df