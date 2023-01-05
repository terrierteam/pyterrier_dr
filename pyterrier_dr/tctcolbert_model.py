from more_itertools import chunked
import numpy as np
import torch
from torch import nn
import ir_datasets
import pyterrier as pt
from transformers import RobertaConfig, AutoTokenizer, AutoModel, AdamW


logger = ir_datasets.log.easy()

class BiEncoder(pt.Transformer):

    def __init__(self, model, batch_size=32, text_field='text', verbose=False, tokenizer=None, cuda=None):
        self.model_name = str(model)
        if isinstance(model,str):
            if tokenizer is None:
                tokenizer = model
            self.model = model
        else:
            self.model = AutoModel.from_pretrained(model).eval()
            
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        assert self.tokenizer is not None

        self.cuda = torch.cuda.is_available() if cuda is None else cuda
        if self.cuda:
            self.model = self.model.cuda()
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose

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

    def _encode_queries(self, texts):
        pass

    def _encode_docs(self, texts):
        pass

    def _transform_D(self, inp):
        """
        Document vectorisation
        """
        res = self._encode_docs(inp[self.text_field])
        return inp.assign(doc_vec=[res[i] for i in range(res.shape[0])])

    def _transform_Q(self, inp):
        """
        Query vectorisation
        """
        it = inp['query']
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        res = self._encode_queries(it)
        return inp.assign(query_vec=[res[i] for i in range(res.shape[0])])

    def _transform_R(self, inp):
        """
        Result re-ranking
        """
        return pt.apply.by_query(self._transform_R_byquery, add_ranks=True, verbose=self.verbose)(inp)

    def _transform_R_byquery(self, query_df):
        query_rep = self._encode_queries([query_df['query'].iloc[0]])
        doc_reps = self._encode_docs(query_df[self.text_field])
        scores = (query_rep * doc_reps).sum(axis=1)
        query_df['score'] = scores
        return query_df

class TctColBert(pt.Transformer):
    def __init__(self, model_name='castorini/tct_colbert-msmarco', **kwargs):
        super().__init__(model_name, **kwargs)
        self._optimizer = None

    def _encode_queries(self, texts):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                inps = self.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=36)
                if self.cuda:
                    inps = {k: v.cuda() for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state
                res = res[:, 4:, :].mean(dim=1) # remove the first 4 tokens (representing [CLS] [ Q ]), and average
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

    def _encode_docs(self, texts):
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, self.batch_size):
                inps = self.tokenizer([f'[CLS] [D] {d}' for d in chunk], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=512)
                if self.cuda:
                    inps = {k: v.cuda() for k, v in inps.items()}
                res = self.model(**inps).last_hidden_state
                res = res[:, 4:, :] # remove the first 4 tokens (representing [CLS] [ D ])
                res = res * inps['attention_mask'][:, 4:].unsqueeze(2) # apply attention mask
                lens = inps['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1)
                lens[lens == 0] = 1 # avoid edge case of div0 errors
                res = res.sum(dim=1) / lens # average based on dim
                results.append(res.cpu().numpy())
        return np.concatenate(results, axis=0)

    def fit(self, dataset, pair_it=None, steps=100_000, lr=3e-6, in_batch_negs=False):
        self.model.train()
        if self._optimizer is None:
            optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr, eps=1e-8)
            self._optimizer = optimizer
        else:
            optimizer = self._optimizer
        optimizer.zero_grad()
        loss_fn = nn.CrossEntropyLoss()
        irds = dataset.irds_ref()
        running_losses, running_accs = [], []
        if pair_it is None:
            pair_it = iter(irds.docpairs)
        with logger.pbar_raw(total=steps) as pbar:
            for i, chunk in enumerate(chunked(pair_it, self.batch_size//2)):
                if i == steps:
                    break
                batch_q, batch_d = [], []
                for qid, pos_did, neg_did in chunk:
                    q = irds.queries.lookup(qid).text
                    batch_q.append(q)
                    if not in_batch_negs:
                        batch_q.append(q)
                    batch_d.append(irds.docs.lookup(pos_did).text)
                    batch_d.append(irds.docs.lookup(neg_did).text)
                batch_q = self.tokenizer([f'[CLS] [Q] {q} ' + ' '.join(['[MASK]'] * 32) for q in batch_q], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=36)
                batch_d = self.tokenizer([f'[CLS] [D] {d}' for d in batch_d], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=256)
                if self.cuda:
                    batch_q = {k: v.cuda() for k, v in batch_q.items()}
                    batch_d = {k: v.cuda() for k, v in batch_d.items()}

                Q = self.model(**batch_q).last_hidden_state
                Q = Q[:, 4:, :].mean(dim=1) # remove the first 4 tokens (representing [CLS] [ Q ]), and average

                D = self.model(**batch_d).last_hidden_state
                D = D[:, 4:, :] # remove the first 4 tokens (representing [CLS] [ D ])
                D = D * batch_d['attention_mask'][:, 4:].unsqueeze(2) # apply attention mask
                D = D.sum(dim=1) / batch_d['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1) # average based on dim

                if in_batch_negs:
                    scores = torch.einsum('qe,de->qd', Q, D)
                    targets = (torch.arange(batch_q['input_ids'].shape[0]) * 2).to(scores.device)
                    loss = loss_fn(scores, targets)
                    acc = ((scores.max(dim=1).indices == targets).sum() / scores.shape[0]).cpu().detach().item()
                else:
                    scores = torch.einsum('be,be->b', Q, D).reshape(-1, 2)
                    loss = loss_fn(scores, torch.zeros_like(scores[:, 0]).long())
                    acc = ((scores.max(dim=1).indices == 0).sum() / scores.shape[0]).cpu().detach().item()
                loss.backward()
                loss = loss.cpu().detach().item()
                optimizer.step()
                optimizer.zero_grad()
                pbar.update()
                running_losses.append(loss)
                running_accs.append(acc)
                pbar.set_postfix({'loss': loss, 'acc': acc})
                if len(running_losses) == 100:
                    logger.info(f'it={i+1} loss={sum(running_losses)/len(running_losses)} acc={sum(running_accs)/len(running_accs)}')
                    running_losses, running_accs = [], []

    

    def reverse(self, doc):
        with torch.no_grad():
            inps = self.tokenizer([f'[CLS] [D] {doc}'], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True, max_length=512)
            if self.cuda:
                inps = {k: v.cuda() for k, v in inps.items()}
            res = self.model(**inps).last_hidden_state
            res = res[:, 4:, :] # remove the first 4 tokens (representing [CLS] [ D ])
            res = res * inps['attention_mask'][:, 4:].unsqueeze(2) # apply attention mask
            res = res.sum(dim=1) / inps['attention_mask'][:, 4:].sum(dim=1).unsqueeze(1) # average based on dim
            target = res
        pembed_inp = torch.arange(32).unsqueeze(0).cuda()
        wembed_inp = torch.arange(len(self.tokenizer)).unsqueeze(0).cuda()
        tembed_inp = torch.zeros(32).long().unsqueeze(0).cuda()
        inp = torch.ones(1, 32, len(self.tokenizer)).float()
        if self.cuda:
            inp = inp.cuda()
        inp = torch.nn.Parameter(inp)
        optimizer = torch.optim.Adam([inp], lr=1.)
        import ir_datasets
        logger = ir_datasets.log.easy()
        lossfn = torch.nn.MSELoss()
        with logger.pbar_raw(desc='optim') as pbar:
            while True:
                model_inp = self.model.embeddings.LayerNorm((inp.softmax(dim=2) @ self.model.embeddings.word_embeddings(wembed_inp).squeeze(0)) + self.model.embeddings.position_embeddings(pembed_inp).squeeze(0) + self.model.embeddings.token_type_embeddings(tembed_inp).squeeze(0))
                out = self.model.encoder(model_inp).last_hidden_state
                out = out[:, 4:, :].mean(dim=1) # remove the first 4 tokens (representing [CLS] [ Q ])
                recreation_loss = lossfn(out, target)
                softmax_loss = (1. - inp.softmax(dim=2).max(dim=1)[0]).mean()
                # loss = recreation_loss + softmax_loss
                loss = softmax_loss
                cos = (out * target).sum() / torch.norm(out) / torch.norm(target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({'loss': loss.cpu().item(), 'recreation_loss': recreation_loss.cpu().item(), 'softmax_loss': softmax_loss.cpu().item(), 'cos': cos.cpu().item()})
                pbar.update()
                res = self.tokenizer.decode(inp.argmax(dim=2)[0])
                logger.info(res)
                # if loss.cpu().item() < 0.1:
                #     break
        import pdb; pdb.set_trace()
        inp
