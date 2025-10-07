import numpy as np
import torch
import pyterrier as pt
import pyterrier_alpha as pta

LIGHTNING_IR_AVAILIBLE = False
try:
    import lightning_ir as L
    LIGHTNING_IR_AVAILIBLE = True
except ImportError:
    pass


class LightningIRMonoScorer(pt.Transformer):
    def __init__(self,
                 model_name='webis/monoelectra-base',
                 batch_size=16,
                 text_field='text',
                 verbose=True,
                 device=None
                 ):
        if not LIGHTNING_IR_AVAILIBLE:
            raise ImportError("lightning_ir is required for LightningIRMonoScorer. Please install it via 'pip install lightning-ir'")
        self.model_name = model_name
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = L.CrossEncoderModule(model_name).eval().to(self.device)

    def transform(self, inp):
        pta.validate.columns(inp, includes=['query', self.text_field])
        if len(inp) == 0:
            res = inp.assign(score=np.empty(shape=(0,), dtype=np.float32))
            pt.model.add_ranks(res)
            return res.sort_values(['qid', 'rank'])

        tmp = inp.reset_index().rename(columns={'index': '_row'})
        g = tmp.groupby('query', sort=False)[['_row', self.text_field]].agg(list)

        queries = g.index.tolist()                 # List[str]
        doclists = g[self.text_field].tolist()      # List[List[str]]
        idxlists = g['_row'].tolist()               # List[List[int]]

        out_scores = np.empty(len(inp), dtype=np.float32)

        n_groups = len(queries)
        iterator = range(n_groups)
        if self.verbose:
            iterator = pt.tqdm(iterator, total=n_groups, unit='query', desc=f'{self.model_name} scoring')

        start = 0
        for _ in iterator:
            if start >= n_groups:
                break

            # Pack queries until adding the next would exceed the pair budget
            pairs = 0
            end = start
            lengths = []
            while end < n_groups:
                need = len(doclists[end])
                if pairs > 0 and (pairs + need > self.batch_size):
                    break
                pairs += need
                lengths.append(need)
                end += 1

            q_batch = queries[start:end]
            d_batch = doclists[start:end]
            idx_batch = idxlists[start:end]

            with torch.no_grad(), torch.inference_mode():
                flat = self.model.score(q_batch, d_batch).scores.detach().cpu().numpy().reshape(-1)

            splits = np.cumsum(lengths)[:-1]
            per_query = np.split(flat, splits) if splits.size > 0 else [flat]
            for row_idxs, sc in zip(idx_batch, per_query):
                out_scores[np.asarray(row_idxs, dtype=int)] = sc

            start = end

        res = inp.assign(score=out_scores)
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res
