import re
import math
import json
import ir_datasets
import torch
import numpy as np
import pyterrier as pt
import pyterrier_dr
from ..indexes import TorchRankedLists
from . import FlexIndex


def _corpus_graph(self, k=16, batch_size=8192):
    from pyterrier_adaptive import CorpusGraph
    key = ('corpus_graph', k)
    if key not in self._cache:
        path = self.index_path/f'corpusgraph_k{k}'
        if not (path/'pt_meta.json').exists():
            candidates = [(p, re.match(r'^corpusgraph_k([0-9]+)$', str(p).split('/')[-2])) for p in self.index_path.glob('corpusgraph_k*/pt_meta.json')]
            candidates = sorted([(int(m.group(1)), p) for p, m in candidates if m is not None and int(m.group(1)) > k])
            if candidates:
                self._cache[key] = CorpusGraph.load(candidates[0][1].parent).to_limit_k(k)
            else:
                _build_corpus_graph(self, k, path, batch_size)
                self._cache[key] = CorpusGraph.load(path)
        else:
            self._cache[key] = CorpusGraph.load(path)
    return self._cache[key]


FlexIndex.corpus_graph = _corpus_graph


def _build_corpus_graph(flex_index, k, out_dir, batch_size):
    S = batch_size
    vectors, meta = flex_index.payload(return_docnos=False)
    num_vecs = vectors.shape[0]
    num_chunks = math.ceil(num_vecs/S)
    rankings = [TorchRankedLists(k, min((i+1)*S, num_vecs) - i*S) for i in range(num_chunks)]
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_path = out_dir/'edges.u32.np'
    weights_path = out_dir/'weights.f16.np'
    device = pyterrier_dr.util.infer_device()
    dtype = torch.half if device.type == 'cuda' else torch.float
    with pt.tqdm(total=int((num_chunks+1)*num_chunks/2), unit='chunk', smoothing=1) as pbar, \
        ir_datasets.util.finialized_file(str(edges_path), 'wb') as fe, \
        ir_datasets.util.finialized_file(str(weights_path), 'wb') as fw:
        for i in range(num_chunks):
            left = torch.from_numpy(vectors[i*S:(i+1)*S]).to(device).to(dtype)
            left = left / left.norm(dim=1, keepdim=True)
            scores = left @ left.T
            scores[torch.eye(left.shape[0], dtype=bool, device=device)] = float('-inf')
            i_scores, i_dids = scores.topk(k, sorted=True, dim=1)
            rankings[i].update(i_scores, (i_dids + i*S))
            pbar.update()
            for j in range(i+1, num_chunks):
                right = torch.from_numpy(vectors[j*S:(j+1)*S]).to(device).to(dtype)
                right = right / right.norm(dim=1, keepdim=True)
                scores = left @ right.T
                i_scores, i_dids = scores.topk(min(k, right.shape[0]), sorted=True, dim=1)
                j_scores, j_dids = scores.topk(min(k, left.shape[0]), sorted=True, dim=0)
                rankings[i].update(i_scores, (i_dids + j*S))
                rankings[j].update(j_scores.T, (j_dids + i*S).T)
                pbar.update()
            sims, idxs = rankings[i].results()
            fe.write(idxs.astype(np.uint32).tobytes())
            fw.write(sims.astype(np.float16).tobytes())
            rankings[i] = None
    (out_dir/'docnos.npids').symlink_to('../docnos.npids')
    with (out_dir/'pt_meta.json').open('wt') as fout:
      json.dump({
        'type': 'corpus_graph',
        'format': 'np_topk',
        'doc_count': vectors.shape[0],
        'k': k,
      }, fout)
