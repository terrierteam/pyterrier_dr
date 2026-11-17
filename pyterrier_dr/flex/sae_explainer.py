"""Sparse autoencoder (SAE) explanations for dense index vectors.

Implements "Decoding Dense Embeddings: Sparse Autoencoders for Interpreting and Discretizing
Dense Retrieval" (https://arxiv.org/abs/2403.16058) as an extension of :class:`FlexIndex`:
a Batch-TopK sparse autoencoder is trained over the indexed document vectors, per-document
latent concepts are extracted, latents can be described with an LLM, and the descriptions
are used to explain queries and documents (see :meth:`FlexIndex.train_sae
<pyterrier_dr.FlexIndex.train_sae>`, :meth:`FlexIndex.extract_latent_concepts
<pyterrier_dr.FlexIndex.extract_latent_concepts>`, :meth:`FlexIndex.generate_latent_descriptions
<pyterrier_dr.FlexIndex.generate_latent_descriptions>` and :meth:`FlexIndex.latent_describer
<pyterrier_dr.FlexIndex.latent_describer>`).

:meth:`FlexIndex.train_sae <pyterrier_dr.FlexIndex.train_sae>` serialises the training results
to the index directory: ``sae.pt`` (the model checkpoint), the ``sae`` section of
``pt_meta.json`` (the configuration, including the activation threshold) and
``latent_concepts.jsonl`` (the latent concepts of all documents).
"""
import heapq
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyterrier as pt
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import FlexIndex

SAE_CHECKPOINT_FORMAT = 'flex_sae'
SAE_CHECKPOINT_VERSION = 1
LATENT_CONCEPTS_FILENAME = 'latent_concepts.jsonl'


def _sae_device(device: Optional[Union[str, t.device]] = None) -> t.device:
    """Resolve a torch device, preferring CUDA, then Apple MPS, then CPU (unless a device is given)."""
    if device is not None:
        return t.device(device)
    if t.cuda.is_available():
        return t.device('cuda')
    if getattr(t.backends, 'mps', None) is not None and t.backends.mps.is_available():
        return t.device('mps')
    return t.device('cpu')


class SAE_bias_pre(nn.Module):
    """Batch-TopK sparse autoencoder with dead-latent auxiliary loss
    (adapted from Decoding-Dense-Embeddings/sae/train.py).

    The decoder weights are initialised as the transpose of the encoder weights, and are
    kept row-unit-normed during training.
    """

    def __init__(self, input_dim: int, sparse_dim: int, k: int, dead_threshold: int, bias_pre: t.Tensor):
        super().__init__()
        self.linear = nn.Linear(input_dim, sparse_dim, bias=False)
        self.linear2 = nn.Linear(sparse_dim, input_dim, bias=False)
        self.linear2.weight = nn.Parameter(self.linear.weight.clone().T)
        self.bias_pre = nn.Parameter(bias_pre.clone())
        self.k = k
        self.threshold = dead_threshold
        self.register_buffer("last_activation", t.zeros(sparse_dim))
        self.mean_of_min_act = 0.0

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        out = self.linear(x - self.bias_pre)
        bs = out.shape[0]
        flat = out.reshape(-1)
        topk_values, topk_indices = t.topk(flat, k=self.k * bs)
        self.mean_of_min_act += topk_values[-1].item()
        topk = t.zeros_like(flat)
        topk.scatter_(0, topk_indices, topk_values)
        topk = topk.reshape(bs, -1)

        # track how many steps each latent has been inactive
        fired = (topk != 0).any(dim=0)
        self.last_activation = (self.last_activation + 1) * (~fired)
        dead_mask = self.last_activation > self.threshold

        dead_act = out.masked_fill(~dead_mask, float("-inf"))
        dead_vals, dead_idx = t.topk(dead_act, k=min(self.k * 2, out.shape[1]))
        dead_vals = t.where(t.isinf(dead_vals), t.zeros_like(dead_vals), dead_vals)
        dead_topk = t.zeros_like(out).scatter_(1, dead_idx, dead_vals)
        dead_recon = self.linear2(dead_topk)

        recon = self.linear2(topk) + self.bias_pre
        return recon, dead_recon

    @t.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        w = self.linear2.weight
        w_normed = w / w.norm(dim=-1, keepdim=True)
        w.grad -= (w.grad * w_normed).sum(-1, keepdim=True) * w_normed
        w.data = w_normed

    @t.no_grad()
    def get_thresh_recon(self, x: t.Tensor, thresh: float) -> t.Tensor:
        out = self.linear(x - self.bias_pre)
        return self.linear2(out * (out > thresh)) + self.bias_pre

    def config(self) -> Dict:
        """The architecture hyperparameters needed to reconstruct this model from a checkpoint."""
        return {
            'input_dim': self.linear.in_features,
            'sparse_dim': self.linear.out_features,
            'k': self.k,
            'dead_threshold': self.threshold,
        }


def train_sae(vecs: np.ndarray, *, hidden_mult: int = 8, k: int = 32, batch_size: int = 256,
              lr: float = 5e-5, epochs: int = 100, dead_threshold_step: int = 20,
              aux_loss_alpha: float = 1 / 32, seed: int = 42,
              device: Optional[Union[str, t.device]] = None, verbose: bool = True) -> Tuple[SAE_bias_pre, float, float]:
    """Train a Batch-TopK SAE on an array of dense vectors (``n x dim``).

    Args:
        vecs: The vectors to train on (e.g. the document vectors of a :class:`FlexIndex`).
        hidden_mult: The number of latents per input dimension.
        k: The number of latents to keep per vector (Batch-TopK).
        batch_size: The number of vectors per optimisation step.
        lr: The learning rate for the AdamW optimiser.
        epochs: The number of epochs to train for.
        dead_threshold_step: The number of steps a latent can go without firing before it is
            targeted by the dead-latent auxiliary loss.
        aux_loss_alpha: The weight of the dead-latent auxiliary loss.
        seed: The random seed.
        device: The device to train on (defaults to cuda, then mps, then cpu).
        verbose: Whether to display a progress bar and print the results.

    Returns:
        Tuple of ``(model, act_thresh, nmse)``:

        - ``model``: the trained :class:`SAE_bias_pre`.
        - ``act_thresh``: the inference activation threshold (mean of the smallest kept
          activation per batch); use it to sparsely encode new vectors.
        - ``nmse``: the normalised mean squared error of the thresholded reconstruction
          (relative to predicting the mean vector).
    """
    device = _sae_device(device)
    t.manual_seed(seed)
    data = t.tensor(vecs, dtype=t.float32)
    input_dim = data.shape[1]
    sparse_dim = input_dim * hidden_mult
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=len(data) >= batch_size)
    n_batches = len(loader)

    model = SAE_bias_pre(input_dim, sparse_dim, k, dead_threshold_step, data.mean(0)).to(device)
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, eps=6e-10)

    model.train()
    pbar = pt.tqdm(range(epochs), disable=not verbose, desc='Training SAE')
    for _ in pbar:
        running, model.mean_of_min_act = 0.0, 0.0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, dead_recon = model(x)
            err = x - recon
            loss = (err.square() + aux_loss_alpha * (err - dead_recon).square()).mean()
            loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
            model.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            running += loss.item()
        pbar.set_postfix(loss=running / n_batches)

    # inference threshold = mean of the smallest kept activation per batch
    act_thresh = model.mean_of_min_act / n_batches

    # NMSE of the thresholded reconstruction vs. predicting the mean
    model.eval()
    x = data.to(device)
    mse = (x - model.get_thresh_recon(x, act_thresh)).square().mean()
    bmse = (x - x.mean(0)).square().mean()
    nmse = (mse / bmse).item()
    if verbose:
        print(f"act_thresh={act_thresh:.4f}  NMSE={nmse:.4f}")
    return model, act_thresh, nmse


@t.no_grad()
def extract_latent_concepts(sae: SAE_bias_pre, vecs: np.ndarray, threshold: float, *,
                            batch_size: int = 4096, device: Optional[Union[str, t.device]] = None,
                            verbose: bool = True) -> List[Dict]:
    """Sparse-encode vectors with a trained SAE.

    Args:
        sae: a trained :class:`SAE_bias_pre`.
        vecs: the vectors to encode (``n x dim``).
        threshold: latents whose activation does not exceed this threshold are dropped
            (use the ``act_thresh`` returned by :func:`train_sae`).
        batch_size: the number of vectors to encode at once.
        device: the device to encode on.
        verbose: whether to display a progress bar.

    Returns:
        A list with one entry per vector: ``{'docid': i, 'ids': [...], 'weight': [...]}``,
        where ``ids`` are the latents whose activation exceeds ``threshold`` and
        ``weight`` the corresponding activations.
    """
    device = _sae_device(device)
    sae.eval()
    results = []
    for start in pt.tqdm(range(0, len(vecs), batch_size), desc='Extracting latent concepts', disable=not verbose):
        batch = t.tensor(vecs[start:start + batch_size], dtype=t.float32).to(device)
        acts = sae.linear(batch - sae.bias_pre)
        rows, cols = (acts > threshold).nonzero(as_tuple=True)
        weights = acts[rows, cols].cpu().tolist()
        rows, cols = rows.cpu().tolist(), cols.cpu().tolist()
        per_doc = [{"docid": start + i, "ids": [], "weight": []} for i in range(len(batch))]
        for r, c, w in zip(rows, cols, weights):
            per_doc[r]["ids"].append(c)
            per_doc[r]["weight"].append(w)
        results.extend(per_doc)
    return results


SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an interpretation that thoroughly encapsulates possible patterns found in it.
Guidelines:
You will be given a list of text examples on which a certain common pattern might be present. How important each text is for the pattern is listed after each text.
- Try to produce a concise final description. Simply describe the text latents that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Based on the found patterns summarize your interpretation in 1-8 words.
- Do not make lists of possible interpretations. Keep your interpretations short and concise.
- The last line of your response must be the formatted interpretation, using [interpretation]:
"""

BATCH_INSTRUCTIONS = """
You will be given several independent patterns, each sent as a separate user message beginning with a header '### Pattern N' followed by its examples. Analyze each pattern separately.
Instead of the single last line above, finish your response with exactly one line per pattern, in order, formatted as [interpretation N]: <1-8 words>, where N is the pattern number.
"""


def generate_descriptions(latent_concepts: List[Dict], get_texts: Callable[[List[int]], Dict[int, str]], *,
                          top_k: int = 10, latent_ids: Optional[List[int]] = None,
                          top_proportion: Optional[float] = None,
                          api_key: Optional[str] = None, base_url: Optional[str] = None,
                          model: str = "gpt-4.1-mini", max_chars: int = 1500,
                          batch_size: int = 8, save_path: Optional[str] = None,
                          verbose: bool = True) -> Dict[int, str]:
    """Label each latent with an LLM, using the documents in which it activates most strongly.

    Args:
        latent_concepts: the result of :func:`extract_latent_concepts`.
        get_texts: a callable that takes a list of ``docid`` and returns ``{docid: text}``.
        top_k: the number of most strongly activating documents to show per latent.
        latent_ids: an optional subset of the latents to describe (defaults to all that ever activate).
        top_proportion: an alternative to ``latent_ids``: only describe the most frequently
            activated latents, up to this proportion of all activated latents (in ``(0, 1]``,
            ranked by the number of documents in which each latent activates).
        api_key, base_url, model: select the OpenAI-compatible LLM endpoint (if omitted, the
            ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` environment variables are used).
        max_chars: the maximum number of characters of each example text to show.
        batch_size: the number of latents to describe per API call (1 = one latent per call).
        save_path: if given, the interpretations are also written as JSONL.
        verbose: whether to display a progress bar.

    Returns:
        A dict of ``{latent_id: interpretation}``.

    Requires the ``openai`` package.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError('the "openai" package is required to generate latent descriptions; install it with `pip install openai`')

    if latent_ids is not None and top_proportion is not None:
        raise ValueError('latent_ids and top_proportion are mutually exclusive')
    if top_proportion is not None and not 0 < top_proportion <= 1:
        raise ValueError('top_proportion must be in (0, 1]')

    client = OpenAI(api_key=api_key, base_url=base_url)

    latent_docs = defaultdict(list)
    for entry in latent_concepts:
        for lid, w in zip(entry["ids"], entry["weight"]):
            latent_docs[lid].append((entry["docid"], w))
    if top_proportion is not None:
        # the most frequently activated latents, up to the requested proportion of all activated latents
        n = max(1, math.ceil(top_proportion * len(latent_docs)))
        latent_ids = [lid for lid, _ in sorted(latent_docs.items(), key=lambda x: len(x[1]), reverse=True)[:n]]
    if latent_ids is not None:
        latent_docs = {l: latent_docs[l] for l in latent_ids if l in latent_docs}

    # selects the top-k activating documents for each latent, and retrieves their text
    top = {l: heapq.nlargest(top_k, docs, key=lambda x: x[1]) for l, docs in latent_docs.items()}
    texts = get_texts([d for docs in top.values() for d, _ in docs])

    items = list(top.items())
    interps = {}
    missing = 0
    for start in pt.tqdm(range(0, len(items), batch_size), desc='Describing latent batches', disable=not verbose):
        batch = items[start:start + batch_size]
        user_messages = []
        for n, (lid, docs) in enumerate(batch):
            content = f"### Pattern {n}\n" + "".join(
                f"Example{i}: {texts[d][:max_chars]}\nActivation: {a}\n" for i, (d, a) in enumerate(docs)
            )
            user_messages.append({"role": "user", "content": content})
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "developer", "content": SYSTEM_PROMPT + BATCH_INSTRUCTIONS}] + user_messages,
        )
        found = {int(n): s.strip() for n, s in re.findall(r"\[interpretation (\d+)\]:(.*)", completion.choices[0].message.content)}
        for n, (lid, _) in enumerate(batch):
            if n in found:
                interps[lid] = found[n]
            else:
                print(f"warning: no interpretation returned for latent {lid}")
                missing += 1
    if missing > 0:
        print(f"warning: {missing} latents out of {len(items)} ({missing/len(items)*100:.1f}%) were not interpreted by the LLM")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            for lid, interp in interps.items():
                f.write(json.dumps({"id": lid, "interp": interp}, ensure_ascii=False) + "\n")
    return interps


class LatentDescriber(pt.Transformer):
    """Annotates queries (``query_vec``) or documents (``doc_vec``) with the descriptions of their most active SAE latents.

    Adds columns: ``latent_ids``, ``latent_weights``, ``latent_descs`` (lists, strongest first).
    ``descriptions`` is a dict of ``{latent_id: text}``; with ``only_described=True``, latents
    lacking a description are skipped.
    """

    def __init__(self, sae: SAE_bias_pre, descriptions: Dict[int, str], *, threshold: float = 0.0,
                 num_latents: int = 5, only_described: bool = True, batch_size: int = 1024,
                 vec_col: Optional[str] = None, device: Optional[Union[str, t.device]] = None):
        self.sae = sae
        self.descriptions = descriptions
        self.threshold = threshold
        self.num_latents = num_latents
        self.only_described = only_described
        self.batch_size = batch_size
        self.vec_col = vec_col
        self.device = _sae_device(device)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        with pt.validate.any(inp, context=self) as v:
            if self.vec_col:
                v.columns(includes=[self.vec_col], mode=self.vec_col)
            else:
                v.columns(includes=["query_vec"], mode="query_vec")
                v.columns(includes=["doc_vec"], mode="doc_vec")
        col = v.mode
        vecs_in = np.stack(inp[col].values).astype(np.float32)

        mask = None
        if self.only_described:
            mask = t.zeros(self.sae.linear.out_features, dtype=t.bool, device=self.device)
            mask[[int(i) for i in self.descriptions]] = True

        ids, weights, descs = [], [], []
        self.sae.eval()
        with t.no_grad():
            for s in range(0, len(vecs_in), self.batch_size):
                x = t.as_tensor(vecs_in[s:s + self.batch_size]).to(self.device)
                acts = self.sae.linear(x - self.sae.bias_pre)
                if mask is not None:
                    acts = acts.masked_fill(~mask, float("-inf"))
                vals, idx = t.topk(acts, k=min(self.num_latents, acts.shape[1]), dim=1)
                for v_row, i_row in zip(vals.cpu().tolist(), idx.cpu().tolist()):
                    keep = [(i, v) for i, v in zip(i_row, v_row) if v > self.threshold]
                    ids.append([i for i, _ in keep])
                    weights.append([v for _, v in keep])
                    descs.append([self.descriptions.get(i, f"latent_{i}") for i, _ in keep])
        return inp.assign(latent_ids=ids, latent_weights=weights, latent_descs=descs)


def _assert_built(self):
    if not self.built():
        raise RuntimeError(f'index {self.index_path} has not been built')


def _sae_meta(self) -> Optional[Dict]:
    """Return the SAE configuration metadata stored in this index's ``pt_meta.json`` (or ``None``)."""
    meta, = self.payload(return_dvecs=False, return_docnos=False)
    return meta.get('sae')


def _update_meta(self, update: Dict):
    """Merge ``update`` into this index's ``pt_meta.json`` on disk."""
    with open(self.index_path / 'pt_meta.json', 'rt') as f:
        meta = json.load(f)
    meta.update(update)
    with open(self.index_path / 'pt_meta.json', 'wt') as f:
        json.dump(meta, f)
    self._meta = meta


def _build_sae(config: Dict) -> SAE_bias_pre:
    return SAE_bias_pre(
        config['input_dim'],
        config['sparse_dim'],
        config['k'],
        config['dead_threshold'],
        t.zeros(config['input_dim'], dtype=t.float32),
    )


def _train_sae(self, *, hidden_mult: int = 8, k: int = 32, batch_size: int = 256,
               lr: float = 5e-5, epochs: int = 100, dead_threshold_step: int = 20,
               aux_loss_alpha: float = 1 / 32, seed: int = 42,
               device: Optional[Union[str, t.device]] = None, save: bool = True):
    """Train a Batch-TopK sparse autoencoder over the vectors in this index.

    At the end of training, the model and the latent concepts of all documents are automatically
    serialised to the index: the checkpoint is written to ``sae.pt`` in the index directory and
    the configuration metadata (including the activation threshold and NMSE) is recorded in the
    index's ``pt_meta.json`` (via :meth:`save_sae`), and the latent concepts are written to
    ``latent_concepts.jsonl`` (via :meth:`extract_latent_concepts`).

    Args:
        hidden_mult: The number of latents per input dimension.
        k: The number of latents to keep per vector (Batch-TopK).
        batch_size: The number of vectors per optimisation step.
        lr: The learning rate for the AdamW optimiser.
        epochs: The number of epochs to train for.
        dead_threshold_step: The number of steps a latent can go without firing before it is
            targeted by the dead-latent auxiliary loss.
        aux_loss_alpha: The weight of the dead-latent auxiliary loss.
        seed: The random seed.
        device: The device to train on (defaults to cuda, then mps, then cpu).
        save: Whether to save the trained model to this index (see :meth:`save_sae`).

    Returns:
        Tuple of ``(model, act_thresh, nmse)``:

        - ``model``: the trained :class:`~pyterrier_dr.flex.sae_explainer.SAE_bias_pre`.
        - ``act_thresh``: the inference activation threshold.
        - ``nmse``: the normalised mean squared error of the thresholded reconstruction.
    """
    self._assert_built()
    vecs = self.np_vecs()
    model, act_thresh, nmse = train_sae(
        vecs,
        hidden_mult=hidden_mult,
        k=k,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        dead_threshold_step=dead_threshold_step,
        aux_loss_alpha=aux_loss_alpha,
        seed=seed,
        device=device,
        verbose=self.verbose,
    )
    if save:
        self.save_sae(model, act_thresh, config={
            'hidden_mult': hidden_mult,
            'epochs': epochs,
            'nmse': nmse,
        })
    # extract and serialise the latent concepts of all documents to the index
    self.extract_latent_concepts(sae=model, threshold=act_thresh, device=device)
    return model, act_thresh, nmse


def _save_sae(self, sae: SAE_bias_pre, act_thresh: float, *, config: Optional[Dict] = None,
              dest: Optional[str] = None):
    """Save a trained SAE to this index.

    The model weights and activation threshold are written to a torch checkpoint (``sae.pt``
    in the index directory, or ``dest`` if given), and the configuration metadata is recorded
    in the index's ``pt_meta.json`` (under the ``sae`` key).

    Args:
        sae: a trained :class:`~pyterrier_dr.flex.sae_explainer.SAE_bias_pre` model.
        act_thresh: the activation threshold to use for sparse encoding (see :meth:`train_sae`).
        config: an optional dict of extra configuration metadata to record (e.g. training hyperparameters).
        dest: an optional path for the checkpoint file (defaults to ``sae.pt`` in the index directory).

    Returns:
        The path of the checkpoint file written.
    """
    cfg = dict(sae.config())
    if config:
        cfg.update(config)
    cfg['act_thresh'] = act_thresh
    dest = Path(dest) if dest is not None else self.index_path / 'sae.pt'
    try:
        cfg['checkpoint'] = str(dest.relative_to(self.index_path))
    except ValueError:
        cfg['checkpoint'] = str(dest)
    t.save({
        'format': SAE_CHECKPOINT_FORMAT,
        'format_version': SAE_CHECKPOINT_VERSION,
        'state_dict': sae.state_dict(),
        'act_thresh': act_thresh,
        'config': cfg,
    }, dest)
    self._update_meta({'sae': cfg})
    return dest


def _load_sae(self, *, path: Optional[str] = None, device: Optional[Union[str, t.device]] = None):
    """Load the SAE saved to this index by :meth:`save_sae`.

    The model is cached in memory after it is first loaded.

    Args:
        path: an optional explicit path of the checkpoint (defaults to the checkpoint recorded
            in the index metadata).
        device: the device to place the model on.

    Returns:
        Tuple of ``(model, act_thresh)``.

    Raises:
        RuntimeError: If no SAE has been saved to this index.
    """
    meta_sae = self.sae_config()
    if path is None:
        if meta_sae is None:
            raise RuntimeError(f'No SAE has been saved for this index. Train one with FlexIndex.train_sae(), or save one with FlexIndex.save_sae().')
        path = self.index_path / meta_sae['checkpoint']
    key = ('sae', str(device), str(path))
    if key in self._cache:
        return self._cache[key]
    ckpt = t.load(path, map_location='cpu', weights_only=False)
    assert ckpt.get('format') == SAE_CHECKPOINT_FORMAT, f'Unrecognised SAE checkpoint at {path}'
    model = _build_sae(ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(_sae_device(device)).eval()
    act_thresh = ckpt['act_thresh']
    self._cache[key] = (model, act_thresh)
    return model, act_thresh


def _write_latent_concepts(results: List[Dict], save_path) -> None:
    with open(save_path, 'w', encoding='utf-8') as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')


def _load_latent_concepts(save_path) -> List[Dict]:
    with open(save_path, 'rt', encoding='utf-8') as fin:
        return [json.loads(line) for line in fin]


def _extract_latent_concepts(self, sae: Optional[SAE_bias_pre] = None, threshold: Optional[float] = None, *,
                             batch_size: int = 4096, device: Optional[Union[str, t.device]] = None,
                             save_path: Optional[str] = None) -> List[Dict]:
    """Extract the latent concepts of each document in this index.

    For each document vector, the latents whose activation exceeds ``threshold`` are recorded.
    The results are serialised to ``latent_concepts.jsonl`` in the index directory (or to
    ``save_path`` if given). If no SAE or threshold is given, and previously serialised latent
    concepts exist in the index directory, they are loaded from disk instead of being
    recomputed.

    Args:
        sae: a trained SAE model (e.g. as returned by :meth:`train_sae`). If omitted, the SAE
            previously saved to this index is loaded automatically (see :meth:`load_sae`).
        threshold: the activation threshold (defaults to the ``act_thresh`` saved by :meth:`train_sae`).
        batch_size: the number of vectors to encode at once.
        device: the device to encode on.
        save_path: the path to serialise the results to (defaults to ``latent_concepts.jsonl``
            in the index directory).

    Returns:
        A list with one entry per document: ``{'docid', 'docno', 'ids', 'weight'}``, where
        ``docid`` is the internal document id, ``ids`` the latents whose activation exceeds
        ``threshold`` and ``weight`` the corresponding activations.
    """
    self._assert_built()
    default_save_path = self.index_path / LATENT_CONCEPTS_FILENAME
    if sae is None and threshold is None and default_save_path.exists():
        results = _load_latent_concepts(default_save_path)
        if save_path is not None and Path(save_path) != default_save_path:
            _write_latent_concepts(results, save_path)
        return results
    if sae is None or threshold is None:
        loaded_sae, loaded_thresh = self.load_sae(device=device)
        if sae is None:
            sae = loaded_sae
        if threshold is None:
            threshold = loaded_thresh
    vecs = self.np_vecs()
    docnos, _ = self.payload(return_dvecs=False)
    results = extract_latent_concepts(sae, vecs, threshold, batch_size=batch_size, device=device, verbose=self.verbose)
    result_docnos = docnos.fwd[np.arange(len(results))]
    for entry, docno in zip(results, result_docnos):
        entry['docno'] = docno
    save_path = default_save_path if save_path is None else save_path
    _write_latent_concepts(results, save_path)
    print(f'Latent concepts saved to {save_path}')
    return results


def _generate_latent_descriptions(self, latent_concepts: Optional[List[Dict]] = None,
                                  text_loader: Optional[pt.Transformer] = None, *,
                                  top_k: int = 10, latent_ids: Optional[List[int]] = None,
                                  top_proportion: Optional[float] = None,
                                  api_key: Optional[str] = None, base_url: Optional[str] = None,
                                  model: str = 'gpt-4.1-mini', max_chars: int = 1500,
                                  batch_size: int = 8, save_path: Optional[str] = None) -> Dict[int, str]:
    """Describe the latents of this index with an LLM, using the documents in which they activate most strongly.

    Args:
        latent_concepts: the result of :meth:`extract_latent_concepts`. If omitted, the latent
            concepts are extracted automatically from the SAE saved to this index.
        text_loader: a :class:`pyterrier.Transformer` that maps a frame of ``docno`` to a frame
            with a ``text`` column (e.g. ``pt.text.get_text(pt.get_dataset(...))``).
        top_k: the number of most strongly activating documents to show per latent.
        latent_ids: an optional subset of the latents to describe (defaults to all that ever activate).
        top_proportion: an alternative to ``latent_ids``: only describe the most frequently
            activated latents, up to this proportion of all activated latents (in ``(0, 1]``,
            ranked by the number of documents in which each latent activates).
        api_key, base_url, model: select the OpenAI-compatible LLM endpoint (if omitted, the
            ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` environment variables are used).
        max_chars: the maximum number of characters of each example text to show.
        batch_size: the number of latents to describe per API call.
        save_path: if given, the interpretations are also written as JSONL.

    Returns:
        A dict of ``{latent_id: interpretation}``.

    Requires the ``openai`` package.
    """
    self._assert_built()
    if text_loader is None:
        raise ValueError('text_loader is required: a pyterrier transformer that adds a "text" column to a frame of docnos')
    if latent_concepts is None:
        latent_concepts = self.extract_latent_concepts()
    docnos, _ = self.payload(return_dvecs=False)

    def get_texts(docids):
        docids = sorted(set(int(d) for d in docids))
        df = text_loader(pd.DataFrame({'docid': docids, 'docno': docnos.fwd[np.asarray(docids)]}))
        by_docno = dict(zip(df['docno'], df['text']))
        res = {}
        for d in docids:
            docno = docnos.fwd[d]
            text = by_docno.get(docno)
            if text is None:
                print(f'warning: no text found for docno {docno!r} (docid {d})')
                text = ''
            res[d] = text
        return res

    return generate_descriptions(
        latent_concepts, get_texts,
        top_k=top_k, latent_ids=latent_ids, top_proportion=top_proportion,
        api_key=api_key, base_url=base_url, model=model,
        max_chars=max_chars, batch_size=batch_size, save_path=save_path,
        verbose=self.verbose,
    )


def _latent_describer(self, sae: Optional[SAE_bias_pre] = None, descriptions: Optional[Dict[int, str]] = None, *,
                      threshold: Optional[float] = None, num_latents: int = 5, only_described: bool = True,
                      batch_size: int = 1024, device: Optional[Union[str, t.device]] = None,
                      vec_col: Optional[str] = None) -> LatentDescriber:
    """Return a transformer that annotates query or document vectors with their most active latent explanations.

    The returned :class:`~pyterrier_dr.flex.sae_explainer.LatentDescriber` expects a frame with
    a ``query_vec`` (or ``doc_vec``) column, and adds ``latent_ids``, ``latent_weights`` and
    ``latent_descs`` columns listing the most strongly active SAE latents.

    Args:
        sae: a trained SAE model (e.g. as returned by :meth:`train_sae`). If omitted, the SAE
            previously saved to this index is loaded automatically (see :meth:`load_sae`).
        descriptions: a dict of ``{latent_id: description}`` (e.g. as returned by
            :meth:`generate_latent_descriptions`).
        threshold: the activation threshold; latents whose activation does not exceed it are
            ignored. Defaults to the ``act_thresh`` saved by :meth:`train_sae`.
        num_latents: the maximum number of latents to annotate per row.
        only_described: if ``True``, only latents with an entry in ``descriptions`` are considered.
        batch_size: the number of vectors to encode at once.
        device: the device to encode on.
        vec_col: the name of the vector column (defaults to detecting ``query_vec`` or ``doc_vec``).

    Returns:
        :class:`~pyterrier_dr.flex.sae_explainer.LatentDescriber`
    """
    self._assert_built()
    if descriptions is None:
        raise ValueError('descriptions is required: a dict of latent_id -> description, e.g. from FlexIndex.generate_latent_descriptions()')
    if sae is None or threshold is None:
        loaded_sae, loaded_thresh = self.load_sae(device=device)
        if sae is None:
            sae = loaded_sae
        if threshold is None:
            threshold = loaded_thresh
    return LatentDescriber(
        sae, descriptions,
        threshold=threshold, num_latents=num_latents, only_described=only_described,
        batch_size=batch_size, device=device, vec_col=vec_col,
    )


FlexIndex._assert_built = _assert_built
FlexIndex._update_meta = _update_meta
FlexIndex.sae_config = _sae_meta
FlexIndex.train_sae = _train_sae
FlexIndex.save_sae = _save_sae
FlexIndex.load_sae = _load_sae
FlexIndex.extract_latent_concepts = _extract_latent_concepts
FlexIndex.generate_latent_descriptions = _generate_latent_descriptions
FlexIndex.latent_describer = _latent_describer
