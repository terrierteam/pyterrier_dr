# pyterrier_dr (Dense Retrieval for PyTerrier)

This provides various Dense Retrieval functionality for [PyTerrier](https://github.com/terrier-org/pyterrier).

## Installation

This repository can be installed using pip.
  
```bash
pip install pyterrier-dr
```

If you want the latest version of `pyterrier_dr`, you can install direct from the Github repo:

```bash
pip install --upgrade git+https://github.com/terrierteam/pyterrier_dr.git
```

if you want to use the BGE-M3 encoder with `pyterrier_dr`, you can install the package with the `bgem3` dependency:

```bash
pip install pyterrier-dr[bgem3]
```

---
You'll also need to install FAISS.

On Colab:

```bash
!pip install faiss-cpu 
```

On Anaconda:

```bash
# CPU-only version
conda install -c pytorch faiss-cpu
# GPU(+CPU) version
conda install -c pytorch faiss-gpu
```

You can then import the package and PyTerrier in Python:

```python
import pyterrier as pt
import pyterrier_dr
```

## Built-in Models

| Model | `.query_encoder()` | `.doc_encoder()` | `.scorer()` |
|-------|:---------------:|:-------------:|:--------:|
| [`TctColBert`](https://arxiv.org/abs/2010.11386) | ✅ | ✅ | ✅ |
| [`TasB`](https://arxiv.org/abs/2104.06967) | ✅ | ✅ | ✅ |
| [`Ance`](https://arxiv.org/abs/2007.00808) | ✅ | ✅ | ✅ |
| [`Query2Query`](https://neeva.com/blog/state-of-the-art-query2query-similarity) | ✅ | | |
| [`BGE-M3`](https://arxiv.org/abs/2402.03216) | ✅ | ✅ | ✅|

## Inference

Bi-encoder models are represented as PyTerrier transformers. For instance,
to load up a TCT-ColBERT model,

```python
model = pyterrier_dr.TctColBert()
# Loads castorini/tct_colbert-msmarco by default.
# You can load up other versions by specifying the huggingface model ID, e.g.,
model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
```

Once you have a bi-encoder transformer, you can use it encode queries, encode documents, or perform on-they-fly scoring, depending
on the input.

```python
# Compute query vectors
model([
  {'qid': '0', 'query': 'Hello Terrier'},
  {'qid': '1', 'query': 'find me some documents'},
])
# qid                   query                                          query_vec
#   0           Hello Terrier  [-0.044920705, 0.08312888, 0.26291823, -0.0690...
#   1  find me some documents  [0.09036196, 0.19262837, 0.13174239, 0.0649483...

# Compute document vectors
model([
  {'docno': '0', 'text': 'The Five Find-Outers and Dog, also known as The Five Find-Outers, is a series of children\'s mystery books written by Enid Blyton.'},
  {'docno': '1', 'text': 'City is a 1952 science fiction fix-up novel by American writer Clifford D. Simak.'},
])
# docno                                               text                                            doc_vec
#     0  The Five Find-Outers and Dog, also known as Th...  [-0.13535342, 0.16328977, 0.16885889, -0.08592...
#     1  City is a 1952 science fiction fix-up novel by...  [-0.06430543, 0.1267311, 0.13813286, 0.0954021...

# Compute on-they-fly scores
model([
  {'qid': '0', 'query': 'Hello Terrier', 'docno': '0', 'text': 'The Five Find-Outers and Dog, also known as The Five Find-Outers, is a series of children\'s mystery books written by Enid Blyton.'},
  {'qid': '0', 'query': 'Hello Terrier', 'docno': '1', 'text': 'City is a 1952 science fiction fix-up novel by American writer Clifford D. Simak.'},
])
# qid          query docno                                               text      score  rank
#   0  Hello Terrier     0  The Five Find-Outers and Dog, also known as Th...  66.522240     0
#   0  Hello Terrier     1  City is a 1952 science fiction fix-up novel by...  64.964241     1
```

Of course you can also use the model within a larger pipeline. For instance, if you want to re-reank BM25 results using the model,
or split a long documents into passages for later indexing:

```python
# Retrieval pipeline
bm25 = pt.TerrierRetrieve.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25')
retr_pipeline = bm25 >> pt.text.get_text(pt.get_dataset('irds:msmarco-passage'), 'text') >> model
retr_pipeline.search('Hello Terrier')
# qid    docid    docno      score          query                                               text  rank
#   1  1899117  1899117  68.693260  Hello Terrier  The key word is Terrier! Do your homework, I'd...     0
#   1  5679466  5679466  68.605782  Hello Terrier  Introduction. The Biewer Terrier, also known a...     1
#   1  3971237  3971237  68.582764  Hello Terrier  Norwich Terrier. The spirited Norwich is one o...     2
# ...

# Indexing pipeline: split long documents into passages of length 50 (stride 25)
idx_pipeline = pt.text.sliding('text', prepend_title=False, length=50, stride=25) >> model
idx_pipeline([
  {'docno': '0', 'text': "The Five Find-Outers and Dog, also known as The Five Find-Outers, is a series of children's mystery books written by Enid Blyton. The first was published in 1943 and the last in 1961. Set in the fictitious village of Peterswood based on Bourne End, close to Marlow, Buckinghamshire, the children Fatty (Frederick Trotteville), who is the leader of the team, Larry (Laurence Daykin), Pip (Philip Hilton), Daisy (Margaret Daykin), Bets (Elizabeth Hilton) and Buster, Fatty's dog, encounter a mystery almost every school holiday, always solving the puzzle before Mr Goon, the unpleasant village policeman, much to his annoyance."},
])
# docno                                               text                                            doc_vec
#  0%p0  The Five Find-Outers and Dog, also known as Th...  [-0.2607395, 0.21450453, 0.25845605, -0.190567...
#  0%p1  published in 1943 and the last in 1961. Set in...  [-0.4286567, 0.2093819, 0.37688383, -0.2590821...
```

## FLEX Index

A FLexible EXecution (FLEX) Index is a dense index format that allows for a variety of retrieval implementations (NumPy,
FAISS, etc.) and algorithms (exhaustive, HNSW, etc.) to be tested. In many cases, the same vector storage can be used across
implementations and algorithms, saving considerably on disk space.

You can use it as part of an indexing pipeline that includes a model to encode documents:

```python
index = pyterrier_dr.FlexIndex('myindex.flex')
idx_pipeline = model >> index
idx_pipeline.index([
  {'docno': '0', 'text': 'The Five Find-Outers and Dog, also known as The Five Find-Outers, is a series of children\'s mystery books written by Enid Blyton.'},
  {'docno': '1', 'text': 'City is a 1952 science fiction fix-up novel by American writer Clifford D. Simak.'},
])
# Creates an index in myindex.flex:
# $ ls myindex.flex/
# docnos.npids  pt_meta.json  vecs.f4
```

Normally you'll run this over a standard corpus. You can use those provided by [ir_datasets](https://ir-datasets.com/):

```python
index = pyterrier_dr.FlexIndex('antique.flex')
idx_pipeline = model >> index
idx_pipeline.index(pt.get_dataset('irds:antique').get_corpus_iter())
```

## Retrieval

Once built, you can use an index object in a retrieval pipeline too. Be sure to include the model in your pipeline to
encode the query text first!

```python
retr_pipeline = model >> index.np_retriever()
retr_pipeline.search('Hello Terrier')
# qid          query       docno      score  rank
#   1  Hello Terrier   3771188_6  68.791359     0
#   1  Hello Terrier    723025_2  68.791359     1
#   1  Hello Terrier   1969155_1  68.357742     2
# ...
```

The above performs an exhaustive (exact) search using numpy. You
can also use other retrievers from a `FlexIndex`:

```python
retr_pipeline = model >> index.torch_retriever()
retr_pipeline.search('Hello Terrier')
# qid          query       docno      score  rank
#   1  Hello Terrier    723025_2  68.774750     0
#   1  Hello Terrier   3771188_6  68.774750     1
#   1  Hello Terrier   1969155_1  68.340683     2
retr_pipeline = model >> index.faiss_hnsw_retriever()
# ...
```

## BGE-M3 Encoder

`pyterrier_dr` also supports using BGE-M3 for indexing and retrieval with the following encoders:
  
  1. `query_encoder()`: Encodes queries into single-vector representations only.
  2. `doc_encoder()`: Encodes documents into single-vector representations only.
  3. `query_multi_encoder()`: Allows user to encode queries in dense, sparse or multi-vector representations.
  4. `doc_multi_encoder()`: Allows user to encode documents in dense, sparse or multi-vector representations.

What encodings are returned by both `query_multi_encoder()` and `doc_multi_encoder()` can be controlled by the `return_dense`, `return_sparse` and `return_colbert_vecs` parameters. By default, all three are set to `True`.

### Dependencies

The BGE-M3 Encoder requires the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) library. You can install it as part of the `bgem3` dependency of `pyterrier_dr` (see Installation section).

### Indexing

```python
factory = BGEM3(batch_size=32, max_length=1024, verbose=True)
encoder = factory.doc_encoder()

index = FlexIndex(f"mmarco/v2/fr_bgem3", verbose=True)
indexing_pipeline = encoder >> index

indexing_pipeline.index(pt.get_dataset(f"irds:mmarco/v2/fr").get_corpus_iter())
```

### Retrieval

```python
    factory = BGEM3(batch_size=32, max_length=1024)
    encoder = factory.query_encoder()

    index = FlexIndex(f"mmarco/v2/fr_bgem3", verbose=True)

    pipeline = encoder >> idx.np_retriever()
```

## References

 - PyTerrier: PyTerrier: Declarative Experimentation in Python from BM25 to Dense Retrieval (Macdonald et al, CIKM 2021)
 - FAISS: Billion-Scale Similarity Search with GPUs (Johnson et al., 2017)
 - TCT-ColBERT: In-Batch Negatives for Knowledge Distillation with Tightly-Coupled Teachers for Dense Retrieval (Lin et al., RepL4NLP 2021)

## Credits

Contributors to this repository:

 - Sean MacAvaney, University of Glasgow
 - Xiao Wang, University of Glasgow
