from typing import Union, Iterable, Dict
import shutil
import itertools
import json
from pathlib import Path
import numpy as np
import more_itertools
import pandas as pd
import pyterrier as pt
from npids import Lookup
from enum import Enum
from .. import SimFn
import pyterrier_alpha as pta


class IndexingMode(Enum):
    create = "create"
    overwrite = "overwrite"
    # append???


class FlexIndex(pta.Artifact, pt.Indexer):
    """ Represents a FLexible EXecution (FLEX) Index, which is a dense index format.

    FLEX allows for a variety of retrieval implementations (NumPy, FAISS, etc.) and algorithms (exhaustive, HNSW, etc.)
    to be tested. In most cases, the same vector storage can be used across implementations and algorithms, saving
    considerably on disk space.
    """

    ARTIFACT_TYPE = 'dense_index'
    ARTIFACT_FORMAT = 'flex'


    def __init__(self,
        path: str,
        *,
        sim_fn: Union[SimFn, str] = SimFn.dot,
        verbose: bool = True
    ):
        """
        Args:
            path: The path to the index directory
            sim_fn: The similarity function to use
            verbose: Whether to display verbose output (e.g., progress bars)
        """
        super().__init__(path)
        self.index_path = Path(path)
        self.sim_fn = SimFn(sim_fn)
        self.verbose = verbose
        self._meta = None
        self._docnos = None
        self._dvecs = None
        self._cache = {}

    def payload(self, return_dvecs=True, return_docnos=True):
        if self._meta is None:
            with open(self.index_path/'pt_meta.json', 'rt') as f_meta:
                meta = json.load(f_meta)
                assert meta.get('type') == 'dense_index' and meta['format'] == 'flex'
                self._meta = meta
        res = [self._meta]
        if return_dvecs:
            if self._dvecs is None:
                self._dvecs = np.memmap(self.index_path/'vecs.f4', mode='r', dtype=np.float32, shape=(self._meta['doc_count'], self._meta['vec_size']))
            res.insert(0, self._dvecs)
        if return_docnos:
            if self._docnos is None:
                self._docnos = Lookup(self.index_path/'docnos.npids')
            res.insert(0, self._docnos)
        return res

    def __len__(self):
        meta, = self.payload(return_dvecs=False, return_docnos=False)
        return meta['doc_count']

    def index(self, inp: Iterable[Dict]) -> pta.Artifact:
        """Index the given input data stream to a new index at this location.

        Each record in ``inp`` is expected to be a dictionary containing at least two keys: ``docno`` (a unique document
        identifier) and ``doc_vec`` (a dense vector representation of the document).

        Typically this method will be used in a pipeline of operations, where the input data is first transformed by a
        document encoder to add the ``doc_vec`` values before it is indexed. For example:

        .. code-block:: python
            :caption: Index documents into a :class:`~pyterrier_dr.FlexIndex` using a :class:`~pyterrier_dr.TasB` encoder.
            
            from pyterrier_dr import TasB, FlexIndex    
            encoder = TasB.dot()
            index = FlexIndex('my_index')
            pipeline = encoder >> index
            pipeline.index([
                {'docno': 'doc1', 'text': 'hello'},
                {'docno': 'doc2', 'text': 'world'},
            ])

        Args:
            inp: An iterable of dictionaries to index.

        Returns:
            :class:`pyterrier_alpha.Artifact`: A reference back to this index (``self``).

        Raises:
            RuntimeError: If the index is aready built.
        """
        return self.indexer().index(inp)

    def indexer(self, *, mode: Union[IndexingMode, str] = IndexingMode.create) -> 'FlexIndexer':
        """Return an indexer for this index with the specified options.

        This transformer gives more fine-grained control over the indexing process, allowing you to specify whether
        to create a new index or overwrite an existing one.

        Similar to :meth:`index`, this method will typically be used in a pipeline of operations, where the input data
        is first transformed by a document encoder to add the ``doc_vec`` values before it is indexed. For example:

        .. code-block:: python
            :caption: Oerwrite a :class:`~pyterrier_dr.FlexIndex` using a :class:`~pyterrier_dr.TasB` encoder.
            
            from pyterrier_dr import TasB, FlexIndex    
            encoder = TasB.dot()
            index = FlexIndex('my_index')
            pipeline = encoder >> index.indexer(mode='overwrite')
            pipeline.index([
                {'docno': 'doc1', 'text': 'hello'},
                {'docno': 'doc2', 'text': 'world'},
            ])

        Args:
            mode: The indexing mode to use (``create`` or ``overwrite``).

        Returns:
            :class:`~pyterrier.Indexer`: A new indexer instance.
        """
        return FlexIndexer(self, mode=mode)

    def transform(self, inp):
        with pta.validate.any(inp) as v:
            v.query_frame(extra_columns=['query_vec'], mode='retriever')
            v.result_frame(extra_columns=['query_vec'], mode='scorer')

        if v.mode == 'retriever':
            return self.retriever()(inp)
        if v.mode == 'scorer':
            return self.scorer()(inp)

    def get_corpus_iter(self, start_idx=None, stop_idx=None, verbose=True) -> Iterable[Dict]:
        """Iterate over the documents in the index.

        Args:
            start_idx: The index of the first document to return (or ``None`` to start at the first document).
            stop_idx: The index of the last document to return (or ``None`` to end on the last document).
            verbose: Whether to display a progress bar.

        Yields:
            Dict[str,Any]: A dictionary with keys ``docno`` and ``doc_vec``.
        """
        docnos, dvecs, meta = self.payload()
        docno_iter = iter(docnos)
        if start_idx is not None or stop_idx is not None:
            docno_iter = itertools.islice(docno_iter, start_idx, stop_idx)
            dvecs = dvecs[start_idx:stop_idx]
        it = zip(docno_iter, range(dvecs.shape[0]))
        if self.verbose:
            it = pt.tqdm(it, total=dvecs.shape[0], desc=f'{str(self.index_path)} dvecs')
        for docno, i in it:
            yield {'docno': docno, 'doc_vec': dvecs[i]}

    def _load_docids(self, inp):
        with pta.validate.any(inp) as v:
            v.columns(includes=['docid'], mode='docid')
            v.columns(includes=['docno'], mode='docno')
        if v.mode == 'docid':
            return inp['docid'].values
        docnos, config = self.payload(return_dvecs=False)
        return docnos.inv[inp['docno'].values] # look up docids from docnos

    def built(self) -> bool:
        """Check if the index has been built.

        Returns:
            bool: ``True`` if the index has been built, otherwise ``False``.
        """
        return self.index_path.exists()

    def docnos(self) -> Lookup:
        """Return the document identifier (docno) lookup data structure.

        Returns:
            :class:`npids.Lookup`: The document number lookup.
        """
        docnos, meta = self.payload(return_dvecs=False)
        return docnos


class FlexIndexer(pt.Indexer):
    def __init__(self, index: FlexIndex, mode: Union[IndexingMode, str] = IndexingMode.create):
        self._index = index
        self.mode = IndexingMode(mode)

    def __repr__(self):
        return f'{self._index}.indexer(mode={self.mode!r})'

    def transform(self, inp):
        raise RuntimeError("FlexIndexer cannot be used as a transformer, use .index() instead")

    def index(self, inp):
        if isinstance(inp, pd.DataFrame):
            inp = inp.to_dict(orient="records")
        inp = more_itertools.peekable(inp)
        path = Path(self._index.index_path)
        if path.exists():
            if self.mode == IndexingMode.overwrite:
                shutil.rmtree(path)
            else:
                raise RuntimeError(f'Index already exists at {self._index.index_path}. If you want to delete and re-create an existing index, you can pass index.indexer(mode="overwrite")')
        path.mkdir(parents=True, exist_ok=True)
        vec_size = None
        count = 0
        if self._index.verbose:
            inp = pt.tqdm(inp, desc='indexing', unit='dvec')
        with open(path/'vecs.f4', 'wb') as fout, Lookup.builder(path/'docnos.npids') as docnos:
            for d in inp:
                vec = d['doc_vec']
                if vec_size is None:
                    vec_size = vec.shape[0]
                elif vec_size != vec.shape[0]:
                    raise ValueError(f'Inconsistent vector shapes detected (expected {vec_size} but found {vec.shape[0]})')
                vec = vec.astype(np.float32)
                fout.write(vec.tobytes())
                docnos.add(d['docno'])
                count += 1
        with open(path/'pt_meta.json', 'wt') as f_meta:
            json.dump({
                "type": self._index.ARTIFACT_TYPE,
                "format": self._index.ARTIFACT_FORMAT,
                "vec_size": vec_size,
                "doc_count": count
            }, f_meta)
        return self._index


def _load_dvecs(flex_index, inp):
    dvecs, config = flex_index.payload(return_docnos=False)
    return dvecs[flex_index._load_docids(inp)]
