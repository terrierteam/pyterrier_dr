import json
import os
import shutil
import tempfile
import types
import unittest
from collections import Counter
from unittest import mock

import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_dr
from pyterrier_dr import FlexIndex


class _FakeTextLoader(pt.Transformer):
    """Stands in for pt.text.get_text: maps a frame of docnos to docno/text."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = 0

    def transform(self, inp):
        self.calls += 1
        return pd.DataFrame({
            'docno': inp['docno'].values,
            'text': [self.texts[d] for d in inp['docno']],
        })


def _fake_openai(api_key=None, base_url=None):
    """Stands in for openai.OpenAI: returns canned [interpretation N]: lines."""
    class _Completions:
        def create(self, model, messages):
            n = len(messages) - 1
            content = '\n'.join(f'[interpretation {i}]: fake description {i}' for i in range(n))
            resp = types.SimpleNamespace()
            resp.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
            return resp

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client()


class TestSaeExplainer(unittest.TestCase):

    def _generate_data(self, count=500, dim=64):
        rng = np.random.RandomState(42)
        dataset = []
        for i in range(count):
            v = rng.rand(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            dataset.append({'docno': str(i), 'doc_vec': v})
        return dataset

    def _build_index(self, destdir, count=500, dim=64):
        index = FlexIndex(destdir + '/index', verbose=False)
        dataset = self._generate_data(count=count, dim=dim)
        index.index(dataset)
        return index, dataset

    def _train(self, index, **kwargs):
        kwargs.setdefault('hidden_mult', 4)
        kwargs.setdefault('k', 8)
        kwargs.setdefault('batch_size', 64)
        kwargs.setdefault('epochs', 2)
        kwargs.setdefault('device', 'cpu')
        return index.train_sae(**kwargs)

    def test_train_sae_saves_checkpoint_and_meta(self):
        with tempfile.TemporaryDirectory() as destdir:
            index, _ = self._build_index(destdir)
            sae, act_thresh, nmse = self._train(index)
            self.assertTrue((index.index_path / 'sae.pt').exists())
            meta = index.sae_config()
            self.assertEqual(meta['input_dim'], 64)
            self.assertEqual(meta['sparse_dim'], 64 * 4)
            self.assertEqual(meta['k'], 8)
            self.assertAlmostEqual(meta['act_thresh'], act_thresh)
            self.assertAlmostEqual(meta['nmse'], nmse)
            self.assertEqual(meta['checkpoint'], 'sae.pt')

    def test_train_sae_no_save(self):
        with tempfile.TemporaryDirectory() as destdir:
            index, _ = self._build_index(destdir)
            sae, act_thresh, nmse = self._train(index, save=False)
            self.assertFalse((index.index_path / 'sae.pt').exists())
            self.assertIsNone(index.sae_config())
            with self.assertRaises(RuntimeError):
                index.load_sae()
            # an in-memory model can still be used
            results = index.extract_latent_concepts(sae, act_thresh, device='cpu')
            self.assertEqual(len(results), 500)

    def test_save_load_roundtrip(self):
        import torch
        with tempfile.TemporaryDirectory() as destdir:
            index, _ = self._build_index(destdir)
            sae, act_thresh, _ = self._train(index)
            sae2, thresh2 = index.load_sae(device='cpu')
            self.assertAlmostEqual(thresh2, act_thresh)
            self.assertEqual(set(sae.state_dict()), set(sae2.state_dict()))
            for name, a in sae.state_dict().items():
                np.testing.assert_allclose(a.cpu().numpy(), sae2.state_dict()[name].cpu().numpy(), rtol=0, atol=0)
            # the model is cached
            sae3, _ = index.load_sae(device='cpu')
            self.assertIs(sae3, sae2)

    def test_extract_latent_concepts(self):
        with tempfile.TemporaryDirectory() as destdir:
            index, _ = self._build_index(destdir)
            sae, act_thresh, _ = self._train(index)
            results = index.extract_latent_concepts(device='cpu') # auto-loads the saved SAE
            self.assertEqual(len(results), 500)
            for r in results:
                self.assertEqual(r['docno'], str(r['docid']))
                self.assertEqual(len(r['ids']), len(r['weight']))
                self.assertTrue(all(w > act_thresh for w in r['weight']))
                self.assertTrue(all(0 <= i < 64 * 4 for i in r['ids']))
            # most documents should have at least one active latent
            self.assertGreater(sum(1 for r in results if r['ids']), 0.9 * len(results))

    def test_extract_latent_concepts_save_path(self):
        with tempfile.TemporaryDirectory() as destdir:
            index, _ = self._build_index(destdir)
            self._train(index)
            save_path = destdir + '/concepts.jsonl'
            results = index.extract_latent_concepts(device='cpu', save_path=save_path)
            with open(save_path, 'rt') as fin:
                lines = [json.loads(line) for line in fin]
            self.assertEqual(len(lines), len(results))
            self.assertEqual(lines[0]['docno'], results[0]['docno'])

    def test_latent_describer(self):
        with tempfile.TemporaryDirectory() as destdir:
            index, dataset = self._build_index(destdir)
            sae, act_thresh, _ = self._train(index)
            descriptions = {0: 'energy', 100: 'policy'}

            with self.subTest('query_vec'):
                describer = index.latent_describer(descriptions=descriptions, threshold=0.0, only_described=False, num_latents=3, device='cpu')
                out = describer(pd.DataFrame({
                    'qid': ['1'],
                    'query_vec': [dataset[0]['doc_vec']],
                }))
                self.assertIn('latent_ids', out.columns)
                self.assertIn('latent_weights', out.columns)
                self.assertIn('latent_descs', out.columns)
                ids, weights, descs = out['latent_ids'].iloc[0], out['latent_weights'].iloc[0], out['latent_descs'].iloc[0]
                self.assertLessEqual(len(ids), 3)
                self.assertGreater(len(ids), 0)
                self.assertEqual(len(ids), len(weights))
                self.assertEqual(len(ids), len(descs))
                self.assertEqual(weights, sorted(weights, reverse=True))
                for i, d in zip(ids, descs):
                    self.assertEqual(d, descriptions.get(i, f'latent_{i}'))

            with self.subTest('doc_vec'):
                out = describer(pd.DataFrame({
                    'docno': ['0'],
                    'doc_vec': [dataset[0]['doc_vec']],
                }))
                self.assertIn('latent_descs', out.columns)
                self.assertEqual(len(out), 1)

            with self.subTest('only_described'):
                describer2 = index.latent_describer(sae, descriptions, threshold=0.0, only_described=True, num_latents=3, device='cpu')
                out = describer2(pd.DataFrame({
                    'qid': ['1'],
                    'query_vec': [dataset[0]['doc_vec']],
                }))
                self.assertTrue(set(out['latent_ids'].iloc[0]) <= set(descriptions))

            with self.subTest('threshold'):
                # a very high threshold should leave no latents
                describer3 = index.latent_describer(sae, descriptions, threshold=1e9, only_described=False, num_latents=3, device='cpu')
                out = describer3(pd.DataFrame({
                    'qid': ['1'],
                    'query_vec': [dataset[0]['doc_vec']],
                }))
                self.assertEqual(out['latent_ids'].iloc[0], [])

            with self.subTest('validation'):
                with self.assertRaises(Exception):
                    describer(pd.DataFrame({'qid': ['1']}))

    def test_generate_latent_descriptions(self):
        with tempfile.TemporaryDirectory() as destdir:
            index, dataset = self._build_index(destdir)
            self._train(index)
            latent_concepts = index.extract_latent_concepts(device='cpu')
            text_loader = _FakeTextLoader({str(i): f'the text of document {i}' for i in range(len(dataset))})
            latent_ids = [i for i, _ in Counter(i for r in latent_concepts for i in r['ids']).most_common(4)]
            save_path = destdir + '/descriptions.jsonl'
            with mock.patch('openai.OpenAI', _fake_openai):
                descriptions = index.generate_latent_descriptions(
                    latent_concepts, text_loader,
                    top_k=3, latent_ids=latent_ids, batch_size=2, save_path=save_path,
                )
            self.assertEqual(len(descriptions), len(latent_ids))
            self.assertTrue(set(descriptions) <= set(latent_ids))
            for interp in descriptions.values():
                self.assertRegex(interp, r'^fake description [0-9]+$')
            self.assertGreater(text_loader.calls, 0)
            with open(save_path, 'rt') as fin:
                lines = [json.loads(line) for line in fin]
            self.assertEqual(len(lines), len(descriptions))
            self.assertTrue(all('id' in line and 'interp' in line for line in lines))

    def test_generate_latent_descriptions_requires_text_loader(self):
        with tempfile.TemporaryDirectory() as destdir:
            index, _ = self._build_index(destdir)
            self._train(index)
            with self.assertRaises(ValueError):
                index.generate_latent_descriptions(text_loader=None)

    def test_generate_latent_descriptions_auto_extract(self):
        with tempfile.TemporaryDirectory() as destdir:
            index, dataset = self._build_index(destdir)
            self._train(index)
            text_loader = _FakeTextLoader({str(i): f'the text of document {i}' for i in range(len(dataset))})
            fired = [i for i, _ in Counter(i for r in index.extract_latent_concepts(device='cpu') for i in r['ids']).most_common(3)]
            with mock.patch('openai.OpenAI', _fake_openai):
                descriptions = index.generate_latent_descriptions(
                    text_loader=text_loader,
                    top_k=2, latent_ids=fired, batch_size=3,
                )
            self.assertEqual(set(descriptions), set(fired))


class TestSaeExplainerBright(unittest.TestCase):
    """Tests against the BRIGHT sustainable living mini dense index.

    The index (pyterrier-tutorial/bright.sustainable.mini.dense) is downloaded from
    HuggingFace and the irds:bright/sustainable-living corpus from ir_datasets; both are
    cached after first use.
    """
    INDEX_ARTIFACT = 'pyterrier-tutorial/bright.sustainable.mini.dense'

    @classmethod
    def setUpClass(cls):
        src = FlexIndex.from_hf(cls.INDEX_ARTIFACT)
        cls._tmpdir = tempfile.TemporaryDirectory()
        shutil.copytree(str(src.index_path), cls._tmpdir.name + '/bright')
        cls.index = FlexIndex(cls._tmpdir.name + '/bright', verbose=False)
        cls.sae, cls.act_thresh, cls.nmse = cls.index.train_sae(
            hidden_mult=4, k=16, batch_size=2048, epochs=1, device='cpu')
        cls.latent_concepts = cls.index.extract_latent_concepts(device='cpu')

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_train_saves_checkpoint_and_meta(self):
        self.assertTrue((self.index.index_path / 'sae.pt').exists())
        meta = self.index.sae_config()
        self.assertEqual(meta['input_dim'], 384)
        self.assertEqual(meta['sparse_dim'], 384 * 4)
        self.assertEqual(meta['k'], 16)
        self.assertAlmostEqual(meta['act_thresh'], self.act_thresh)
        self.assertEqual(meta['checkpoint'], 'sae.pt')

    def test_load_sae(self):
        sae, act_thresh = self.index.load_sae(device='cpu')
        self.assertAlmostEqual(act_thresh, self.act_thresh)

    def test_extract_latent_concepts(self):
        docnos, _ = self.index.payload(return_dvecs=False)
        results = self.latent_concepts
        self.assertEqual(len(results), 60792)
        for r in results[:10]:
            self.assertEqual(r['docno'], docnos.fwd[r['docid']])
            self.assertTrue(all(w > self.act_thresh for w in r['weight']))
        mean_active = sum(len(r['ids']) for r in results) / len(results)
        self.assertGreater(mean_active, 0)

    def test_latent_describer(self):
        descriptions = {0: 'sustainable energy', 1: 'environmental policy'}
        describer = self.index.latent_describer(
            descriptions=descriptions, threshold=0.0, only_described=False, num_latents=5, device='cpu')
        vecs = self.index.np_vecs()
        out = describer(pd.DataFrame({
            'qid': ['1', '2'],
            'query_vec': [vecs[0], vecs[1]],
        }))
        self.assertEqual(len(out), 2)
        for i in range(2):
            ids, weights, descs = out['latent_ids'].iloc[i], out['latent_weights'].iloc[i], out['latent_descs'].iloc[i]
            self.assertLessEqual(len(ids), 5)
            self.assertEqual(len(ids), len(weights))
            self.assertEqual(len(ids), len(descs))
            self.assertEqual(weights, sorted(weights, reverse=True))
            for lid, d in zip(ids, descs):
                self.assertEqual(d, descriptions.get(lid, f'latent_{lid}'))

    def test_generate_latent_descriptions(self):
        text_loader = pt.text.get_text(pt.get_dataset('irds:bright/sustainable-living'))
        latent_ids = [i for i, _ in Counter(i for r in self.latent_concepts for i in r['ids']).most_common(3)]
        save_path = self._tmpdir.name + '/descriptions.jsonl'
        with mock.patch('openai.OpenAI', _fake_openai):
            descriptions = self.index.generate_latent_descriptions(
                self.latent_concepts, text_loader,
                top_k=3, latent_ids=latent_ids, batch_size=2, save_path=save_path,
            )
        self.assertEqual(len(descriptions), len(latent_ids))
        self.assertTrue(all(interp for interp in descriptions.values()))
        self.assertTrue(os.path.exists(save_path))


if __name__ == '__main__':
    unittest.main()
