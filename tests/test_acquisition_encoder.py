"""Scientific provenance and checkpoint tests; all data/models are temporary fakes.

No downloads or real model inference. Import through src.encode_v2 after installation.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from src import encode_v2 as enc
from src import acquire_confirmation as acquisition
from src.acquire_confirmation import deduplicate


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def record(identifier, year=2020, title=None, abstract=None, doi=''):
    title = title or 'Title ' + identifier
    abstract = abstract or 'Abstract ' + identifier
    return dict(id=identifier, year=year, source='ACL', title=title,
                abstract=abstract, doi=doi, topic=7, title_sha256=digest_text(title),
                abstract_sha256=digest_text(abstract))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_records(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class TemporaryRoot(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_model(self):
        folder = self.root / 'data/model'
        folder.mkdir(parents=True)
        contents = {'pytorch_model.bin': b'fake pinned weights',
                    'config.json': b'{}', 'tokenizer_config.json': b'{}',
                    'special_tokens_map.json': b'{}', 'vocab.txt': b'[SEP]\n'}
        for name, data in contents.items():
            (folder / name).write_bytes(data)
        manifest = {'model': enc.MODEL, 'revision': enc.REV,
                    'files': {n: enc.sha(folder / n) for n in contents}}
        write_json(self.root / 'data/manifests/model_files.json', manifest)
        (self.root / 'requirements.lock').write_text('numpy==test-version\n')
        return manifest

    def make_reuse(self, rows=None):
        rows = rows or [record('a'), record('b')]
        path = self.root / 'data/processed/records.jsonl'
        write_records(path, rows)
        folder = self.root / 'data/embeddings'
        folder.mkdir(parents=True)
        arrays = {}
        for mode in ['full', 'title']:
            values = np.zeros((len(rows), 768), dtype=np.float32)
            for i in range(len(rows)):
                values[i, i + (0 if mode == 'full' else 8)] = 1
            np.save(folder / (mode + '.npy'), values)
            arrays[mode] = values
        fingerprint = {'data_sha256': enc.sha(path), 'model': enc.MODEL,
                       'revision': enc.REV, 'pooling': 'last_hidden_state_CLS',
                       'normalize': 'L2', 'truncation': 512,
                       'input': 'title + [SEP] + abstract; title-only title + [SEP]',
                       'dtype': 'float32', 'device': 'cpu', 'batch': 8,
                       'chunk': 64}
        manifest = {'status': 'complete', 'fingerprint': fingerprint,
                    'output_sha256': {m: enc.sha(folder / (m + '.npy'))
                                      for m in arrays}}
        write_json(folder / 'encoding_manifest.json', manifest)
        return rows, manifest, arrays

    def save_reuse_manifest(self, manifest):
        write_json(self.root / 'data/embeddings/encoding_manifest.json', manifest)


class ModelProvenanceTests(TemporaryRoot):
    def test_pinned_files_and_lock_are_accepted(self):
        expected = self.make_model()
        with mock.patch.object(enc.importlib.metadata, 'version', return_value='test-version'):
            self.assertEqual(enc.verify_model(self.root), expected)

    def test_modified_pinned_weight_rejected(self):
        self.make_model()
        (self.root / 'data/model/pytorch_model.bin').write_bytes(b'changed')
        with self.assertRaises((ValueError, AssertionError)):
            enc.verify_model(self.root)

    def test_unlisted_preferred_weights_rejected(self):
        self.make_model()
        (self.root / 'data/model/model.safetensors').write_bytes(b'unverified preferred weights')
        with mock.patch.object(enc.importlib.metadata, 'version', return_value='test-version'):
            with self.assertRaises((ValueError, AssertionError)):
                enc.verify_model(self.root)

    def test_unlisted_tokenizer_json_rejected(self):
        self.make_model()
        (self.root / 'data/model/tokenizer.json').write_text('{}')
        with mock.patch.object(enc.importlib.metadata, 'version', return_value='test-version'):
            with self.assertRaises((ValueError, AssertionError)):
                enc.verify_model(self.root)

    def test_weight_omitted_from_manifest_rejected(self):
        manifest = self.make_model()
        del manifest['files']['pytorch_model.bin']
        write_json(self.root / 'data/manifests/model_files.json', manifest)
        with mock.patch.object(enc.importlib.metadata, 'version', return_value='test-version'):
            with self.assertRaises((ValueError, AssertionError)):
                enc.verify_model(self.root)

    def test_dependency_version_drift_rejected(self):
        self.make_model()
        with mock.patch.object(enc.importlib.metadata, 'version', return_value='wrong-version'):
            with self.assertRaises((ValueError, AssertionError)):
                enc.verify_model(self.root)


class ReuseProvenanceTests(TemporaryRoot):
    def test_reordered_ids_reuse_correct_source_rows(self):
        rows, _, arrays = self.make_reuse()
        mapping, legacy = enc.reuse_arrays(self.root, list(reversed(rows)))
        self.assertEqual(mapping, {0: 1, 1: 0})
        np.testing.assert_array_equal(legacy['arrays']['full'][mapping[0]], arrays['full'][1])

    def test_changed_text_with_fresh_hash_is_not_reused(self):
        rows, _, _ = self.make_reuse()
        changed = dict(rows[0], title='Changed title', title_sha256=digest_text('Changed title'))
        mapping, _ = enc.reuse_arrays(self.root, [changed, rows[1]])
        self.assertEqual(mapping, {1: 1})

    def test_stale_destination_text_hash_rejected(self):
        rows, _, _ = self.make_reuse()
        changed = dict(rows[0], title='Changed title with stale stored hash')
        with self.assertRaises((ValueError, AssertionError)):
            enc.reuse_arrays(self.root, [changed])

    def test_source_file_mutation_rejected(self):
        rows, _, _ = self.make_reuse()
        path = self.root / 'data/processed/records.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaises((ValueError, AssertionError)):
            enc.reuse_arrays(self.root, rows)

    def test_array_byte_mutation_rejected(self):
        rows, _, _ = self.make_reuse()
        path = self.root / 'data/embeddings/full.npy'
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaises((ValueError, AssertionError)):
            enc.reuse_arrays(self.root, rows)

    def test_changed_semantic_encoding_settings_rejected(self):
        rows, manifest, _ = self.make_reuse()
        changes = {'pooling': 'mean', 'normalize': 'none', 'truncation': 128,
                   'input': 'abstract only', 'dtype': 'float16', 'device': 'mps'}
        for name, value in changes.items():
            with self.subTest(setting=name):
                changed = json.loads(json.dumps(manifest))
                changed['fingerprint'][name] = value
                self.save_reuse_manifest(changed)
                with self.assertRaises((ValueError, AssertionError)):
                    enc.reuse_arrays(self.root, rows)

    def test_malformed_array_shape_rejected_even_if_digest_matches(self):
        rows, manifest, _ = self.make_reuse()
        path = self.root / 'data/embeddings/full.npy'
        np.save(path, np.ones((len(rows), 3), dtype=np.float32))
        manifest['output_sha256']['full'] = enc.sha(path)
        self.save_reuse_manifest(manifest)
        with self.assertRaises((ValueError, AssertionError)):
            enc.reuse_arrays(self.root, rows)

    def test_nonfinite_cached_vectors_rejected_even_if_digest_matches(self):
        rows, manifest, arrays = self.make_reuse()
        values = arrays['full']; values[0, 0] = np.nan
        path = self.root / 'data/embeddings/full.npy'; np.save(path, values)
        manifest['output_sha256']['full'] = enc.sha(path)
        self.save_reuse_manifest(manifest)
        with self.assertRaises((ValueError, AssertionError)):
            enc.reuse_arrays(self.root, rows)

    def test_duplicate_source_id_rejected(self):
        rows, _, _ = self.make_reuse([record('a'), record('a', title='Different title')])
        with self.assertRaises((ValueError, AssertionError)):
            enc.reuse_arrays(self.root, rows)

    def test_duplicate_destination_id_rejected(self):
        rows, _, _ = self.make_reuse()
        with self.assertRaises((ValueError, AssertionError)):
            enc.reuse_arrays(self.root, [rows[0], rows[0]])

    def test_reuse_disabled_does_not_require_legacy_files(self):
        self.assertEqual(enc.reuse_arrays(self.root, [record('x')], False), ({}, None))


class DeduplicationTests(unittest.TestCase):
    def test_cross_year_duplicate_retains_earliest_chronological_record(self):
        early = record('z-2019', 2019, doi='10.1/shared')
        late = record('a-2021', 2021, doi='10.1/shared')
        kept, dropped = deduplicate([late, early])
        self.assertEqual([r['id'] for r in kept], ['z-2019'])
        self.assertEqual(dropped[0]['reason'], 'duplicate_doi')

    def test_same_year_tie_break_is_lexical_and_order_independent(self):
        a = record('a', title='Same', abstract='Same')
        b = record('b', title='Same', abstract='Same')
        self.assertEqual(deduplicate([b, a]), deduplicate([a, b]))
        self.assertEqual(deduplicate([b, a])[0][0]['id'], 'a')

    def test_same_title_different_abstracts_are_preserved(self):
        rows = [record('a', title='Shared', abstract='First'),
                record('b', title='Shared', abstract='Second')]
        self.assertEqual(len(deduplicate(rows)[0]), 2)

    def test_missing_dois_do_not_collapse_distinct_papers(self):
        kept, dropped = deduplicate([record('a'), record('b')])
        self.assertEqual(len(kept), 2)
        self.assertFalse(dropped)

    def test_excluded_duplicate_maps_to_retained_identity(self):
        a = record('old', 2019, doi='10.1/shared')
        b = record('new', 2021, doi='10.1/shared')
        _, dropped = deduplicate([b, a])
        self.assertEqual(dropped[0].get('canonical_id'), 'old')


class FakeTensor:
    def __init__(self, values): self.values = np.asarray(values)
    def __getitem__(self, key): return FakeTensor(self.values[key])
    def numpy(self): return self.values


class FakeTokenizer:
    sep_token = '[SEP]'
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str): return {'input_ids': list(range(len(texts.split()) + 2))}
        return {'texts': texts}


class FakeModel:
    def eval(self): return self
    def __call__(self, texts):
        result = []
        for text in texts:
            seed = hashlib.sha256(text.encode()).digest()
            vector = np.tile(np.frombuffer(seed, dtype=np.uint8).astype('float32') + 1, 24)
            result.append(vector)
        return types.SimpleNamespace(last_hidden_state=FakeTensor(np.asarray(result)[:, None, :]))


def fake_backends():
    def normalized(value, **kwargs):
        a = value.values
        return FakeTensor(a / np.linalg.norm(a, axis=1, keepdims=True))
    torch = types.SimpleNamespace(__version__='fixture', set_num_threads=lambda _: None,
            manual_seed=lambda _: None, inference_mode=contextlib.nullcontext,
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(normalize=normalized)))
    transformers = types.SimpleNamespace(__version__='fixture',
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()),
            AutoModel=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()))
    return {'torch': torch, 'transformers': transformers}


class CheckpointTests(TemporaryRoot):
    def setUp(self):
        super().setUp()
        self.records = self.root / 'records.jsonl'
        write_records(self.records, [record('paper-' + str(i)) for i in range(65)])
        (self.root / 'requirements.lock').write_text('fixture==1\n')

    def run_encoder(self, output, *flags):
        previous_cwd = os.getcwd()
        try:
            os.chdir(self.root)
            argv = ['encode_v2', '--records', str(self.records), '--output', str(output), '--no-reuse', *flags]
            with mock.patch.object(sys, 'argv', argv), mock.patch.dict(sys.modules, fake_backends()), \
                 mock.patch.object(enc, 'verify_model', return_value={'files': {'fixture': 'hash'}}), \
                 contextlib.redirect_stdout(io.StringIO()):
                enc.main()
        finally:
            os.chdir(previous_cwd)

    def test_duplicate_records_rejected_when_reuse_disabled(self):
        write_records(self.records, [record('same'), record('same')])
        with self.assertRaises((ValueError, AssertionError)):
            self.run_encoder(self.root / 'duplicate-output')

    def test_stale_input_hash_rejected_when_reuse_disabled(self):
        row = record('one')
        row['abstract'] = 'Actual content changed but metadata digest did not'
        write_records(self.records, [row])
        with self.assertRaises((ValueError, AssertionError)):
            self.run_encoder(self.root / 'stale-output')

    def test_partial_resume_matches_uninterrupted_outputs(self):
        resumed = self.root / 'resumed'; fresh = self.root / 'fresh'
        self.run_encoder(resumed, '--max-chunks', '1')
        state = json.loads((resumed / 'encoding_manifest.json').read_text())
        self.assertEqual(state['status'], 'partial')
        self.assertEqual(len(state['chunks']), 1)
        first_hash = enc.sha(resumed / 'full_000000.npy')
        self.run_encoder(resumed, '--resume')
        self.run_encoder(fresh)
        self.assertEqual(enc.sha(resumed / 'full_000000.npy'), first_hash)
        for mode in ['full', 'title']:
            self.assertEqual(enc.sha(resumed / (mode + '.npy')), enc.sha(fresh / (mode + '.npy')))
            self.assertEqual(np.load(resumed / (mode + '.npy')).shape, (65, 768))
        self.assertEqual(json.loads((resumed / 'encoding_manifest.json').read_text())['status'], 'complete')

    def test_resume_rejects_corrupt_recorded_chunk(self):
        output = self.root / 'output'
        self.run_encoder(output, '--max-chunks', '1')
        (output / 'full_000000.npy').write_bytes(b'corrupt')
        with self.assertRaises((ValueError, AssertionError)):
            self.run_encoder(output, '--resume')

    def test_resume_rejects_changed_record_file(self):
        output = self.root / 'output'
        self.run_encoder(output, '--max-chunks', '1')
        self.records.write_text(self.records.read_text().replace('Abstract paper-0', 'Changed abstract'))
        with self.assertRaises((ValueError, AssertionError)):
            self.run_encoder(output, '--resume')

    def test_uncommitted_orphan_chunk_is_recomputed(self):
        output = self.root / 'output'
        self.run_encoder(output, '--max-chunks', '1')
        orphan = output / 'full_000064.npy'; orphan.write_bytes(b'orphan not in manifest')
        (output / 'full_000064.tmp').write_bytes(b'interrupted temporary write')
        self.run_encoder(output, '--resume')
        self.assertEqual(np.load(orphan).shape, (1, 768))
        self.assertFalse((output / 'full_000064.tmp').exists())


class AcquisitionFixtureTests(TemporaryRoot):
    def make_fixture(self):
        history = [record('h-' + str(y), y) for y in [2018, 2019, 2020]]
        old = self.root / 'data/processed/records.jsonl'
        write_records(old, history)
        write_json(self.root / 'data/embeddings/encoding_manifest.json',
                   {'fingerprint': {'data_sha256': acquisition.sha(old.read_bytes())}})
        blobs = []
        for source in ['acl', 'emnlp']:
            volumes = []
            for volume in ['main', 'long', 'short', 'findings', 'demos']:
                paper = (f'<paper id="1"><title>{source} {volume} title</title>'
                         f'<abstract>{source} {volume} abstract</abstract>'
                         '<author><first>A</first><last>B</last></author>'
                         f'<doi>https://doi.org/10.1/{source}-{volume}</doi></paper>')
                volumes.append(f'<volume id="{volume}"><meta><year>2021</year>'
                               f'<booktitle>{volume}</booktitle></meta>{paper}</volume>')
            data = (f'<collection id="2021.{source}">' + ''.join(volumes) + '</collection>').encode()
            path = self.root / 'data/raw' / f'2021.{source}.xml'
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            blobs.append({'path': f'data/xml/2021.{source}.xml', 'sha': blob})
        write_json(self.root / 'data/manifests/source_tree.json', {'tree': blobs})
        return history

    def run_acquisition(self):
        with mock.patch.object(sys, 'argv', ['acquire_confirmation', '--root', str(self.root)]), \
             mock.patch.object(acquisition, 'fetch', side_effect=lambda url, path: path.read_bytes()), \
             contextlib.redirect_stdout(io.StringIO()):
            acquisition.main()

    def test_main_long_short_mapping_and_excluded_volume_ledger(self):
        self.make_fixture()
        # Tiny fixture intentionally fails sample-size feasibility after writing its audit.
        with self.assertRaisesRegex(SystemExit, 'quality criteria'):
            self.run_acquisition()
        out = self.root / 'data/confirmation_v1'
        rows = [json.loads(s) for s in (out / 'records.jsonl').read_text().splitlines()]
        actual = {r['id']: r['source'] for r in rows if r['year'] == 2021}
        expected = {f'2021.{source}-{volume}.1': source.upper()
                    for source in ['acl', 'emnlp'] for volume in ['main', 'long', 'short']}
        self.assertEqual(actual, expected)
        self.assertTrue(all(r['year'] < 2021 for r in rows if r['id'].startswith('h-')))
        ledger = [json.loads(s) for s in (out / 'exclusions.jsonl').read_text().splitlines()]
        excluded = [r for r in ledger if not r['selected_main']]
        self.assertEqual(len(excluded), 4)
        self.assertTrue(all('outside_main_venue_frame' in r['reasons'] for r in excluded))
        manifest = json.loads((out / 'cohort_manifest.json').read_text())
        self.assertEqual(manifest['focal_count'], 6)
        self.assertEqual(manifest['history_count'], 3)
        self.assertFalse(manifest['quality_pass'])
        self.assertTrue(all(len(x['git_blob_sha1']) == 40 for x in manifest['metadata_files']))

    def test_cached_xml_mutation_rejected_against_frozen_tree(self):
        self.make_fixture()
        path = self.root / 'data/raw/2021.acl.xml'
        path.write_bytes(path.read_bytes().replace(b'acl main title', b'changed title'))
        with self.assertRaisesRegex((AssertionError, ValueError), 'source tree'):
            self.run_acquisition()

    def test_prior_records_mutation_rejected(self):
        self.make_fixture()
        path = self.root / 'data/processed/records.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex((AssertionError, ValueError), 'Original record file changed'):
            self.run_acquisition()


if __name__ == '__main__':
    unittest.main()
