"""Independent numerical and intervention-contract checks on synthetic cases."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.measure import (build_cache, score, direct_score, unit_rows,
    title_neighbors, local_exposure, summarize_scores)
from src.masks import make_mask, mask_rng, year_matched_control, exact_count
from src.pilot import run, select_cohorts, validate_encoding_manifest, file_sha256


def fixture():
    records = []
    for year in (2018, 2019, 2020):
        for j in range(4):
            records.append({"id": f"{year}-{j}", "year": year,
                "source": "ACL" if j % 2 == 0 else "EMNLP",
                "title": f"Synthetic title {j}", "abstract": f"Authentic fixture text {year} {j}",
                "doi": None, "topic": j % 3})
    rng = np.random.default_rng(1209)
    full, title = rng.normal(size=(12, 7)), rng.normal(size=(12, 7))
    return records, full.astype(np.float32), title.astype(np.float32)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.records, self.full, self.title = fixture()
        self.focal, self.history = np.arange(8, 12), np.arange(8)
        self.cache = build_cache(self.records, self.full, self.title, self.focal, self.history, block_size=2)

    def test_strict_temporal_and_self_exclusion(self):
        c = build_cache(self.records, self.full, self.title, [4, 8], [0, 4, 8, 9])
        np.testing.assert_array_equal(c.eligible, [[True, False, False, False], [True, True, False, False]])
        self.assertTrue(np.isneginf(c.full_full[~c.eligible]).all())
        empty = build_cache(self.records, self.full, self.title, [0], [0, 4])
        self.assertTrue(np.isnan(score(empty, [False, False]))[0])

    def test_fixed_focal_text_and_cached_direct_agreement(self):
        full_before, title_before = self.full.copy(), self.title.copy()
        text_before = json.dumps(self.records, sort_keys=True)
        mask = np.array([True, False, True, False, False, True, False, True])
        for policy in ("mixed", "complete_case", "all_title", "restored"):
            actual = score(self.cache, mask, policy)
            expected = direct_score(self.records, self.full, self.title, self.focal, self.history, mask, policy)
            np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
        np.testing.assert_array_equal(self.full, full_before)
        np.testing.assert_array_equal(self.title, title_before)
        self.assertEqual(json.dumps(self.records, sort_keys=True), text_before)

    def test_no_mask_and_authentic_restoration_equality(self):
        zero, missing = np.zeros(8, bool), np.array([True] * 4 + [False] * 4)
        baseline = score(self.cache, zero, "mixed")
        np.testing.assert_array_equal(baseline, score(self.cache, zero, "complete_case"))
        np.testing.assert_array_equal(baseline, score(self.cache, missing, "restored"))

    def test_all_title_uses_both_sides_and_is_mask_invariant(self):
        actual = score(self.cache, np.zeros(8, bool), "all_title")
        other = score(self.cache, np.ones(8, bool), "all_title")
        expected = direct_score(self.records, self.full, self.title, self.focal, self.history,
                               np.zeros(8, bool), "all_title")
        np.testing.assert_array_equal(actual, other)
        np.testing.assert_allclose(actual, expected, atol=2e-7)
        # This fixture intentionally differs from title-history with full focals.
        self.assertGreater(np.max(np.abs(actual - score(self.cache, np.ones(8, bool), "mixed"))), 0.01)

    def test_whole_record_removal_cannot_decrease_distance(self):
        rng = np.random.default_rng(9)
        baseline = score(self.cache, np.zeros(8, bool), "complete_case")
        for count in range(1, 8):
            for _ in range(10):
                mask = np.zeros(8, bool)
                mask[rng.choice(8, count, replace=False)] = True
                reduced = score(self.cache, mask, "complete_case")
                self.assertTrue((reduced >= baseline - 1e-7).all())
        self.assertTrue(np.isnan(score(self.cache, np.ones(8, bool), "complete_case")).all())

    def test_title_substitution_need_not_inflate_novelty(self):
        records = [{"id": "history", "year": 2019}, {"id": "focal", "year": 2020}]
        full = np.array([[0., 1.], [1., 0.]])
        title = np.array([[1., 0.], [1., 0.]])
        c = build_cache(records, full, title, [1], [0])
        self.assertLess(score(c, [True])[0], score(c, [False])[0])

    def test_title_diagnostic_never_reads_full_embeddings(self):
        changed = self.full[::-1].copy()
        other = build_cache(self.records, changed, self.title, self.focal, self.history)
        neighbors = title_neighbors(self.cache, 3)
        other_neighbors = title_neighbors(other, 3)
        for a, b in zip(neighbors, other_neighbors):
            np.testing.assert_array_equal(a, b)
        mask = np.array([True, False, False, False, True, False, True, False])
        a = local_exposure(self.cache, mask, neighbors)
        b = local_exposure(other, mask, other_neighbors)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a[0], np.repeat(3 / 8, 4))

    def test_invalid_embeddings_and_duplicate_ids_rejected(self):
        for invalid in (np.zeros((2, 3)), np.array([[np.nan, 1]]), np.array([[np.inf, 1]])):
            with self.assertRaises(ValueError):
                unit_rows(invalid)
        records = [dict(r) for r in self.records]
        records[1]["id"] = records[0]["id"]
        with self.assertRaises(ValueError):
            build_cache(records, self.full, self.title, self.focal, self.history)

    def test_distortion_and_top_decile_denominator(self):
        records = [{"id": f"x{i:02}", "year": 2020, "source": "ACL" if i < 10 else "EMNLP"} for i in range(20)]
        baseline = np.arange(20, dtype=float)
        changed = baseline.copy()
        changed[0], changed[19] = changed[19], changed[0]
        result = summarize_scores(records, np.arange(20), baseline, changed)
        self.assertAlmostEqual(result["mean_baseline_top_decile_replacement_fraction"], 0.5)
        self.assertAlmostEqual(result["mean_absolute_group_distortion"], 3.8)
        self.assertAlmostEqual(result["mean_score_drift"], 0)


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.history = fixture()[0][:8]

    def test_exact_rounding_and_deterministic_masks(self):
        self.assertEqual(exact_count(5, 0.5), 3)
        for mechanism in ("random", "source", "topic"):
            for rate in (0, 0.1, 0.25, 0.5, 1):
                first = make_mask(self.history, rate, mechanism, mask_rng(42, rate, mechanism, 0))
                second = make_mask(self.history, rate, mechanism, mask_rng(42, rate, mechanism, 0))
                self.assertEqual(int(first.sum()), exact_count(8, rate))
                np.testing.assert_array_equal(first, second)

    def test_control_preserves_every_historical_year_count(self):
        for mechanism in ("source", "topic"):
            for replicate in range(20):
                mask = make_mask(self.history, 0.5, mechanism, mask_rng(42, 0.5, mechanism, replicate))
                control = year_matched_control(self.history, mask, mask_rng(42, 0.5, mechanism, replicate, control=True))
                self.assertEqual(mask.sum(), control.sum())
                for year in (2018, 2019):
                    ix = np.array([r["year"] == year for r in self.history])
                    self.assertEqual(mask[ix].sum(), control[ix].sum())

    def test_lexical_proxy_must_be_preassigned(self):
        history = [dict(r) for r in self.history]
        del history[0]["topic"]
        with self.assertRaises(ValueError):
            make_mask(history, 0.5, "topic", np.random.default_rng(1))


class ResumeTests(unittest.TestCase):
    def test_bounded_interruption_resume_and_fingerprint_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            records, full, title = fixture()
            (p / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            np.save(p / "full.npy", full)
            np.save(p / "title.npy", title)
            config = {"stage": "exploratory_pilot", "seed": 42, "rates": [0.25],
                "masks_per_condition": 2, "history_years": [2018, 2019], "focal_years": [2020],
                "sources": ["ACL", "EMNLP"], "focal_sample_size": 4,
                "source_contrast": ["ACL", "EMNLP"], "mask_preferred_source": "ACL",
                "mask_preferred_topic": 0, "structured_weight": 8.0, "topic_buckets": 8,
                "top_k": 3, "block_size": 2}
            (p / "config.json").write_text(json.dumps(config))
            args = argparse.Namespace(records=str(p / "records.jsonl"), full=str(p / "full.npy"),
                title=str(p / "title.npy"), config=str(p / "config.json"), output=str(p / "resumed"),
                resume=False, dry_run=False, max_masks=1, synthetic_test=True)
            with contextlib.redirect_stdout(io.StringIO()):
                partial = run(args)
                self.assertEqual(partial["status"], "partial")
                self.assertEqual(partial["completed_assignments"], 1)
                first_shard = next((p / "resumed" / "raw").glob("*.json"))
                first_bytes = first_shard.read_bytes()
                first_time = first_shard.stat().st_mtime_ns
                args.resume, args.max_masks = True, None
                complete = run(args)
                self.assertEqual(complete["status"], "complete")
                self.assertEqual(complete["completed_assignments"], 6)
                self.assertEqual(first_shard.read_bytes(), first_bytes)
                self.assertEqual(first_shard.stat().st_mtime_ns, first_time)
                self.assertFalse((p / "resumed" / ".run.lock").exists())
                run(args)  # Verifies completed-run output checksums without redoing work.
                args.output, args.resume = str(p / "clean"), False
                run(args)
                self.assertEqual((p / "resumed" / "mask_results.jsonl").read_bytes(),
                                 (p / "clean" / "mask_results.jsonl").read_bytes())
                self.assertEqual((p / "resumed" / "summary.json").read_bytes(),
                                 (p / "clean" / "summary.json").read_bytes())
                config["seed"] = 43
                (p / "config.json").write_text(json.dumps(config))
                args.resume = True
                with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                    run(args)

    def test_encoding_manifest_guards_record_order_and_output_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            args = argparse.Namespace(full=str(p / "full.npy"), synthetic_test=False)
            hashes = {"records": "correct-record-hash", "full": "correct-full-hash", "title": "correct-title-hash"}
            with self.assertRaisesRegex(ValueError, "manifest required"):
                validate_encoding_manifest(args, hashes)
            manifest = {"status": "complete", "fingerprint": {"data_sha256": hashes["records"],
                "model": "synthetic-fixture", "revision": "test-v1"},
                "output_sha256": {"full": hashes["full"], "title": hashes["title"]}}
            path = p / "encoding_manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(validate_encoding_manifest(args, hashes)["status"], "verified")
            wrong = dict(hashes, records="reordered-record-hash")
            with self.assertRaisesRegex(ValueError, "records hash/order"):
                validate_encoding_manifest(args, wrong)
            wrong = dict(hashes, title="other-model-title-hash")
            with self.assertRaisesRegex(ValueError, "output hash mismatch"):
                validate_encoding_manifest(args, wrong)
            manifest["status"] = "running"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "not complete"):
                validate_encoding_manifest(args, hashes)

    def test_sample_freezes_focals_and_retains_all_history(self):
        records, _, _ = fixture()
        config = {"seed": 42, "history_years": [2018, 2019], "focal_years": [2020],
            "sources": ["ACL", "EMNLP"], "source_contrast": ["ACL", "EMNLP"], "focal_sample_size": 4}
        f1, h1 = select_cohorts(records, config)
        f2, h2 = select_cohorts(records, config)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(h1, h2)
        self.assertEqual(len(h1), 8)
        self.assertEqual(len(f1), 4)


if __name__ == "__main__":
    unittest.main()
