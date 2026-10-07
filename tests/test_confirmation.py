"""Confirmation-specific contracts, using synthetic data only."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.confirm import (STAGE, CONTROL_FIELDS, create_seed_manifest,
    validate_seed_manifest, validate_config, select_all_cohorts, matched_control,
    confirmation_provenance, run)
from src.pilot import file_sha256, fingerprints
import src.confirm as confirmation_module


def configuration():
    return {"stage": STAGE, "rates": [0.25], "masks_per_condition": 2,
        "history_years": [2018, 2019, 2020], "focal_years": [2021],
        "focal_selection": "all_eligible", "sources": ["ACL", "EMNLP"],
        "source_contrast": ["ACL", "EMNLP"], "mask_preferred_source": "ACL",
        "mask_preferred_topic": 0, "structured_weight": 8,
        "control_matching": {key: list(value) for key, value in CONTROL_FIELDS.items()},
        "topic_buckets": 8, "top_k": 3, "block_size": 2}


def records_fixture():
    records = []
    for year in (2018, 2019, 2020, 2021):
        for source in ("ACL", "EMNLP"):
            for j in range(3):
                records.append({"id": f"{year}-{source}-{j}", "year": year,
                    "source": source, "title": f"Title {j}",
                    "abstract": f"Abstract {year} {source} {j}", "topic": j % 2})
    return records


def prepare_run(directory):
    p = Path(directory)
    records = records_fixture()
    rng = np.random.default_rng(234)
    for mode in ("full", "title"):
        np.save(p / (mode + ".npy"), rng.normal(size=(len(records), 9)).astype(np.float32))
    (p / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    config = configuration()
    (p / "config.json").write_text(json.dumps(config))
    (p / "seeds.json").write_text(json.dumps(create_seed_manifest(config)))
    (p / "protocol.md").write_text("Synthetic protocol fixture; not an empirical freeze.\n")
    (p / "requirements.lock").write_text("numpy==" + np.__version__ + "\n")
    (p / "analysis").mkdir()
    (p / "analysis/confirmatory_readout.py").write_text("# Synthetic bound analysis stub\n")
    (p / "encoding_manifest.json").write_text(json.dumps({"status": "complete",
        "fingerprint": {"schema": 2, "data_sha256": file_sha256(p / "records.jsonl"),
            "model": "synthetic-fixture", "revision": "synthetic-v1",
            "model_files": {"weights.synthetic": "0" * 64},
            "requirements_lock_sha256": file_sha256(p / "requirements.lock")},
        "output_sha256": {name: file_sha256(p / (name + ".npy")) for name in ("full", "title")}}))
    args = argparse.Namespace(records=str(p / "records.jsonl"), full=str(p / "full.npy"),
        title=str(p / "title.npy"), config=str(p / "config.json"),
        seed_manifest=str(p / "seeds.json"), protocol=str(p / "protocol.md"),
        freeze_manifest=str(p / "freeze.json"), output=str(p / "resumed"),
        root=str(p), resume=False, dry_run=False, max_masks=1, synthetic_test=True)
    provenance = fingerprints(args)
    provenance["source_code_sha256"]["confirm.py"] = file_sha256(confirmation_module.__file__)
    freeze = {"status": "frozen", "stage": STAGE,
        "git_commit": "0" * 40, "frozen_utc": "2026-10-04T00:00:00+00:00",
        "originality_gate": "PASS WITH NARROWER CLAIM",
        "sha256": {key: file_sha256(getattr(args, attribute)) for key, attribute in (
            ("records", "records"), ("config", "config"), ("seeds", "seed_manifest"), ("protocol", "protocol"),
            ("full", "full"), ("title", "title"))},
        "source_code_sha256": provenance["source_code_sha256"],
        "analysis_code_sha256": {"analysis/confirmatory_readout.py": file_sha256(p / "analysis/confirmatory_readout.py")}}
    freeze["sha256"].update(encoding_manifest=file_sha256(p / "encoding_manifest.json"),
        requirements_lock=file_sha256(p / "requirements.lock"))
    Path(args.freeze_manifest).write_text(json.dumps(freeze))
    return args


class ConfirmationMatchingTests(unittest.TestCase):
    def test_topic_controls_preserve_every_year_source_cell(self):
        history = records_fixture()[:18]
        mask = np.array([True, False, False, True, True, False] * 3)
        control, cells = matched_control(history, mask, np.random.default_rng(94), ("year", "source"))
        self.assertEqual(len(cells), 6)
        self.assertEqual(int(control.sum()), int(mask.sum()))
        for year in (2018, 2019, 2020):
            for source in ("ACL", "EMNLP"):
                ix = np.array([r["year"] == year and r["source"] == source for r in history])
                self.assertEqual(int(control[ix].sum()), int(mask[ix].sum()))
        repeat, _ = matched_control(history, mask, np.random.default_rng(94), ("year", "source"))
        np.testing.assert_array_equal(control, repeat)

    def test_source_control_matches_year_without_fixing_source(self):
        history = records_fixture()[:18]
        mask = np.array([r["source"] == "ACL" for r in history])
        control, cells = matched_control(history, mask, np.random.default_rng(94), ("year",))
        for year in (2018, 2019, 2020):
            ix = np.array([r["year"] == year for r in history])
            self.assertEqual(int(control[ix].sum()), int(mask[ix].sum()))
        acl = np.array([r["source"] == "ACL" for r in history])
        self.assertNotEqual(int(control[acl].sum()), int(mask[acl].sum()))
        self.assertEqual(len(cells), 3)

    def test_degenerate_cells_are_explicit_and_exact(self):
        history = records_fixture()[:18]
        mask = np.array([r["year"] == 2018 for r in history])
        control, cells = matched_control(history, mask, np.random.default_rng(1), ("year", "source"))
        np.testing.assert_array_equal(mask, control)
        self.assertTrue(all(not c["nondegenerate"] for c in cells))


class ConfirmationSeedAndScopeTests(unittest.TestCase):
    def test_explicit_seeds_unique_deterministic_and_bound_to_config(self):
        config = configuration()
        manifest = create_seed_manifest(config)
        self.assertEqual(manifest, create_seed_manifest(config))
        self.assertEqual(len(validate_seed_manifest(manifest, config)), 6)
        vectors = [tuple(row[key]) for row in manifest["assignments"]
            for key in ("mask_seed", "control_seed") if key in row]
        self.assertEqual(len(vectors), len(set(vectors)))
        altered = copy.deepcopy(manifest)
        altered["assignments"][0]["mask_seed"][0] += 1
        with self.assertRaisesRegex(ValueError, "Explicit seed assignments"):
            validate_seed_manifest(altered, config)
        with self.assertRaises(ValueError):
            create_seed_manifest(config, master_seed=20261004)
        config["masks_per_condition"] = 3
        with self.assertRaises(ValueError):
            validate_seed_manifest(manifest, config)

    def test_all_focal_selection_and_strict_stage_controls(self):
        config = configuration()
        validate_config(config, synthetic_test=True)
        records = records_fixture()
        focal, history = select_all_cohorts(records, config)
        self.assertEqual(set(focal), {i for i, r in enumerate(records) if r["year"] == 2021})
        self.assertEqual(len(focal), 6)
        self.assertEqual(len(history), 18)
        for key, value in (("stage", "exploratory_pilot"), ("focal_selection", "sample"),
                           ("focal_sample_size", 3), ("control_matching", {"source": ["year"], "topic": ["year"]})):
            altered = dict(config, **{key: value})
            with self.assertRaises(ValueError):
                validate_config(altered, synthetic_test=True)
        with self.assertRaisesRegex(ValueError, "exactly 200"):
            validate_config(config, synthetic_test=False)


class ConfirmationProvenanceAndResumeTests(unittest.TestCase):
    def test_all_embedding_lock_and_analysis_bindings_required(self):
        with tempfile.TemporaryDirectory() as directory:
            args = prepare_run(directory)
            original = json.loads(Path(args.freeze_manifest).read_text())
            for key in ("full", "title", "encoding_manifest", "requirements_lock"):
                altered = copy.deepcopy(original)
                del altered["sha256"][key]
                Path(args.freeze_manifest).write_text(json.dumps(altered))
                with self.subTest(missing=key), self.assertRaisesRegex(ValueError, "Frozen artifact hash mismatch"):
                    confirmation_provenance(args)
            altered = copy.deepcopy(original)
            del altered["analysis_code_sha256"]
            Path(args.freeze_manifest).write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "analysis code must be bound"):
                confirmation_provenance(args)

    def test_embedding_manifest_lock_and_analysis_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = prepare_run(directory)
            p = Path(directory)
            for relative in ("full.npy", "title.npy", "requirements.lock", "analysis/confirmatory_readout.py"):
                path = p / relative
                content = path.read_bytes()
                path.write_bytes(content + b"tampered")
                with self.subTest(tampered=relative), self.assertRaises(ValueError):
                    confirmation_provenance(args)
                path.write_bytes(content)
            path = p / "encoding_manifest.json"
            encoder = json.loads(path.read_text())
            original = path.read_text()
            encoder["fingerprint"]["revision"] = "changed"
            path.write_text(json.dumps(encoder))
            with self.assertRaisesRegex(ValueError, "encoding_manifest"):
                confirmation_provenance(args)
            path.write_text(original)
            for key in ("model_files", "requirements_lock_sha256"):
                encoder = json.loads(original)
                del encoder["fingerprint"][key]
                path.write_text(json.dumps(encoder))
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    confirmation_provenance(args)
                path.write_text(original)

    def test_freeze_protocol_and_seed_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = prepare_run(directory)
            provenance = confirmation_provenance(args)
            self.assertEqual(provenance["stage"], STAGE)
            self.assertIn("seeds", provenance["inputs"])
            self.assertIn("freeze_manifest_sha256", provenance)
            freeze = json.loads(Path(args.freeze_manifest).read_text())
            altered = dict(freeze, status="draft")
            Path(args.freeze_manifest).write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "before a frozen"):
                confirmation_provenance(args)
            Path(args.freeze_manifest).write_text(json.dumps(freeze))
            Path(args.protocol).write_text("Unfrozen protocol amendment")
            with self.assertRaisesRegex(ValueError, "protocol"):
                confirmation_provenance(args)

    def test_bounded_resume_clean_equality_and_confirmation_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            args = prepare_run(directory)
            partial = run(args)
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(partial["completed_assignments"], 1)
            shard = next((Path(args.output) / "raw").glob("*.json"))
            original_bytes, original_time = shard.read_bytes(), shard.stat().st_mtime_ns
            args.resume, args.max_masks = True, None
            finished = run(args)
            self.assertEqual(finished["status"], "complete")
            self.assertEqual(finished["scope"]["focal_count"], 6)
            self.assertEqual(finished["stage"], STAGE)
            self.assertEqual(shard.read_bytes(), original_bytes)
            self.assertEqual(shard.stat().st_mtime_ns, original_time)
            summary = json.loads((Path(args.output) / "summary.json").read_text())
            self.assertEqual(summary["stage"], STAGE)
            conditions = {r["condition"] for r in summary["conditions"]}
            self.assertIn("topic_year_source_matched_control", conditions)
            self.assertNotIn("topic_year_matched_control", conditions)
            self.assertTrue(all(set(row["outcomes"]) == {"G", "T", "Q"} for row in summary["paired_primary"]))
            rows = [json.loads(line) for line in (Path(args.output) / "mask_results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(row["stage"], STAGE)
                if row["mechanism"] != "random":
                    self.assertEqual(row["control_matching"], list(CONTROL_FIELDS[row["mechanism"]]))
            clean = Path(directory) / "clean"
            resumed = Path(args.output)
            args.output, args.resume = str(clean), False
            run(args)
            for name in ("baseline.json", "mask_results.jsonl", "summary.json"):
                self.assertEqual((clean / name).read_bytes(), (resumed / name).read_bytes())
            args.resume = True
            self.assertEqual(run(args)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
