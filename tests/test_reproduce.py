"""CLI contracts and portable, one-epoch synthetic-data integration tests."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import reproduce as entry


class ReproductionContracts(unittest.TestCase):
    def args(self, *extra):
        return entry.make_parser().parse_args(
            ["--data-dir", "data", "--output-dir", "outputs", *extra])

    def hash_fixture(self, root):
        manifest = {"source_commit": "synthetic-fixture", "files": {}}
        for system in entry.SYSTEMS:
            path = root / (system + "_data.pt")
            path.write_bytes(("hash-only fixture: " + system).encode())
            manifest["files"][path.name] = {"sha256": entry.file_sha256(path)}
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))
        return ["--data-dir", str(root), "--output-dir", str(root / "out"),
                "--data-manifest", str(path)]

    def test_main_plan_and_published_settings(self):
        jobs = entry.build_plan(self.args())
        self.assertEqual(len(jobs), 120)
        self.assertEqual({j["epochs"] for j in jobs}, {100})
        self.assertEqual({j["system"] for j in jobs}, set(entry.SYSTEMS))
        self.assertEqual({j["method"] for j in jobs}, set(entry.METHODS))
        self.assertTrue(all(j["options"] == {"mu_init": 0.1, "effort_weight": 0.05}
                            for j in jobs))
        self.assertTrue(all(j["evaluation_scale"] is None for j in jobs))
        self.assertEqual(len({j["relative_output"] for j in jobs}), 120)
        for job in jobs:
            self.assertEqual(job["relative_output"],
                f"main/{job['system']}/{job['method']}/seed{job['seed']}")

    def test_additional_suite_plans_and_distinct_outputs(self):
        expected = {
            "ablation": (27, {"NODE-LAC"}),
            "data-efficiency": (81, {"NODE-LAC", "NODE", "SNDE"}),
            "noise": (54, {"NODE-LAC", "NODE"}),
            "long-horizon": (54, {"NODE-LAC", "NODE", "SNDE", "PNODE", "CPNODE", "ConCerNet"}),
        }
        for suite, (count, methods) in expected.items():
            with self.subTest(suite=suite):
                jobs = entry.build_plan(self.args("--suite", suite, "--model-dir", "models"))
                self.assertEqual(len(jobs), count)
                self.assertEqual({j["system"] for j in jobs}, set(entry.SYSTEMS[:3]))
                self.assertEqual({j["method"] for j in jobs}, methods)
                self.assertEqual(len({j["relative_output"] for j in jobs}), count)
                for job in jobs:
                    self.assertEqual(job["relative_output"],
                        f"{suite}/{job['system']}/{job['method']}/{job['variant']}/seed{job['seed']}")
                if suite == "data-efficiency":
                    self.assertEqual({j["fraction"] for j in jobs}, {0.25, 0.5, 0.75})
                    self.assertEqual({j["noise"] for j in jobs}, {0.0})
                elif suite == "noise":
                    self.assertEqual({j["noise"] for j in jobs}, {0.05, 0.1, 0.2})
                    self.assertEqual({j["fraction"] for j in jobs}, {1.0})
                elif suite == "long-horizon":
                    self.assertEqual({j["variant"] for j in jobs}, {"1000"})

    def test_ablation_changes_one_requested_component(self):
        jobs = entry.build_plan(self.args("--suite", "ablation", "--systems", "fitzhugh_nagumo",
                                         "--seeds", "42"))
        flags = {"NoGainNet": "fixed_gain", "NoConstraintLoss": "no_constraint_loss",
                 "NoCorrection": "no_correction"}
        self.assertEqual({j["variant"] for j in jobs}, set(flags))
        for job in jobs:
            self.assertEqual(job["options"],
                {"mu_init": 0.1, "effort_weight": 0.05, flags[job["variant"]]: True})

    def test_residual_pairing_fixed_scales_and_output_paths(self):
        jobs = entry.build_plan(self.args("--suite", "residual"))
        self.assertEqual(len(jobs), 21)
        for system in entry.SYSTEMS:
            for seed in entry.SEEDS:
                matched = [j for j in jobs if j["system"] == system and j["seed"] == seed]
                self.assertEqual(len(matched), 1 if system == "fitzhugh_nagumo" else 2)
                rho = entry.RESIDUAL_SCALES[system][entry.SEEDS.index(seed)]
                self.assertEqual({j["evaluation_scale"] for j in matched}, {rho})
                if system == "fitzhugh_nagumo":
                    self.assertTrue(matched[0]["also_selected_model"])
                    self.assertEqual(matched[0]["variant"], "zero-control")
                else:
                    self.assertEqual({j["lambda_L"] for j in matched},
                                     {0.0, entry.RESIDUAL_WEIGHTS[system]})
                for job in matched:
                    self.assertEqual(job["relative_output"],
                        f"residual/{system}/NODE-LAC/{job['variant']}/seed{seed}")

    def test_residual_single_variant_selection(self):
        for variant, weight in (("zero-control", 0.0), ("regularized", 0.001)):
            with self.subTest(variant=variant):
                jobs = entry.build_plan(self.args("--suite", "residual", "--systems", "lotka_volterra",
                    "--seeds", "123", "--variant", variant))
                self.assertEqual(len(jobs), 1)
                self.assertEqual((jobs[0]["variant"], jobs[0]["lambda_L"]), (variant, weight))

    def test_invalid_plan_arguments(self):
        invalid = (["--epochs", "0"], ["--systems", "shallow_water", "shallow_water"],
                   ["--methods", "NODE", "NODE"], ["--seeds", "42", "42"],
                   ["--suite", "residual", "--methods", "NODE"],
                   ["--suite", "ablation", "--methods", "NODE"],
                   ["--variant", "zero-control"],
                   ["--suite", "noise", "--systems", "franka_robot"],
                   ["--suite", "long-horizon"])
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                entry.build_plan(self.args(*extra))

    def test_long_horizon_dry_run_requires_model_dir_before_data_access(self):
        stderr = io.StringIO()
        with patch.object(entry, "verify_data", side_effect=AssertionError("data accessed")), \
                patch.object(entry, "load_runtime", side_effect=AssertionError("runtime loaded")), \
                contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            entry.main(["--suite", "long-horizon", "--output-dir", "unused", "--dry-run"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--model-dir is required", stderr.getvalue())

    def test_all_dry_runs_verify_hashes_without_loading_runtime_or_writing(self):
        counts = {"main": 120, "residual": 21, "ablation": 27,
                  "data-efficiency": 81, "noise": 54, "long-horizon": 54}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = self.hash_fixture(root)
            for suite, count in counts.items():
                stdout = io.StringIO()
                argv = common + ["--suite", suite, "--model-dir", str(root / "models"), "--dry-run"]
                with self.subTest(suite=suite), \
                        patch.object(entry, "load_runtime", side_effect=AssertionError("runtime loaded")), \
                        contextlib.redirect_stdout(stdout):
                    self.assertEqual(entry.main(argv), 0)
                plan = json.loads(stdout.getvalue())
                self.assertEqual(plan["run_count"], count)
                self.assertEqual(len(plan["runs"]), count)
                self.assertFalse((root / "out").exists())
                self.assertFalse((root / "models").exists())
            args = entry.make_parser().parse_args(common + ["--systems", "fitzhugh_nagumo"])
            entry.build_plan(args)
            data = root / "fitzhugh_nagumo_data.pt"
            data.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                entry.verify_data(args)
            data.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing released dataset"):
                entry.verify_data(args)

    def test_existing_run_is_rejected_before_training_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = self.hash_fixture(root)
            target = root / "out/main/fitzhugh_nagumo/NODE/seed42"
            target.mkdir(parents=True)
            sentinel = target / "model.pt"
            sentinel.write_bytes(b"existing checkpoint")
            with patch.object(entry, "load_runtime", side_effect=AssertionError("runtime loaded")), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                entry.main(common + ["--systems", "fitzhugh_nagumo", "--methods", "NODE", "--seeds", "42"])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(sentinel.read_bytes(), b"existing checkpoint")
            self.assertEqual(list(target.iterdir()), [sentinel])

    def test_training_dispatch_contract(self):
        training = SimpleNamespace(train_lac=Mock(return_value=("lac", {"scale": 0.5})),
                                   train_baseline=Mock(return_value=("baseline", True)))
        residual = SimpleNamespace(train_lac=Mock(return_value=("residual", {"scale": 0.5})))
        runtime = (None, training, residual, None)
        tr, val, times, constraint, path = object(), object(), object(), object(), Path("out")
        job = entry.build_plan(self.args("--systems", "lotka_volterra",
                                        "--methods", "NODE-LAC", "--seeds", "42"))[0]
        self.assertEqual(entry.dispatch_training(job, tr, val, times, constraint, "cpu", path, runtime),
                         ("lac", False, {"scale": 0.5}))
        training.train_lac.assert_called_once_with(
            tr, val, times, constraint, "cpu", 100, path, job["options"])
        pair = entry.build_plan(self.args("--suite", "residual", "--systems", "lotka_volterra",
                                         "--seeds", "42"))
        entry.dispatch_training(pair[0], tr, val, times, constraint, "cpu", path, runtime)
        self.assertEqual(training.train_lac.call_count, 2)
        entry.dispatch_training(pair[1], tr, val, times, constraint, "cpu", path, runtime)
        residual.train_lac.assert_called_once_with(
            tr, val, times, constraint, "cpu", 100, path, pair[1]["options"], 0.001)
        self.assertEqual(entry.dispatch_training(
            dict(job, method="HNN"), tr, val, times, constraint, "cpu", path, runtime),
            ("baseline", True, {}))
        training.train_baseline.assert_called_once_with("HNN", tr, times, constraint, "cpu", 100, path)

    def test_residual_scale_override_preserves_training_selection(self):
        jobs = entry.build_plan(self.args("--suite", "residual", "--systems", "lotka_volterra",
                                         "--seeds", "42"))
        for job in jobs:
            original = {"scale": 0.5, "validation_scale_scores": {"0.5": 1.0}}
            model = SimpleNamespace(correction_scale=0.5)
            selection = entry.use_published_scale(model, original, job)
            self.assertEqual(model.correction_scale, 0.4)
            self.assertEqual(selection["scale"], 0.4)
            self.assertEqual(selection["training_validation_scale"], 0.5)
            self.assertEqual(original["scale"], 0.5)

    def test_summary_groups_variants_and_uses_sample_sd(self):
        rows = [{"system": "s", "method": "m", "variant": "paired", "metrics": {"MSE": x}}
                for x in [1.0, 3.0]]
        rows.append({"system": "s", "method": "m", "variant": "single", "metrics": {"MSE": 7.0}})
        with tempfile.TemporaryDirectory() as tmp:
            result = entry.write_summary(rows, Path(tmp), "residual")
            with result.open() as stream:
                summary = {row["variant"]: row for row in csv.DictReader(stream)}
            self.assertEqual(set(summary), {"paired", "single"})
            self.assertEqual(summary["paired"]["n_seeds"], "2")
            self.assertEqual(float(summary["paired"]["MSE_mean"]), 2.0)
            self.assertEqual(float(summary["paired"]["MSE_sample_sd"]), statistics.stdev([1.0, 3.0]))
            self.assertEqual(summary["single"]["n_seeds"], "1")
            self.assertEqual(summary["single"]["MSE_sample_sd"], "")


class SyntheticSmoke(unittest.TestCase):
    def setUp(self):
        import torch
        from experiments import training
        self.torch = torch
        self.addCleanup(setattr, training, "DATA", training.DATA)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        generator = torch.Generator().manual_seed(17)
        manifest = {"source_commit": "synthetic-fixture", "files": {}}
        for system, dim in [("fitzhugh_nagumo", 17), ("lotka_volterra", 31)]:
            data = self.root / (system + "_data.pt")
            base = torch.randn(256, 1, dim, dtype=torch.float64, generator=generator) * 0.2
            states = base + torch.arange(3, dtype=torch.float64)[None, :, None] * 0.001
            torch.save({"train_states": states[:128], "test_states": states[128:],
                        "times": torch.tensor([0.0, 0.01, 0.02], dtype=torch.float64)}, data)
            manifest["files"][data.name] = {"sha256": entry.file_sha256(data)}
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        self.common = ["--data-dir", str(self.root), "--output-dir", str(self.root / "out"),
                       "--data-manifest", str(manifest_path), "--epochs", "1", "--device", "cpu",
                       "--seeds", "42"]

    def run_cli(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(self.common + list(extra)), 0)

    def check_run(self, relative):
        folder = self.root / "out" / relative
        for name in ("model.pt", "result.json", "provenance.json", "history.json", "curves.json"):
            self.assertTrue((folder / name).is_file(), str(folder / name))
        result = json.loads((folder / "result.json").read_text())
        self.assertFalse(result["published_epoch_schedule"])
        self.assertEqual(result["epochs"], 1)
        self.assertEqual(result["dataset"]["source_commit"], "synthetic-fixture")
        self.assertEqual(len(json.loads((folder / "history.json").read_text())), 1)
        for name in ("MSE", "MAE", "TCE", "CE"):
            self.assertTrue(math.isfinite(result["metrics"][name]), name)
        for source in ("reproduce.py", "experiments/training.py"):
            self.assertEqual(result["source_sha256"][source], entry.file_sha256(entry.REPO / source))
        curves = json.loads((folder / "curves.json").read_text())
        self.assertEqual(curves["times"], [0.0, 0.01, 0.02])
        self.assertEqual(len(curves["MSE"]), 3)
        self.assertEqual(len(curves["CE"]), 3)
        return folder, result

    def check_summary(self, suite, count):
        with (self.root / "out" / suite / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), count)
        self.assertTrue(all(row["n_seeds"] == "1" and row["MSE_sample_sd"] == "" for row in rows))

    def test_one_epoch_main_and_paired_residual_with_portable_outputs(self):
        self.run_cli("--suite", "main", "--systems", "fitzhugh_nagumo",
                     "--methods", "NODE", "NODE-LAC", "--save-predictions")
        for method in ("NODE", "NODE-LAC"):
            folder, result = self.check_run(f"main/fitzhugh_nagumo/{method}/seed42")
            saved = self.torch.load(folder / "predictions.pt", weights_only=True)
            self.assertEqual(tuple(saved["pred"].shape), (128, 3, 17))
            data = self.torch.load(self.root / "fitzhugh_nagumo_data.pt", weights_only=True)
            self.torch.testing.assert_close(saved["true"], data["test_states"], rtol=0, atol=0)
            self.torch.testing.assert_close(saved["pred"][:, 0], saved["true"][:, 0])
            self.assertIn("test_reference", result["diagnostics"])
        self.check_summary("main", 2)
        self.run_cli("--suite", "residual", "--systems", "lotka_volterra")
        for variant in ("zero-control", "regularized"):
            folder, result = self.check_run(f"residual/lotka_volterra/NODE-LAC/{variant}/seed42")
            checkpoint = self.torch.load(folder / "model.pt", weights_only=True)
            self.assertEqual(result["selection"]["scale"], 0.4)
            self.assertEqual(checkpoint["selected_scale"], 0.4)
            self.assertEqual(checkpoint["lambda_L"], 0.0 if variant == "zero-control" else 0.001)
            self.assertEqual(result["diagnostics"]["reference"]["sample_count"], 384)
            self.assertGreaterEqual(result["metrics"]["residual_score"], 0.0)
            self.assertFalse((folder / "predictions.pt").exists())
        self.check_summary("residual", 2)

    def test_one_epoch_ablation_data_efficiency_and_noise(self):
        self.run_cli("--suite", "ablation", "--systems", "fitzhugh_nagumo")
        for variant in ("NoGainNet", "NoConstraintLoss", "NoCorrection"):
            folder, result = self.check_run(f"ablation/fitzhugh_nagumo/NODE-LAC/{variant}/seed42")
            if variant == "NoCorrection":
                checkpoint = self.torch.load(folder / "model.pt", weights_only=True)
                self.assertEqual(checkpoint["training_scale"], 0.0)
                self.assertEqual(result["selection"]["scale"], 0.0)
        self.check_summary("ablation", 3)
        for suite, field, values, prefix in (
                ("data-efficiency", "fraction", (0.25, 0.5, 0.75), "fraction"),
                ("noise", "noise", (0.05, 0.1, 0.2), "sigma")):
            with self.subTest(suite=suite):
                self.run_cli("--suite", suite, "--systems", "fitzhugh_nagumo", "--methods", "NODE")
                for value in values:
                    folder, result = self.check_run(f"{suite}/fitzhugh_nagumo/NODE/{prefix}_{value}/seed42")
                    self.assertEqual(result[field], value)
                    self.assertFalse((folder / "predictions.pt").exists())
                self.check_summary(suite, 3)


if __name__ == "__main__":
    unittest.main()
