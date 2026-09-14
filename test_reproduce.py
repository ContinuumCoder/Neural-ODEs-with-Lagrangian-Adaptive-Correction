"""Portable-entrypoint contracts and small synthetic-data smoke tests."""
import contextlib
import csv
import io
import json
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

    def test_main_plan_and_published_settings(self):
        jobs = entry.build_plan(self.args())
        self.assertEqual(len(jobs), 120)
        self.assertEqual({j["epochs"] for j in jobs}, {100})
        self.assertTrue(all(j["options"] == {"mu_init": 0.1, "effort_weight": 0.05}
                            for j in jobs))
        self.assertTrue(all(j["evaluation_scale"] is None for j in jobs))
        self.assertEqual(len({j["relative_output"] for j in jobs}), 120)

    def test_residual_pairing_and_fixed_scales(self):
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
                else:
                    self.assertEqual({j["lambda_L"] for j in matched},
                                     {0.0, entry.RESIDUAL_WEIGHTS[system]})

    def test_invalid_plan_arguments(self):
        for extra in (["--epochs", "0"], ["--systems", "shallow_water", "shallow_water"],
                      ["--suite", "residual", "--methods", "NODE"],
                      ["--variant", "zero-control"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                entry.build_plan(self.args(*extra))

    def test_hash_validation_and_dry_run_without_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "fitzhugh_nagumo_data.pt"
            data.write_bytes(b"hash-only fixture: this is never unpickled")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"source_commit": "synthetic-fixture",
                "files": {data.name: {"sha256": entry.file_sha256(data)}}}))
            argv = ["--data-dir", str(root), "--output-dir", str(root / "out"),
                    "--data-manifest", str(manifest), "--systems", "fitzhugh_nagumo",
                    "--methods", "NODE-LAC", "--seeds", "42", "--dry-run"]
            captured = io.StringIO()
            with patch.object(entry, "load_runtime", side_effect=AssertionError("runtime loaded")), \
                    contextlib.redirect_stdout(captured):
                self.assertEqual(entry.main(argv), 0)
            self.assertEqual(json.loads(captured.getvalue())["run_count"], 1)
            self.assertFalse((root / "out").exists())
            data.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                entry.verify_data(entry.make_parser().parse_args(argv))

    def test_training_dispatch_contract(self):
        frozen = SimpleNamespace(train_lac=Mock(return_value=("lac", {"scale": 0.5})),
                                 train_baseline=Mock(return_value=("baseline", True)))
        residual = SimpleNamespace(train_lac=Mock(return_value=("residual", {"scale": 0.5})))
        runtime = (None, frozen, residual, None)
        tr, val, times, constraint, path = object(), object(), object(), object(), Path("out")
        main_job = entry.build_plan(self.args("--systems", "lotka_volterra",
                                             "--methods", "NODE-LAC", "--seeds", "42"))[0]
        entry.dispatch_training(main_job, tr, val, times, constraint, "cpu", path, runtime)
        frozen.train_lac.assert_called_once_with(
            tr, val, times, constraint, "cpu", 100, path, main_job["options"])
        pair = entry.build_plan(self.args("--suite", "residual", "--systems", "lotka_volterra",
                                         "--seeds", "42"))
        entry.dispatch_training(pair[0], tr, val, times, constraint, "cpu", path, runtime)
        self.assertEqual(frozen.train_lac.call_count, 2)
        entry.dispatch_training(pair[1], tr, val, times, constraint, "cpu", path, runtime)
        residual.train_lac.assert_called_once_with(
            tr, val, times, constraint, "cpu", 100, path, pair[1]["options"], 0.001)
        baseline_job = dict(main_job, method="HNN")
        self.assertEqual(entry.dispatch_training(
            baseline_job, tr, val, times, constraint, "cpu", path, runtime),
            ("baseline", True, {}))
        frozen.train_baseline.assert_called_once_with("HNN", tr, times, constraint, "cpu", 100, path)

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

    def test_summary_uses_sample_sd_and_does_not_invent_single_seed_sd(self):
        rows = [{"system": "s", "method": "m", "variant": "v", "metrics": {"MSE": x}}
                for x in [1.0, 3.0]]
        with tempfile.TemporaryDirectory() as tmp:
            result = entry.write_summary(rows, Path(tmp), "main")
            with result.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row["MSE_mean"]), 2.0)
            self.assertEqual(float(row["MSE_sample_sd"]), statistics.stdev([1.0, 3.0]))
            result = entry.write_summary(rows[:1], Path(tmp), "main")
            with result.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["MSE_sample_sd"], "")


class SyntheticSmoke(unittest.TestCase):
    def test_zero_residual_loop_matches_frozen_tensors_on_cpu(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = entry.load_runtime(root)
            _, frozen, residual, _ = runtime
            physical = frozen.FitzHughNagumo().get_manifold()
            constraint = frozen.StandardizedConstraint(
                physical, torch.zeros(17, dtype=torch.float64),
                torch.ones(17, dtype=torch.float64))
            torch.manual_seed(19)
            train = torch.randn(4, 3, 17, dtype=torch.float64) * 0.2
            validation = torch.randn(2, 3, 17, dtype=torch.float64) * 0.2
            times = torch.tensor([0., .01, .02], dtype=torch.float64)
            options = {"mu_init": .01, "effort_weight": .01}
            one, two = root / "frozen", root / "residual"
            one.mkdir()
            two.mkdir()
            frozen.seed_everything(987)
            frozen.train_lac(train, validation, times, constraint, "cpu", 1, one, options)
            frozen.seed_everything(987)
            with patch.object(residual, "loss_residual",
                              side_effect=AssertionError("zero weight evaluated residual")):
                residual.train_lac(train, validation, times, constraint, "cpu", 1, two, options, 0.)
            old = torch.load(one / "model.pt", weights_only=True)
            new = torch.load(two / "model.pt", weights_only=True)
            for key in ("node", "gain"):
                self.assertEqual(old[key].keys(), new[key].keys())
                for name in old[key]:
                    torch.testing.assert_close(old[key][name], new[key][name], rtol=0, atol=0)
            torch.testing.assert_close(old["log_mu"], new["log_mu"], rtol=0, atol=0)
            self.assertEqual(old["selected_scale"], new["selected_scale"])
            self.assertEqual(json.loads((one / "history.json").read_text()),
                             json.loads((two / "history.json").read_text()))

    def test_one_epoch_cpu_entrypoint_without_parent_artifacts(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(17)
            manifest = {"source_commit": "synthetic-fixture", "files": {}}
            for system, dim in [("fitzhugh_nagumo", 17), ("lotka_volterra", 31)]:
                data = root / (system + "_data.pt")
                base = torch.randn(256, 1, dim, dtype=torch.float64) * 0.2
                states = base + torch.arange(3, dtype=torch.float64)[None, :, None] * 0.001
                torch.save({"train_states": states[:128], "test_states": states[128:],
                            "times": torch.tensor([0.0, 0.01, 0.02], dtype=torch.float64)}, data)
                manifest["files"][data.name] = {"sha256": entry.file_sha256(data)}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            common = ["--data-dir", str(root), "--output-dir", str(root / "out"),
                      "--data-manifest", str(manifest_path), "--epochs", "1", "--device", "cpu",
                      "--seeds", "42"]
            runtime = entry.load_runtime(root)
            _, frozen, residual, evaluation = runtime
            with patch.object(residual, "check_parent", side_effect=AssertionError("parent accessed")), \
                    patch.object(evaluation, "verify_lock", side_effect=AssertionError("lock accessed")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(common + ["--suite", "main", "--systems",
                    "fitzhugh_nagumo", "--methods", "NODE"]), 0)
                self.assertEqual(entry.main(common + ["--suite", "residual", "--systems",
                    "lotka_volterra"]), 0)
            for variant in ("zero-control", "regularized"):
                folder = root / "out/residual/lotka_volterra" / variant / "seed42"
                result = json.loads((folder / "result.json").read_text())
                checkpoint = torch.load(folder / "model.pt", weights_only=False)
                self.assertEqual(result["selection"]["scale"], 0.4)
                self.assertEqual(checkpoint["selected_scale"], 0.4)
                self.assertFalse(result["published_epoch_schedule"])
                self.assertEqual(result["diagnostics"]["reference"]["sample_count"], 384)
                self.assertEqual(len(json.loads((folder / "history.json").read_text())), 1)
                self.assertEqual(result["dataset"]["source_commit"], "synthetic-fixture")
                self.assertGreaterEqual(result["metrics"]["residual_score"], 0.0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                entry.main(common + ["--suite", "residual", "--systems", "lotka_volterra"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
