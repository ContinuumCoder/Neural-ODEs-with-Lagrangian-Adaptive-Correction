"""Synthetic inference and integrity checks for long-horizon evaluation."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from experiments import extrapolation, training
from nodesac.core.manifold import FHNManifold


class TinyNode(nn.Module):
    def __init__(self, dim, hidden=None, solver=None):
        super().__init__()
        self.f = nn.Linear(dim, dim, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.f.weight.zero_()


class TinyGain(nn.Module):
    def __init__(self, dim, hidden=None):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(-2., dtype=torch.float64))

    def forward(self, x):
        return self.bias.exp().expand(*x.shape[:-1], 1)


class TinyBaseline(TinyNode):
    def forward(self, time, x):
        return self.f(x)

    def predict(self, initial, times):
        return training.euler_integrate(self, initial, times)


class TinySystem:
    def get_manifold(self):
        return FHNManifold(n_grid=1, e_threshold=1.)


class ExtrapolationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def generated(self, count=1000):
        times = torch.arange(count, dtype=torch.float64) * .05
        initial = torch.linspace(-.5, .5, 256, dtype=torch.float64)
        states = initial[:, None, None] + torch.tensor([.1, .2, 2.], dtype=torch.float64)
        states = states + .002 * times[None, :, None]
        return {"train_states": states[:128].clone(), "test_states": states[128:].clone(), "times": times}

    def generator(self, **kwargs):
        self.assertEqual(kwargs, {"n_trajectories": 256, "t_span": (0, 50.),
                                  "dt_save": .05, "dt_integrate": .001, "seed": 42, "device": "cpu"})
        return self.generated()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.system = "fitzhugh_nagumo"
        self.args = SimpleNamespace(
            model_dir=self.root / "models", output_dir=self.root / "outputs",
            data_dir=self.root / "data", reference_dir=self.root / "references",
            save_predictions=True, device="cpu")
        self.args.data_dir.mkdir()
        self.data = self.generated(100)
        self.data_path = self.args.data_dir / f"{self.system}_data.pt"
        torch.save(self.data, self.data_path)
        self.datasets = {self.system: {"file": self.data_path.name, "sha256": training.hash_file(self.data_path)}}
        self.engine = SimpleNamespace(**{name: getattr(training, name) for name in [
            "hash_file", "write_json", "DTYPE", "PROTOCOL", "SYSTEMS", "LotkaVolterra",
            "ShallowWater", "FrankaRobot", "StandardizedConstraint", "ClosedLoopDynamics",
            "euler_integrate", "pad", "metrics"]})
        self.engine.FitzHughNagumo = TinySystem
        self.engine.NeuralODE = TinyNode
        self.engine.GainNet = TinyGain
        self.engine.build_baseline = lambda name, dim, constraint: (TinyBaseline(dim), False)
        self.engine.train_lac = Mock(side_effect=AssertionError("Evaluation must not train"))
        self.engine.train_baseline = Mock(side_effect=AssertionError("Evaluation must not train"))
        self.runtime = (torch, self.engine, None, None)
        split = torch.randperm(128, generator=torch.Generator().manual_seed(20260906))
        fit = self.data["train_states"][split[:102]].reshape(-1, 3)
        self.mean, self.std = fit.mean(0), fit.std(0).clamp_min(1e-6)
        self.physical = TinySystem().get_manifold()
        self.constraint = training.StandardizedConstraint(self.physical, self.mean, self.std)
        self.provenance = {
            "source_sha256": self.datasets[self.system]["sha256"],
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "fit_indices": split[:102].tolist(), "normalizer_fit_indices": split[:102].tolist(),
            "validation_indices": split[102:].tolist(), "constraint_coordinates": "z=(raw_state-mean)/std",
            "constraint_energy_threshold": 1., "constraint_amplitude_threshold": 2.,
            "data_fraction": 1., "normalized_noise_sigma": 0.}
        self.make_main("NODE-LAC")
        self.make_main("NODE")

    def make_main(self, method):
        path = self.args.model_dir / "main" / self.system / method / "seed42"
        path.mkdir(parents=True, exist_ok=True)
        if method == "NODE-LAC":
            node, gain = TinyNode(3), TinyGain(3)
            model = training.ClosedLoopDynamics(node.f, gain, self.constraint, correction_scale=.5)
            checkpoint = {"node": node.state_dict(), "gain": gain.state_dict(), "selected_scale": .5,
                          "training_scale": .3, "options": {"mu_init": .1, "effort_weight": .05}}
        else:
            model = TinyBaseline(3)
            checkpoint = model.state_dict()
        z0 = (self.data["test_states"][:, 0] - self.mean) / self.std
        with torch.no_grad():
            z = training.euler_integrate(model, z0, self.data["times"]).permute(1, 0, 2)
        metrics = extrapolation._public_metrics(training.metrics(
            z * self.std + self.mean, self.data["test_states"], self.physical, z,
            (self.data["test_states"] - self.mean) / self.std))
        result = {"protocol": training.PROTOCOL, "suite": "main", "variant": "main",
                  "system": self.system, "method": method, "seed": 42, "epochs": 100,
                  "dataset": self.datasets[self.system], "metrics": metrics,
                  "selection": {"scale": .5} if method == "NODE-LAC" else {},
                  "published_epoch_schedule": True}
        torch.save(checkpoint, path / "model.pt")
        training.write_json(path / "result.json", result)
        training.write_json(path / "provenance.json", self.provenance)
        return path

    def job(self, method="NODE-LAC", suffix=""):
        return {"suite": "long-horizon", "system": self.system, "method": method, "variant": "1000",
                "seed": 42, "epochs": 100,
                "relative_output": f"long-horizon/{self.system}/{method}/1000/seed42{suffix}"}

    def evaluate(self, job=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return extrapolation.evaluate(job or self.job(), self.args, self.datasets, self.runtime)

    def cache(self):
        with patch.dict(extrapolation.GENERATORS, {self.system: self.generator}):
            return extrapolation._reference(self.system, self.args, self.datasets[self.system], self.data)

    def test_complete_inference_reuses_fixed_scale_and_cpu_cache(self):
        with patch.dict(extrapolation.GENERATORS, {self.system: self.generator}), \
             patch("torch.cuda.synchronize", side_effect=AssertionError("CPU must not synchronize CUDA")), \
             patch.object(self.engine, "ClosedLoopDynamics", wraps=training.ClosedLoopDynamics) as build:
            result = self.evaluate()
        self.assertEqual(build.call_args.kwargs["correction_scale"], .5)
        self.assertFalse(result["training_performed"])
        self.assertTrue(result["main_replay"]["metrics_reproduced"])
        self.assertEqual(result["selection"]["scale"], .5)
        self.assertEqual(result["observations"], 1000)
        self.assertNotIn("Stability", result["metrics"])
        self.assertIn("near_feasible_fraction", result["metrics"])
        destination = self.args.output_dir / self.job()["relative_output"]
        curves = json.loads((destination / "curves.json").read_text())
        self.assertEqual(len(curves["MSE"]), 1000)
        self.assertEqual(len(curves["CE"]), 1000)
        arrays = torch.load(destination / "predictions.pt", weights_only=True)
        self.assertEqual(set(arrays), {"pred", "true", "times"})
        self.assertEqual(arrays["pred"].shape, (128, 1000, 3))
        forbidden = Mock(side_effect=AssertionError("A verified reference must be reused"))
        forbidden.__name__ = self.generator.__name__
        with patch.dict(extrapolation.GENERATORS, {self.system: forbidden}):
            replay = self.evaluate(self.job(suffix="_repeat"))
        self.assertEqual(result["metrics"], replay["metrics"])
        self.engine.train_lac.assert_not_called()
        self.engine.train_baseline.assert_not_called()

    def test_baseline_checkpoint_and_optional_prediction_storage(self):
        self.args.save_predictions = False
        with patch.dict(extrapolation.GENERATORS, {self.system: self.generator}):
            result = self.evaluate(self.job("NODE"))
        destination = self.args.output_dir / self.job("NODE")["relative_output"]
        self.assertFalse((destination / "predictions.pt").exists())
        self.assertTrue(result["main_replay"]["metrics_reproduced"])

    def test_main_integrity_guards_precede_reference_generation(self):
        path = self.args.model_dir / "main" / self.system / "NODE-LAC" / "seed42"
        original_result = json.loads((path / "result.json").read_text())
        cases = [
            ("epochs", 99, "100-epoch"),
            ("dataset", {"sha256": "incorrect"}, "dataset SHA-256"),
            ("metrics", {**original_result["metrics"], "MSE": 123.}, "metric mismatch"),
            ("selection", {"scale": .75}, "correction scale"),
        ]
        for key, value, message in cases:
            with self.subTest(key=key):
                training.write_json(path / "result.json", {**original_result, key: value})
                forbidden = Mock(side_effect=AssertionError("No reference generation before replay"))
                forbidden.__name__ = "forbidden"
                with patch.dict(extrapolation.GENERATORS, {self.system: forbidden}):
                    with self.assertRaisesRegex(ValueError, message):
                        self.evaluate()
        training.write_json(path / "result.json", original_result)
        training.write_json(path / "provenance.json",
                            {**self.provenance, "mean": [0., 0., 0.]})
        with self.assertRaisesRegex(ValueError, "normalization"):
            self.evaluate()

    def test_reference_file_hash_and_manifest_guards(self):
        _, _, target = self.cache()
        original = target.read_bytes()
        target.write_bytes(original + b"changed")
        with patch.dict(extrapolation.GENERATORS, {self.system: self.generator}):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                extrapolation._reference(self.system, self.args, self.datasets[self.system], self.data)
        target.write_bytes(original)
        manifest_path = target.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text())
        manifest["generator_sha256"] = "changed"
        training.write_json(manifest_path, manifest)
        with patch.dict(extrapolation.GENERATORS, {self.system: self.generator}):
            with self.assertRaisesRegex(ValueError, "source hash"):
                extrapolation._reference(self.system, self.args, self.datasets[self.system], self.data)

    def test_rehashed_cache_still_checks_both_prefixes(self):
        reference, manifest, target = self.cache()
        for key in ["train_prefix", "true"]:
            with self.subTest(key=key):
                corrupted = {name: value.clone() for name, value in reference.items()}
                corrupted[key][0, 0, 0] += 1.
                torch.save(corrupted, target)
                training.write_json(target.with_suffix(".json"),
                                    {**manifest, "reference_sha256": training.hash_file(target)})
                with patch.dict(extrapolation.GENERATORS, {self.system: self.generator}):
                    with self.assertRaisesRegex(ValueError, "prefix mismatch"):
                        extrapolation._reference(self.system, self.args, self.datasets[self.system], self.data)

    def test_cpu_generator_prefix_mismatch_is_explicit_and_not_cached(self):
        for partition in ["train_states", "test_states"]:
            with self.subTest(partition=partition):
                def different_generator(**kwargs):
                    generated = self.generated()
                    generated[partition][0, 0, 0] += 1.
                    return generated
                with patch.dict(extrapolation.GENERATORS, {self.system: different_generator}):
                    with self.assertRaisesRegex(ValueError, "CPU generation.*random sequence"):
                        extrapolation._reference(self.system, self.args, self.datasets[self.system], self.data)
                self.assertFalse((self.args.reference_dir / f"{self.system}_long.pt").exists())


if __name__ == "__main__":
    unittest.main()
