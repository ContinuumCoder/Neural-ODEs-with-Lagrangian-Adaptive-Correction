"""Deterministic CPU checks for training updates and residual diagnostics."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from experiments import diagnostics, residual, training
from nodesac.core.manifold import FHNManifold


class TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def problem(self, energy=0.2):
        physical = FHNManifold(n_grid=1, e_threshold=1.)
        constraint = training.StandardizedConstraint(
            physical, torch.zeros(3, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64))
        states = torch.tensor([0.1, 0.2, energy], dtype=torch.float64).repeat(4, 5, 1)
        times = torch.linspace(0., 0.02, 5, dtype=torch.float64)
        return states, times, constraint

    def test_standardized_gradient_and_metric_units(self):
        physical = FHNManifold(n_grid=1, e_threshold=1.)
        mean = torch.tensor([10., -2., 5.], dtype=torch.float64)
        scale = torch.tensor([2., 3., 4.], dtype=torch.float64)
        constraint = training.StandardizedConstraint(physical, mean, scale)
        z = torch.tensor([[3., 2., 1.5]], dtype=torch.float64, requires_grad=True)
        expected = physical.k(z)
        torch.testing.assert_close(constraint.k(z), expected, rtol=0, atol=0)
        u = torch.autograd.grad(constraint.distance(z).sum(), z)[0]
        v = torch.autograd.grad(physical.distance(z).sum(), z)[0]
        torch.testing.assert_close(u, v, rtol=0, atol=0)
        pred_z = z.detach().reshape(1, 1, 3).repeat(1, 2, 1)
        true_z = torch.zeros_like(pred_z)
        metrics = training.metrics(mean + scale * pred_z, mean + scale * true_z,
                                   physical, pred_z, true_z)
        self.assertEqual(metrics["CE"], float(physical.k(pred_z).square().sum(-1).mean()))
        self.assertEqual(metrics["MSE"], float((scale * pred_z).square().mean()))

    def test_residual_field_and_parameter_gradients(self):
        class Constraint:
            def k(self, x):
                return torch.relu(x - 1.)

        layer = nn.Linear(2, 2, bias=False, dtype=torch.float64)
        gain = nn.Sequential(nn.Linear(2, 1, dtype=torch.float64), nn.Softplus())
        with torch.no_grad():
            layer.weight.copy_(10. * torch.eye(2, dtype=torch.float64))
            gain[0].weight.zero_()
            gain[0].bias.zero_()
        node = SimpleNamespace(f=layer)
        x = torch.tensor([[0., 0.], [1., 1.], [2., 3.], [10., 20.]],
                         dtype=torch.float64, requires_grad=True)
        constraint = Constraint()
        for rho in [0., 0.3, 0.75, 2.]:
            values = residual.residual_values(node, gain, x, constraint, rho, True)
            field = training.ClosedLoopDynamics(layer, gain, constraint, correction_scale=rho)(
                torch.tensor(0.), x.detach())
            true_gradient = torch.relu(x.detach() - 1.)
            torch.testing.assert_close(values["field"], field, rtol=0, atol=0)
            torch.testing.assert_close(values["u"], true_gradient, rtol=0, atol=0)
            torch.testing.assert_close(values["dotV"], (true_gradient * field).sum(-1),
                                       rtol=0, atol=0)
        loss = residual.loss_residual(node, gain, x, x * 2., constraint, .3)
        layer.zero_grad()
        gain.zero_grad()
        loss.backward()
        self.assertIsNone(x.grad)
        for network in [layer, gain]:
            gradients = [parameter.grad for parameter in network.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_residual_sampling_has_no_rng_or_state_gradient(self):
        states = torch.arange(12800, dtype=torch.float64).reshape(64, 100, 2).requires_grad_(True)
        before = torch.get_rng_state().clone()
        sample = residual.sample_states(states)
        indices = torch.linspace(0, 6399, 512, dtype=torch.float64).long()
        self.assertTrue(torch.equal(sample, states.reshape(-1, 2)[indices]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(tuple(sample.shape), (512, 2))
        self.assertFalse(sample.requires_grad)
        small = states[:1, :5]
        self.assertTrue(torch.equal(residual.sample_states(small), small.reshape(-1, 2)))

    def test_zero_residual_weight_matches_base_training(self):
        states, times, constraint = self.problem(energy=2.)
        options = {"mu_init": .05, "effort_weight": .05}
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "base"
            zero = Path(temporary) / "zero"
            base.mkdir()
            zero.mkdir()
            training.seed_everything(987)
            training.train_lac(states, states, times, constraint, "cpu", 2, base, options)
            training.seed_everything(987)
            with patch.object(residual, "loss_residual", side_effect=AssertionError("Zero weight must skip the residual")):
                residual.train_lac(states, states, times, constraint, "cpu", 2, zero, options, 0.)
            a = torch.load(base / "model.pt", map_location="cpu", weights_only=True)
            b = torch.load(zero / "model.pt", map_location="cpu", weights_only=True)
            for network in ["node", "gain"]:
                self.assertEqual(a[network].keys(), b[network].keys())
                for name in a[network]:
                    self.assertTrue(torch.equal(a[network][name], b[network][name]), (network, name))
            self.assertTrue(torch.equal(a["log_mu"], b["log_mu"]))
            self.assertEqual(a["selected_scale"], b["selected_scale"])
            self.assertEqual(json.loads((base / "history.json").read_text()),
                             json.loads((zero / "history.json").read_text()))

    def test_actual_dual_update_and_positive_residual_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            for energy, sign in [(0.2, -1), (2., 1)]:
                with self.subTest(energy=energy):
                    states, times, constraint = self.problem(energy)
                    path = Path(temporary) / str(energy)
                    path.mkdir()
                    training.seed_everything(123)
                    training.train_lac(states, states, times, constraint, "cpu", 1, path,
                                       {"mu_init": .1, "effort_weight": .05})
                    ck = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
                    self.assertGreater(sign * (float(ck["log_mu"]) - math.log(.1)), 0.)
            states, times, constraint = self.problem(2.)
            path = Path(temporary) / "positive"
            path.mkdir()
            training.seed_everything(123)
            with patch.object(residual, "loss_residual", wraps=residual.loss_residual) as evaluated:
                residual.train_lac(states, states, times, constraint, "cpu", 1, path,
                                   {"mu_init": .1, "effort_weight": .05}, .1)
            self.assertEqual(evaluated.call_count, 1)
            history = json.loads((path / "history.json").read_text())
            self.assertTrue(math.isfinite(history[0]["Lyapunov_loss"]))
            self.assertGreaterEqual(history[0]["Lyapunov_loss"], 0.)
            ck = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(ck["lambda_L"], .1)

    def test_all_baseline_training_interfaces(self):
        states, times, constraint = self.problem()
        with tempfile.TemporaryDirectory() as temporary:
            for name in training.METHODS[1:]:
                with self.subTest(method=name):
                    training.seed_everything(42)
                    path = Path(temporary) / name
                    path.mkdir()
                    model, padded = training.train_baseline(
                        name, states, times, constraint, "cpu", 1, path)
                    prediction = model.predict(training.pad(states[:, 0], padded), times)
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertEqual(prediction.shape[:2], (len(times), len(states)))
                    history = json.loads((path / "history.json").read_text())
                    self.assertEqual(history[0]["epoch"], 1)
                    self.assertTrue(math.isfinite(history[0]["trajectory_MSE"]))

    def test_full_diagnostics_matches_residual_values_and_empty_active_set(self):
        states, _, constraint = self.problem(2.)
        layer = nn.Linear(3, 3, bias=False, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(3, dtype=torch.float64))
        node = SimpleNamespace(f=layer)
        gain = training.FixedGain()
        dyn = training.ClosedLoopDynamics(layer, gain, constraint, correction_scale=.3)
        states = states[:1, :1].repeat(13, 100, 1)
        expected = residual.diagnostic_summary(residual.residual_values(node, gain, states, constraint, .3))
        actual = diagnostics.full_diagnostics(dyn, states, constraint)
        self.assertEqual(actual["sample_count"], 1300)
        for key, value in expected.items():
            if value is None:
                self.assertIsNone(actual[key])
            else:
                self.assertTrue(math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-12), key)
        self.assertAlmostEqual(diagnostics.score({"reference": actual, "rollout": actual}),
                               actual["normalized_residual_sq_mean"])
        empty, _, _ = self.problem(0.2)
        empty_stats = diagnostics.full_diagnostics(dyn, empty, constraint)
        self.assertEqual(empty_stats["active_count"], 0)
        self.assertIsNone(diagnostics.score({"reference": empty_stats, "rollout": actual}))


if __name__ == "__main__":
    unittest.main()
