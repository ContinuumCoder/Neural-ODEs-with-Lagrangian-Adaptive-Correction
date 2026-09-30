"""Synthetic tests for published plotting statistics and input completeness."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from experiments import plotting as plot


def write_record(root, system, method, seed, suite="main", variant=None, predictions=False):
    directory = plot.run_directory(root, suite, system, method, seed, variant)
    directory.mkdir(parents=True, exist_ok=True)
    index = plot.SEEDS.index(seed)
    value = .02*(index+1) + .003*(len(method)+1)
    record = {"suite": suite, "system": system, "method": method, "seed": seed,
              "variant": variant, "metrics": {"MSE": value, "MAE": value*2, "TCE": value/3}}
    (directory/"result.json").write_text(json.dumps(record))
    history = [{"epoch": epoch, "trajectory_MSE": value/epoch} for epoch in range(1, 101)]
    (directory/"history.json").write_text(json.dumps(history))
    if predictions:
        dim = 31 if system in ("lotka_volterra", "shallow_water") else 17
        times = torch.arange(100, dtype=torch.float64)*(.015 if system == "lotka_volterra" else .05)
        true = torch.sin(times[None, :, None] + torch.arange(dim, dtype=torch.float64)[None, None, :]/10)
        pred = true + value*torch.cos(3*times)[None, :, None]
        torch.save({"pred": pred, "true": true, "times": times}, directory/"predictions.pt")
    return directory


class PlottingTests(unittest.TestCase):
    def test_sample_sd_and_invalid_groups(self):
        mean, sd = plot.seed_statistics([1., 2., 4.])
        self.assertAlmostEqual(mean, 7/3)
        self.assertAlmostEqual(sd, (7/3)**.5)
        for values in ([1., 2.], [1., 2., float("nan")]):
            with self.assertRaises(ValueError):
                plot.seed_statistics(values)

    def test_centered_edges_and_sd_after_smoothing(self):
        pulse = np.array([0., 0., 9., 0., 0.])
        data = np.stack([pulse*x for x in (1., 2., 3.)])[:, :, None]
        expected = np.array([3., 2.25, 1.8, 2.25, 3.])
        displayed = plot.centered_mean(data)
        np.testing.assert_array_equal(data[0, :, 0], pulse)
        np.testing.assert_allclose(displayed[0, :, 0], expected, rtol=0, atol=1e-15)
        mean, sd = plot.seed_statistics(displayed)
        np.testing.assert_allclose(mean[:, 0], expected*2, rtol=0, atol=1e-15)
        np.testing.assert_allclose(sd[:, 0], expected, rtol=0, atol=1e-15)
        with self.assertRaises(ValueError):
            plot.centered_mean(data, width=4)

    def test_surface_uses_unsmoothed_mean_then_absolute_error(self):
        truth = np.zeros((100, 17))
        offsets = np.zeros((3, 100, 17))
        offsets[:, 1, 0] = [-2., 0., 2.]
        offsets[:, 2, 0] = [0., 3., 6.]
        arrays = {method: offsets.copy() for method in plot.QUALITATIVE_METHODS}
        states, errors = plot.surface_statistics(arrays, truth)
        self.assertEqual(states[1][2, 0], 3.)
        self.assertEqual(states[1][1, 0], 0.)
        self.assertEqual(errors[1][1, 0], 0.)  # abs(mean), not mean(abs)
        self.assertEqual(errors[1][2, 0], 3.)
        self.assertEqual(states[1].shape, (50, 8))

    def test_missing_seed_and_non_main_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in plot.SEEDS[:2]:
                write_record(root, "fitzhugh_nagumo", "NODE", seed)
            with self.assertRaisesRegex(plot.MissingGroup, "missing seed 456"):
                plot.read_group(root, "fitzhugh_nagumo", "NODE")
            for seed in plot.SEEDS:
                write_record(root, "fitzhugh_nagumo", "NODE", seed, "noise", "sigma_0.05")
            mean, sd, paths = plot.metric_group(root, "fitzhugh_nagumo", "NODE", suite="noise", variant="sigma_0.05")
            self.assertGreater(mean, 0)
            self.assertAlmostEqual(sd, .02)
            self.assertIn("sigma_0.05", str(paths[0]))
            report = plot.render(root, root/"plots", "summary")
            self.assertFalse(report["exported"])
            self.assertEqual(len(report["skipped"]), 3)
            self.assertFalse(report["failed"])

    def test_reference_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in plot.SEEDS:
                directory = write_record(root, "fitzhugh_nagumo", "NODE", seed, predictions=True)
            pack = torch.load(directory/"predictions.pt", weights_only=True)
            pack["true"][0, 0, 0] += 1
            torch.save(pack, directory/"predictions.pt")
            with self.assertRaisesRegex(ValueError, "differ between seeds"):
                plot.prediction_group(root, "fitzhugh_nagumo", "NODE")

    def test_complete_render_and_input_immutability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, out = Path(temporary)/"inputs", Path(temporary)/"plots"
            methods = ("NODE-LAC", "NODE", "SNDE", "PORT-HJNN", "ConCerNet", "PNODE", "CPNODE")
            for system in plot.SYSTEMS:
                for method in methods:
                    for seed in plot.SEEDS:
                        write_record(root, system, method, seed, predictions=True)
            for system in plot.SYSTEMS[:3]:
                for method in ("NODE-LAC", "NODE", "SNDE"):
                    for fraction in (.25, .5, .75):
                        for seed in plot.SEEDS:
                            write_record(root, system, method, seed, "data-efficiency", f"fraction_{fraction}")
                for method in ("NODE-LAC", "NODE"):
                    for sigma in (.05, .1, .2):
                        for seed in plot.SEEDS:
                            write_record(root, system, method, seed, "noise", f"sigma_{sigma}")
            inputs = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            report = plot.render(root, out)
            self.assertEqual(report["failed"], [])
            self.assertEqual(report["skipped"], [])
            self.assertEqual(len(report["exported"]), 9)
            for stem in report["exported"]:
                self.assertTrue((out/f"{stem}.pdf").read_bytes().startswith(b"%PDF"))
                self.assertTrue((out/f"{stem}.png").read_bytes().startswith(b"\x89PNG"))
                meta = json.loads((out/f"{stem}.json").read_text())
                self.assertEqual(meta["seed_ids"], list(plot.SEEDS))
                self.assertTrue(meta["source_sha256"])
            meta = json.loads((out/"fitzhugh_nagumo_trajectories.json").read_text())
            self.assertEqual(meta["smoothing"]["methods"], ["PNODE", "CPNODE"])
            meta = json.loads((out/"fitzhugh_nagumo_3d_surface.json").read_text())
            self.assertFalse(meta["smoothing"])
            after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
            self.assertEqual(inputs, after)


if __name__ == "__main__":
    unittest.main()
