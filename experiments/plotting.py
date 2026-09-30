"""Plot complete three-seed experiment results.

Usage: python -m experiments.plotting --input-dir outputs --output-dir plots
Trajectory/surface plots require main runs saved with --save-predictions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

SEEDS = (42, 123, 456)
SYSTEMS = ("fitzhugh_nagumo", "lotka_volterra", "shallow_water", "franka_robot")
LABELS = dict(zip(SYSTEMS, ("FitzHugh–Nagumo", "Lotka–Volterra", "Shallow Water", "Robot Arm")))
COLORS = {"NODE-LAC": "#2166AC", "NODE": "#E38D4A", "SNDE": "#38966D",
          "PORT-HJNN": "#8754A1", "ConCerNet": "#D59A26", "PNODE": "#9C665A",
          "CPNODE": "#C84C64"}
QUALITATIVE_METHODS = ("NODE-LAC", "NODE", "PNODE", "CPNODE")
STYLES = {"PNODE": {"color": "#858585", "ls": "--", "lw": .65, "alpha": .85, "zorder": 1},
          "CPNODE": {"color": "#555555", "ls": ":", "lw": .7, "alpha": .9, "zorder": 2},
          "NODE": {"color": COLORS["NODE"], "lw": 1., "alpha": 1, "zorder": 4},
          "NODE-LAC": {"color": COLORS["NODE-LAC"], "lw": 1.2, "alpha": 1, "zorder": 5}}

class MissingGroup(FileNotFoundError):
    """A target figure lacks a required seed or prediction file."""

def seed_statistics(values):
    """Arithmetic mean and sample SD across the three training seeds."""
    array = np.asarray(values, dtype=float)
    if array.shape[0] != len(SEEDS) or not np.isfinite(array).all():
        raise ValueError("Expected exactly three finite seed values/arrays")
    return array.mean(axis=0), array.std(axis=0, ddof=1)

def centered_mean(values, width=5):
    """Centered time-axis mean with actual sample-count denominators at edges."""
    values = np.asarray(values)
    if values.ndim != 3 or width < 1 or width % 2 != 1:
        raise ValueError("Expected (seed,time,coordinate) data and a positive odd width")
    half = width // 2
    return np.stack([values[:, max(0, j-half):min(values.shape[1], j+half+1)].mean(axis=1)
                     for j in range(values.shape[1])], axis=1)

def run_directory(root, suite, system, method, seed, variant=None):
    path = Path(root) / suite / system / method
    if suite != "main":
        if variant is None:
            raise ValueError("Non-main results require a variant")
        path = path / variant
    return path / f"seed{seed}"

def read_group(root, system, method, suite="main", variant=None):
    records, paths = [], []
    for seed in SEEDS:
        path = run_directory(root, suite, system, method, seed, variant) / "result.json"
        if not path.is_file():
            raise MissingGroup(f"{suite}/{system}/{method}/{variant or 'main'}: missing seed {seed}: {path}")
        record = json.loads(path.read_text())
        for key, expected in (("suite", suite), ("system", system), ("method", method), ("seed", seed)):
            if record.get(key) != expected:
                raise ValueError(f"{path}: {key} does not match {expected!r}")
        if suite != "main" and record.get("variant") != variant:
            raise ValueError(f"{path}: variant mismatch")
        records.append(record)
        paths.append(path)
    return records, paths

def metric_group(root, system, method, metric="MSE", suite="main", variant=None):
    records, paths = read_group(root, system, method, suite, variant)
    mean, sd = seed_statistics([r["metrics"][metric] for r in records])
    return float(mean), float(sd), paths

def prediction_group(root, system, method):
    _, results = read_group(root, system, method)
    predictions, truth, times, paths = [], None, None, []
    for result in results:
        path = result.parent / "predictions.pt"
        if not path.is_file():
            raise MissingGroup(f"{path}: main reproduction requires --save-predictions")
        pack = torch.load(path, map_location="cpu", weights_only=True)
        pred, true, clock = [np.asarray(pack[key].detach().cpu(), dtype=float)
                             for key in ("pred", "true", "times")]
        if pred.ndim != 3 or pred.shape != true.shape or clock.ndim != 1 or len(clock) != pred.shape[1]:
            raise ValueError(f"{path}: incompatible prediction/reference/time shapes")
        if not all(np.isfinite(a).all() for a in (pred, true, clock)):
            raise ValueError(f"{path}: non-finite prediction data")
        if pred.shape[0] < 1 or pred.shape[1] < 50 or not np.all(np.diff(clock) > 0):
            raise ValueError(f"{path}: expected increasing trajectories with at least 50 observations")
        if truth is not None and (not np.array_equal(truth, true[0]) or not np.array_equal(times, clock)):
            raise ValueError(f"{path}: reference/time data differ between seeds")
        truth, times = true[0], clock
        predictions.append(pred[0])
        paths.extend((result, path))
    return np.stack(predictions), truth, times, paths

def qualitative_data(root, system):
    arrays, sources, truth, times = {}, [], None, None
    for method in QUALITATIVE_METHODS:
        array, reference, clock, paths = prediction_group(root, system, method)
        if truth is not None and (not np.array_equal(truth, reference) or not np.array_equal(times, clock)):
            raise ValueError(f"{system}: reference/time data differ between methods")
        arrays[method], truth, times = array, reference, clock
        sources.extend(paths)
    return arrays, truth, times, sources

def surface_statistics(arrays, truth):
    """Unsmoothed first-field states and absolute errors of seed-mean predictions."""
    n = (truth.shape[-1]-1)//2
    states = [truth[:50, :n]] + [seed_statistics(arrays[m])[0][:50, :n] for m in QUALITATIVE_METHODS]
    errors = [np.zeros_like(states[0])] + [np.abs(state-states[0]) for state in states[1:]]
    return states, errors

def save_figure(fig, out, stem, sources, details, input_root):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{extension}", bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    source_hashes = {str(path.relative_to(input_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(set(sources))}
    metadata = {"seed_ids": list(SEEDS), "uncertainty": "sample standard deviation across seeds, ddof=1",
                "source_sha256": source_hashes, **details}
    (out / f"{stem}.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")

def log_limits(axis, means, sds):
    means, sds = np.asarray(means), np.asarray(sds)
    positive = means[means > 0]
    if positive.size == 0:
        raise ValueError("Logarithmic plots require at least one positive mean")
    axis.set_yscale("log")
    axis.set_ylim(float(positive.min())/2, float((means+sds).max())*2)
    axis.grid(axis="y", alpha=.15, linewidth=.5)
    axis.set_axisbelow(True)

def summary_figure(root, out, kind):
    methods = {"bar": ("NODE-LAC", "SNDE", "PORT-HJNN", "NODE", "ConCerNet", "PNODE"),
               "radar": ("NODE-LAC", "SNDE", "NODE", "ConCerNet", "PNODE", "CPNODE"),
               "loss": ("NODE-LAC", "NODE", "SNDE", "ConCerNet", "PNODE")}[kind]
    sources, data = [], {}
    for system in SYSTEMS:
        for method in methods:
            if kind == "loss":
                _, paths = read_group(root, system, method)
                histories = []
                for path in paths:
                    history_path = path.parent / "history.json"
                    if not history_path.is_file():
                        raise MissingGroup(str(history_path))
                    history = json.loads(history_path.read_text())
                    if len(history) != 100 or [e["epoch"] for e in history] != list(range(1, 101)):
                        raise ValueError(f"{history_path}: expected all 100 ordered epochs")
                    histories.append([e["trajectory_MSE"] for e in history])
                    sources.extend((path, history_path))
                data[system, method] = seed_statistics(histories)
            else:
                for metric in (("MSE", "MAE", "TCE") if kind == "radar" else ("MSE",)):
                    mean, sd, paths = metric_group(root, system, method, metric)
                    data[system, method, metric] = (mean, sd)
                    sources.extend(paths)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.35 if kind == "bar" else 2.65),
                             subplot_kw={"projection": "polar"} if kind == "radar" else {})
    fig.subplots_adjust(left=.025 if kind == "radar" else .085, right=.965 if kind == "radar" else .995,
                        top=.75 if kind == "radar" else .83, bottom=.31 if kind == "loss" else .24,
                        wspace=.65 if kind == "radar" else .50)
    for axis, system in zip(axes, SYSTEMS):
        if kind == "bar":
            means, sds = np.array([data[system, m, "MSE"] for m in methods]).T
            axis.bar(np.arange(len(methods)), means, yerr=sds, capsize=2,
                     color=[COLORS[m] for m in methods], error_kw={"elinewidth": .8, "capthick": .8})
            axis.set_xticks([])
            log_limits(axis, means, sds)
        elif kind == "radar":
            metrics = ("MSE", "MAE", "TCE")
            denominator = np.array([max(data[system, m, k][0] for m in methods) for k in metrics])
            if np.any(denominator <= 0):
                raise ValueError("Radar normalization requires positive metric maxima")
            angles = np.linspace(0, 2*np.pi, 3, endpoint=False)
            angles = np.r_[angles, angles[0]]
            for method in methods:
                mean, sd = np.array([data[system, method, k] for k in metrics]).T / denominator
                mean, sd = np.r_[mean, mean[0]], np.r_[sd, sd[0]]
                axis.plot(angles, mean, label=method, color=COLORS[method], lw=1.8 if method == "NODE-LAC" else 1)
                axis.fill_between(angles, np.maximum(mean-sd, 0), mean+sd, color=COLORS[method], alpha=.08)
            axis.set_xticks(angles[:-1], metrics)
            axis.tick_params(axis="x", pad=1)
            limits = axis.get_ylim()
            axis.set_yticks([.5, 1.])
            axis.set_ylim(limits)
            axis.set_yticklabels([])
            for radius, angle, offset, horizontal, vertical in ((.5, 180, (-3, 0), "right", "center"),
                                                                (1., 90, (0, 2), "center", "top")):
                axis.annotate(f"{radius:.1f}", xy=(np.deg2rad(angle), radius), xytext=offset,
                              textcoords="offset points", ha=horizontal, va=vertical, fontsize=9.5,
                              bbox={"facecolor": "white", "edgecolor": "none", "pad": .05}, zorder=20)
            axis.grid(linewidth=.5, alpha=.5)
        else:
            for method in methods:
                mean, sd = data[system, method]
                axis.plot(np.arange(1, 101), mean, color=COLORS[method], label=method, lw=1.4)
                axis.fill_between(np.arange(1, 101), np.maximum(mean-sd, 1e-14), mean+sd,
                                  color=COLORS[method], alpha=.12)
            log_limits(axis, [data[system, m][0] for m in methods], [data[system, m][1] for m in methods])
            axis.set_xlim(1, 100)
            axis.set_xticks([1, 50, 100])
        axis.set_title(LABELS[system], pad=16 if kind == "radar" else 7)
    handles = ([Patch(facecolor=COLORS[m]) for m in methods] if kind == "bar"
               else axes[0].get_legend_handles_labels()[0])
    if kind == "bar":
        axes[0].set_ylabel("Test MSE")
    if kind == "loss":
        axes[0].set_ylabel("Training trajectory MSE")
        fig.supxlabel("Epoch", x=.54, y=.14)
    fig.legend(handles, methods, loc="lower center", bbox_to_anchor=(.5, -.015 if kind == "loss" else .005),
               ncol=len(methods), frameon=False, handlelength=1.6, columnspacing=1.1)
    stem = {"bar": "table1_bar_chart", "radar": "summary_radar", "loss": "fig2_loss_curves"}[kind]
    details = {"kind": kind, "methods": methods, "systems": SYSTEMS,
               "statistics": {"/".join(key): [np.asarray(v).tolist() for v in value] for key, value in data.items()},
               "normalization": "per-system, per-metric largest displayed mean" if kind == "radar" else None,
               "loss_statistic": "normalized trajectory MSE, unweighted mean of minibatch means" if kind == "loss" else None}
    save_figure(fig, out, stem, sources, details, root)
    return stem

def robustness_figure(root, out, suite):
    fraction = suite == "data-efficiency"
    methods = ("NODE-LAC", "NODE", "SNDE") if fraction else ("NODE-LAC", "NODE")
    xs = (.25, .5, .75, 1.) if fraction else (0., .05, .1, .2)
    statistics, sources = {}, []
    for system in SYSTEMS[:3]:
        for method in methods:
            values = []
            for x in xs:
                main = x == (1. if fraction else 0.)
                variant = None if main else f"{'fraction' if fraction else 'sigma'}_{x}"
                mean, sd, paths = metric_group(root, system, method, suite="main" if main else suite, variant=variant)
                values.append((mean, sd))
                sources.extend(paths)
            statistics[system, method] = np.array(values)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=.095, right=.99, bottom=.23, top=.79, wspace=.34)
    for axis, system in zip(axes, SYSTEMS[:3]):
        for index, method in enumerate(methods):
            mean, sd = statistics[system, method].T
            if fraction:
                axis.errorbar(np.array(xs)*100, mean, yerr=sd, capsize=2.4, elinewidth=.9,
                              marker=("s", "o", "^")[index], markersize=4.3, label=method,
                              color=COLORS[method], linewidth=1.7 if method == "NODE-LAC" else 1.4)
            else:
                axis.bar(np.arange(4)+(index-.5)*.36, mean, .36, yerr=sd, capsize=2.3,
                         label=method, color=COLORS[method], error_kw={"elinewidth": .8, "capthick": .8})
        axis.set_xticks(np.array(xs)*100 if fraction else np.arange(4),
                        ["25", "50", "75", "100"] if fraction else ["0", "0.05", "0.1", "0.2"])
        axis.set_xlim((20, 105) if fraction else (-.65, 3.65))
        axis.set_ylim(0, 1.14*max(float(statistics[system, m].sum(axis=1).max()) for m in methods))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.set_title(LABELS[system], pad=6)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#DADFE5", linewidth=.5, alpha=.65)
    fig.supylabel("Test MSE", x=.005, y=.51, fontsize=10)
    fig.supxlabel("Training subset (%)" if fraction else "Training noise σ (standardized units)", y=.045, fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, 1.015),
               ncol=len(methods), frameon=False, columnspacing=1.7)
    stem = "data_efficiency" if fraction else "noise_robustness"
    save_figure(fig, out, stem, sources, {"suite": suite, "x_values": xs, "methods": methods,
                "statistics": {"/".join(k): v.tolist() for k, v in statistics.items()}}, root)
    return stem

def trajectory_figure(root, out, system):
    arrays, true, times, sources = qualitative_data(root, system)
    if true.shape[1] < 8:
        raise ValueError("Trajectory panels require eight state coordinates")
    displayed = {m: centered_mean(a) if system == "fitzhugh_nagumo" and m in ("PNODE", "CPNODE")
                 else a for m, a in arrays.items()}
    statistics = {m: seed_statistics(a) for m, a in displayed.items()}
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 3.8), sharex=True)
    for site, axis in enumerate(axes.flat):
        for method in ("PNODE", "CPNODE", "NODE", "NODE-LAC"):
            mean, sd = statistics[method]
            axis.fill_between(times, mean[:, site]-sd[:, site], mean[:, site]+sd[:, site],
                              color=STYLES[method]["color"], alpha=.055 if method in ("PNODE", "CPNODE") else .12,
                              linewidth=0, zorder=0)
            axis.plot(times, mean[:, site], label=method, **STYLES[method])
        axis.plot(times, true[:, site], color="black", lw=1.1, label="Ground truth", zorder=6)
        axis.set_title(f"Site {site}", pad=3)
        if site % 4 == 0:
            axis.set_ylabel("state")
        axis.set_xlim(times[0], times[-1])
        axis.set_xticks([times[0], 2.5, times[-1]], ["0", "2.5", f"{times[-1]:.3g}"])
        axis.tick_params(labelsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(m) for m in ("Ground truth", *QUALITATIVE_METHODS)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=5,
               bbox_to_anchor=(.5, .005), frameon=False, columnspacing=1.3, handlelength=2.2)
    fig.tight_layout(rect=(0, .075, 1, 1), h_pad=.8, w_pad=.8)
    stem = f"{system}_trajectories"
    save_figure(fig, out, stem, sources, {"system": system, "test_trajectory_index": 0,
                "coordinates": list(range(8)), "smoothing": {"methods": ["PNODE", "CPNODE"] if system == "fitzhugh_nagumo" else [],
                "window_observations": 5, "alignment": "centered; truncated edges; per seed before mean/SD"},
                "metrics_changed": False}, root)
    return stem

def surface_figure(root, out, system):
    arrays, true, times, sources = qualitative_data(root, system)
    states, errors = surface_statistics(arrays, true)
    n = states[0].shape[1]
    lower, upper = min(float(a.min()) for a in states), max(float(a.max()) for a in states)
    error_max = max(float(a.max()) for a in errors)
    # Avoid singular axes for constant synthetic examples without altering plotted values.
    zupper = upper if upper > lower else lower+1e-12
    eupper = max(error_max, 1e-12)
    xx, tt = np.meshgrid(np.arange(n), times[:50])
    fig = plt.figure(figsize=(7.5, 3.4))
    names = ("Ground truth", *QUALITATIVE_METHODS)
    for row, collection in enumerate((states, errors)):
        for col, values in enumerate(collection):
            axis = fig.add_subplot(2, 5, row*5+col+1, projection="3d")
            axis.plot_surface(xx, tt, values, cmap="viridis" if row == 0 else "magma",
                              vmin=lower if row == 0 else 0, vmax=zupper if row == 0 else eupper,
                              linewidth=0, antialiased=True)
            axis.set_zlim((lower, zupper) if row == 0 else (0, eupper))
            axis.set_xlim(0, n-1)
            axis.set_ylim(times[0], times[49])
            axis.set_xticks([0, n-1])
            axis.set_yticks([times[0], times[49]])
            axis.set_yticklabels(["0", f"{times[49]:.3g}"] if col == 4 else [])
            ticks = (lower, zupper) if row == 0 else (0, eupper)
            axis.set_zticks(ticks)
            axis.set_zticklabels([f"{x:.2f}" if x else "0" for x in ticks] if col == 4 else [])
            axis.tick_params(labelsize=7, pad=-2)
            axis.set_xlabel("site" if row == 1 else "", fontsize=8, labelpad=-7)
            axis.set_ylabel("time" if col == 4 else "", fontsize=8, labelpad=-7)
            if row == 0:
                axis.set_title(names[col], fontsize=9, pad=3)
            axis.view_init(elev=25, azim=-60)
            axis.set_box_aspect((1.4, 1, .7), zoom=1.05)
    fig.text(.012, .715, "State", rotation=90, va="center", ha="center", fontsize=9)
    fig.text(.012, .28, "Absolute error", rotation=90, va="center", ha="center", fontsize=9)
    fig.subplots_adjust(left=.045, right=.975, bottom=.06, top=.93, hspace=.025, wspace=.10)
    stem = f"{system}_3d_surface"
    save_figure(fig, out, stem, sources, {"system": system, "test_trajectory_index": 0, "observations": 50,
                "smoothing": False, "aggregation": "arithmetic seed mean; absolute error of mean prediction",
                "state_color_range": [lower, upper], "error_color_range": [0, error_max]}, root)
    return stem

def render(input_dir, output_dir, kind="all"):
    root, out = Path(input_dir), Path(output_dir)
    tasks = []
    if kind in ("summary", "all"):
        tasks.extend((f"summary/{k}", summary_figure, (root, out, k)) for k in ("bar", "radar", "loss"))
    if kind in ("robustness", "all"):
        tasks.extend((k, robustness_figure, (root, out, k)) for k in ("data-efficiency", "noise"))
    if kind in ("trajectories", "all"):
        tasks.extend((f"trajectories/{s}", trajectory_figure, (root, out, s)) for s in ("fitzhugh_nagumo", "shallow_water"))
    if kind in ("surfaces", "all"):
        tasks.extend((f"surfaces/{s}", surface_figure, (root, out, s)) for s in SYSTEMS[:2])
    report = {"exported": [], "skipped": [], "failed": []}
    style = {"font.family": "DejaVu Serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
             "legend.fontsize": 8, "savefig.dpi": 220, "pdf.fonttype": 42, "ps.fonttype": 42,
             "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        for name, function, args in tasks:
            try:
                target_style = {"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                                "legend.fontsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 10} if name.startswith("summary/") else {}
                with plt.rc_context(target_style):
                    report["exported"].append(function(*args))
            except MissingGroup as error:
                report["skipped"].append({"target": name, "reason": str(error)})
            except (ValueError, KeyError) as error:
                plt.close("all")
                report["failed"].append({"target": name, "reason": str(error)})
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("summary", "robustness", "trajectories", "surfaces", "all"), default="all")
    args = parser.parse_args(argv)
    report = render(args.input_dir, args.output_dir, args.kind)
    print(json.dumps(report, indent=2))
    return 1 if report["failed"] or not report["exported"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
