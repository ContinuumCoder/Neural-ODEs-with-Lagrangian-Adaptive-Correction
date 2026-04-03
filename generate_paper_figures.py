#!/usr/bin/env python3
"""Generate publication-quality figures from experiment data.
Run AFTER run_all_figures.py completes.

Style: clean, professional, Nature/Science quality.
- Consistent color scheme across all figures
- Proper fonts, labels, legends
- Both PNG (300dpi) and PDF output
"""
import json, numpy as np, os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Style ──
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
})

# Color scheme
COLORS = {
    'NODE-LAC': '#2166AC',   # deep blue
    'NODE': '#F4A582',       # salmon
    'SNDE': '#4DAF4A',       # green
    'PORT-HJNN': '#984EA3',  # purple
    'ConCerNet': '#FF7F00',  # orange
    'PNODE': '#A65628',      # brown
    'CPNODE': '#E41A1C',     # red
    'HNN': '#999999',        # gray
    'SymODEN': '#CCCCCC',    # light gray
    'CLNN': '#666666',       # dark gray
}

SYSTEM_LABELS = {
    'fitzhugh_nagumo': 'FitzHugh-Nagumo',
    'lotka_volterra': 'Lotka-Volterra',
    'shallow_water': 'Shallow Water',
    'franka_robot': 'Robot Arm',
}

EXP = Path('experiments')
FIGS = Path('figures')
FIGS.mkdir(exist_ok=True)


def fig2_loss_curves():
    """Figure 2: Training loss curves — NODE-LAC vs baselines."""
    data = json.load(open(EXP / 'fig2_loss_curves' / 'loss_data.json'))

    methods_to_plot = ['NODE-LAC', 'NODE', 'SNDE', 'ConCerNet', 'PNODE']
    linestyles = {'NODE-LAC': '-', 'NODE': '--', 'SNDE': ':', 'ConCerNet': '-.', 'PNODE': '--'}

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
    for i, sys_name in enumerate(['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water', 'franka_robot']):
        ax = axes[i]
        d = data[sys_name]

        for m in methods_to_plot:
            if m not in d:
                continue
            epochs = range(1, len(d[m]) + 1)
            lw = 2.5 if m == 'NODE-LAC' else 1.3
            ax.semilogy(epochs, d[m], linestyles.get(m, '-'), color=COLORS.get(m, '#888'),
                       label=m, linewidth=lw, alpha=0.85)

        ax.set_title(SYSTEM_LABELS[sys_name], fontweight='bold')
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.set_ylabel('Training Loss (log scale)')
        ax.legend(fontsize=7, framealpha=0.9, edgecolor='none')

    plt.tight_layout(w_pad=1.5)
    plt.savefig(FIGS / 'fig2_loss_curves.png')
    plt.savefig(FIGS / 'fig2_loss_curves.pdf')
    plt.close()
    print('  Fig 2: loss curves ✓')


def table1_bar_chart():
    """Table 1 visualization: grouped bar chart of MSE across methods."""
    data = json.load(open(EXP / 'table1_main' / 'table1_data.json'))

    top_methods = ['NODE-LAC', 'SNDE', 'PORT-HJNN', 'NODE', 'ConCerNet', 'PNODE']

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, sys_name in enumerate(['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water', 'franka_robot']):
        ax = axes[i]
        d = data[sys_name]
        methods = [m for m in top_methods if m in d]
        mses = [d[m].get('MSE', 0) for m in methods]
        colors = [COLORS.get(m, '#888888') for m in methods]

        bars = ax.bar(range(len(methods)), mses, color=colors, edgecolor='white', linewidth=0.5)

        # Highlight winner
        min_idx = np.argmin(mses)
        bars[min_idx].set_edgecolor('gold')
        bars[min_idx].set_linewidth(2.5)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
        ax.set_title(SYSTEM_LABELS[sys_name], fontweight='bold')
        if i == 0:
            ax.set_ylabel('Test MSE')
        ax.set_yscale('log')

    plt.tight_layout(w_pad=1.0)
    plt.savefig(FIGS / 'table1_bar_chart.png')
    plt.savefig(FIGS / 'table1_bar_chart.pdf')
    plt.close()
    print('  Table 1: bar chart ✓')


def table3_ablation():
    """Table 3: Ablation study visualization."""
    if not (EXP / 'table3_ablation' / 'table3_data.json').exists():
        print('  Table 3: data not ready, skip'); return
    data = json.load(open(EXP / 'table3_ablation' / 'table3_data.json'))

    variants = ['Full NODE-LAC', 'NoGainNet', 'NoConstraintLoss', 'NoCorrection']
    variant_colors = ['#2166AC', '#E41A1C', '#FF7F00', '#999999']

    ablation_systems = ['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, sys_name in enumerate(ablation_systems):
        ax = axes[i]
        if sys_name not in data:
            ax.set_visible(False); continue
        d = data[sys_name]
        mses = [d.get(v, {}).get('MSE', 0) for v in variants]
        bars = ax.bar(range(len(variants)), mses, color=variant_colors, edgecolor='white')
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(['Full', 'No Gain', 'No Constr', 'No Corr'], rotation=25, ha='right', fontsize=8)
        ax.set_title(SYSTEM_LABELS[sys_name], fontweight='bold')
        if i == 0: ax.set_ylabel('Test MSE')

    plt.tight_layout()
    plt.savefig(FIGS / 'table3_ablation.png')
    plt.savefig(FIGS / 'table3_ablation.pdf')
    plt.close()
    print('  Table 3: ablation ✓')


def reviewer_data_efficiency():
    """Reviewer: Data efficiency curves (3 systems, NODE vs SNDE vs NODE-LAC)."""
    if not (EXP / 'reviewer_data_efficiency' / 'data_efficiency.json').exists():
        print('  Data efficiency: data not ready, skip'); return
    data = json.load(open(EXP / 'reviewer_data_efficiency' / 'data_efficiency.json'))

    fracs = [0.25, 0.5, 0.75, 1.0]
    systems_3 = ['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water']
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for i, sys_name in enumerate(systems_3):
        ax = axes[i]
        if sys_name not in data: continue
        d = data[sys_name]
        node_mses = [d[str(f)]['NODE_MSE'] for f in fracs]
        sac_mses = [d[str(f)]['SAC_MSE'] for f in fracs]

        ax.plot([f*100 for f in fracs], node_mses, 'o-', color=COLORS['NODE'], label='NODE', markersize=6)
        # SNDE if available
        if 'SNDE_MSE' in d[str(fracs[0])]:
            snde_mses = [d[str(f)]['SNDE_MSE'] for f in fracs]
            ax.plot([f*100 for f in fracs], snde_mses, '^--', color=COLORS['SNDE'], label='SNDE', markersize=6)
        ax.plot([f*100 for f in fracs], sac_mses, 's-', color=COLORS['NODE-LAC'], label='NODE-LAC',
                markersize=7, linewidth=2.5, zorder=5)

        ax.set_title(SYSTEM_LABELS[sys_name], fontweight='bold')
        ax.set_xlabel('Training Data (%)')
        if i == 0: ax.set_ylabel('Test MSE')
        ax.legend(framealpha=0.9, edgecolor='none')

    plt.tight_layout(w_pad=1.5)
    plt.savefig(FIGS / 'reviewer_data_efficiency.png')
    plt.savefig(FIGS / 'reviewer_data_efficiency.pdf')
    plt.close()
    print('  Data efficiency: figure ✓')


def reviewer_noise():
    """Reviewer: Noise robustness bar chart."""
    if not (EXP / 'reviewer_noise' / 'noise_robustness.json').exists():
        print('  Noise: data not ready, skip'); return
    data = json.load(open(EXP / 'reviewer_noise' / 'noise_robustness.json'))

    sigmas = [0.0, 0.05, 0.1, 0.2]
    systems_3 = ['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    x = np.arange(len(sigmas))
    w = 0.35

    for i, sys_name in enumerate(systems_3):
        ax = axes[i]
        if sys_name not in data: continue
        d = data[sys_name]
        node_mses = [d[str(sg)]['NODE_MSE'] for sg in sigmas]
        sac_mses = [d[str(sg)]['SAC_MSE'] for sg in sigmas]

        ax.bar(x - w/2, node_mses, w, color=COLORS['NODE'], label='NODE', edgecolor='white')
        ax.bar(x + w/2, sac_mses, w, color=COLORS['NODE-LAC'], label='NODE-LAC', edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels([f'σ={sg}' for sg in sigmas])
        ax.set_title(SYSTEM_LABELS[sys_name], fontweight='bold')
        if i == 0: ax.set_ylabel('Test MSE')
        ax.legend(framealpha=0.9, edgecolor='none')

    plt.tight_layout(w_pad=1.5)
    plt.savefig(FIGS / 'reviewer_noise_robustness.png')
    plt.savefig(FIGS / 'reviewer_noise_robustness.pdf')
    plt.close()
    print('  Noise robustness: figure ✓')


def summary_radar():
    """Summary: radar chart showing NODE-LAC vs top baselines across metrics."""
    ckpt = json.load(open('results/checkpoint.json'))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), subplot_kw=dict(projection='polar'))
    top_methods = ['NODE-LAC', 'SNDE', 'NODE', 'ConCerNet', 'PNODE', 'CPNODE']
    metrics = ['MSE', 'MAE', 'TCE']

    for i, sys_name in enumerate(['fitzhugh_nagumo', 'lotka_volterra', 'shallow_water', 'franka_robot']):
        ax = axes[i]
        methods_data = defaultdict(lambda: defaultdict(list))
        for k, v in ckpt['results'].items():
            if sys_name in k and '|main|' in k:
                m = k.split('|')[1]
                if m in top_methods:
                    for met in metrics:
                        methods_data[m][met].append(v.get(met, 0))

        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        for method in top_methods:
            if method in methods_data:
                vals = [np.mean(methods_data[method][met]) for met in metrics]
                # Normalize to [0,1] relative to worst
                max_vals = [max(np.mean(methods_data[m][met]) for m in methods_data) for met in metrics]
                norm_vals = [v/mx if mx > 0 else 0 for v, mx in zip(vals, max_vals)]
                norm_vals += norm_vals[:1]
                ax.plot(angles, norm_vals, color=COLORS[method], label=method, linewidth=2 if method == 'NODE-LAC' else 1.2)
                ax.fill(angles, norm_vals, color=COLORS[method], alpha=0.1 if method == 'NODE-LAC' else 0.03)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_title(SYSTEM_LABELS[sys_name], fontweight='bold', pad=20)
        if i == 0: ax.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1), fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGS / 'summary_radar.png')
    plt.savefig(FIGS / 'summary_radar.pdf')
    plt.close()
    print('  Summary radar ✓')


if __name__ == '__main__':
    print('=== Generating Publication Figures ===')
    fig2_loss_curves()
    table1_bar_chart()
    table3_ablation()
    reviewer_data_efficiency()
    reviewer_noise()
    summary_radar()
    print(f'\nAll figures saved to {FIGS}/')
    print('Files:', list(FIGS.glob('*')))
