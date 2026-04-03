"""Visualization utilities."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_trajectories(true, pred, title='Trajectories', save_path=None, dims=(0, 1)):
    """Plot true vs predicted trajectories for selected dimensions."""
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4))
    if len(dims) == 1:
        axes = [axes]
    true_np = true.detach().cpu().numpy() if isinstance(true, torch.Tensor) else true
    pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred

    for i, d in enumerate(dims):
        axes[i].plot(true_np[:, d], label='True', linewidth=2)
        axes[i].plot(pred_np[:, d], '--', label='Pred', linewidth=2)
        axes[i].set_title(f'Dim {d}')
        axes[i].legend()
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(losses_dict, title='Training Loss', save_path=None):
    """Plot training loss curves for multiple methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(title)
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_constraint_violation(times, violations_dict, title='Constraint Violation', save_path=None):
    """Plot constraint violation over time for multiple methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    t_np = times.cpu().numpy() if isinstance(times, torch.Tensor) else times
    for name, viol in violations_dict.items():
        v_np = viol.cpu().numpy() if isinstance(viol, torch.Tensor) else viol
        ax.plot(t_np[:len(v_np)], v_np, label=name)
    ax.set_xlabel('Time')
    ax.set_ylabel('Constraint Error')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bar(results, metric='MSE', title=None, save_path=None):
    """Bar chart comparing methods on a metric."""
    methods = list(results.keys())
    values = [results[m][metric] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, values)
    # Highlight NODE-LAC
    for i, m in enumerate(methods):
        if 'LAC' in m:
            bars[i].set_color('tab:red')
    ax.set_ylabel(metric)
    ax.set_title(title or f'{metric} Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_spatiotemporal(data, title='Spatiotemporal', save_path=None):
    """2D heatmap for PDE solutions."""
    d_np = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(d_np.T, aspect='auto', origin='lower', cmap='RdBu_r')
    ax.set_xlabel('Time')
    ax.set_ylabel('Space')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
