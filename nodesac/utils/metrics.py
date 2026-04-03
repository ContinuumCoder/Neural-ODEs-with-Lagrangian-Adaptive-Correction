"""Evaluation metrics for NODE-LAC."""

import torch


def mse(pred, true):
    """Mean Squared Error."""
    return (pred - true).pow(2).mean().item()


def mae(pred, true):
    """Mean Absolute Error."""
    return (pred - true).abs().mean().item()


def constraint_error(states, constraint_fn):
    """Constraint Error: CE = (1/T) * sum ||k(x_t)||^2."""
    T = states.shape[-2] if states.dim() >= 2 else 1
    total = 0.0
    if states.dim() == 2:
        # (T, D)
        for t in range(T):
            kx = constraint_fn(states[t:t+1])
            total += kx.pow(2).sum().item()
    elif states.dim() == 3:
        # (batch, T, D)
        for t in range(states.shape[1]):
            kx = constraint_fn(states[:, t])
            total += kx.pow(2).mean().item()
        T = states.shape[1]
    return total / max(T, 1)


def stability_score(states, constraint_fn, threshold=0.1):
    """Fraction of timesteps where constraint is satisfied (within threshold)."""
    satisfied = 0
    total = 0
    if states.dim() == 3:
        for t in range(states.shape[1]):
            kx = constraint_fn(states[:, t])
            dist = kx.pow(2).sum(dim=-1)
            satisfied += (dist < threshold).float().sum().item()
            total += dist.shape[0]
    elif states.dim() == 2:
        for t in range(states.shape[0]):
            kx = constraint_fn(states[t:t+1])
            dist = kx.pow(2).sum(dim=-1)
            satisfied += (dist < threshold).float().sum().item()
            total += 1
    return satisfied / max(total, 1)


def temporal_coherence_error(pred, true):
    r"""Temporal Coherence Error (TCE): measures whether predicted dynamics
    match true dynamics evolution.
    TCE = (1/NT) * sum |(\hat{x}_{t+1} - \hat{x}_t) - (x_{t+1} - x_t)|^2
    """
    if pred.dim() == 2:
        # (T, D)
        pred_diff = pred[1:] - pred[:-1]
        true_diff = true[1:] - true[:-1]
        return (pred_diff - true_diff).pow(2).mean().item()
    elif pred.dim() == 3:
        # (batch, T, D)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]
        return (pred_diff - true_diff).pow(2).mean().item()
    return 0.0


def compute_all_metrics(pred, true, constraint_fn):
    """Compute all metrics and return dict."""
    return {
        'MSE': mse(pred, true),
        'MAE': mae(pred, true),
        'TCE': temporal_coherence_error(pred, true),
        'Stability': stability_score(pred, constraint_fn),
    }
