"""NODE-LAC: Neural ODE with Lagrangian Adaptive Correction.

Core idea: dx/dt = f(x) - kappa(x) * gate(||k||) * grad||k(x)||^2

kappa(x) is a state-dependent gain learned via one-step lookahead.
NODE is trained with closed-loop dynamics — the correction acts as
constraint-aware regularization, forcing NODE to learn dynamics that
co-adapt with the correction term.

"LAC" = Lagrangian Adaptive Correction:
  - Soft: softplus gain + effort regularization (minimal intervention)
  - Adaptive: state-dependent kappa(x) + dual variable mu (Lagrangian dual variable)
  - Correction: constraint-gradient correction term
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .neural_ode import NeuralODE, MLP, euler_integrate


class GainNet(nn.Module):
    """Deterministic gain network: state -> kappa(x) > 0."""

    def __init__(self, state_dim, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.gain_head = nn.Linear(prev, 1)

    def forward(self, state):
        h = self.backbone(state)
        return F.softplus(self.gain_head(h))


class ClosedLoopDynamics(nn.Module):
    """dx/dt = f_theta(x) - kappa(x) * gate * grad||k(x)||^2

    gain is detached (no_grad) — NODE trains through f(x) with the correction
    as a fixed perturbation. GainNet is trained separately via one-step lookahead.
    """

    def __init__(self, node_net, gain_net, manifold, correction_scale=1.0):
        super().__init__()
        self.node_net = node_net
        self.gain_net = gain_net
        self.manifold = manifold
        self.correction_scale = correction_scale

    def forward(self, t, x):
        f_x = self.node_net(x)

        try:
            with torch.enable_grad():
                x_d = x.detach().clone().requires_grad_(True)
                kx = self.manifold.k(x_d)
                grad_k = torch.autograd.grad(kx.pow(2).sum(), x_d)[0]

            if grad_k is not None:
                grad_clipped = grad_k.detach().clamp(-5.0, 5.0)
                k_norm = kx.detach().pow(2).sum(dim=-1, keepdim=True).sqrt()
                gate = torch.tanh(k_norm)

                with torch.no_grad():
                    gain = self.gain_net(x)
                correction = self.correction_scale * gain * gate * grad_clipped
                return f_x - correction
        except Exception:
            pass
        return f_x


class NODESAC(nn.Module):
    """NODE-LAC: Neural ODE with Lagrangian Adaptive Correction.

    Training:
      - NODE: multiple shooting with closed-loop dynamics (correction as regularizer)
      - GainNet: one-step lookahead minimizing constraint violation (direct gradient)
      - mu: dual ascent on constraint violation (Lagrangian dual variable)

    The coupled training is key: NODE co-adapts with the correction term,
    learning dynamics that work synergistically with the gain.
    """

    def __init__(self, state_dim, manifold, action_dim=None,
                 node_hidden=(64, 64), sac_hidden=(64, 64),
                 gamma=0.99, tau=0.005, alpha_init=0.1,
                 lambda1=1.0, lambda2=0.05, reg_lambda=1e-4,
                 solver='euler', device=None, dtype=torch.float64):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        self.state_dim = state_dim
        self.manifold = manifold
        self.lambda_effort = lambda2
        self.device = device
        self.dtype = dtype
        self.solver = solver

        self.node = NeuralODE(state_dim, node_hidden, solver=solver).to(device=device, dtype=dtype)
        self.gain_net = GainNet(state_dim, sac_hidden).to(device=device, dtype=dtype)

        self.node_optim = torch.optim.Adam(self.node.parameters(), lr=6e-3, weight_decay=reg_lambda)
        self.gain_optim = torch.optim.Adam(self.gain_net.parameters(), lr=1e-3)

        # Dual variable mu: Lagrangian "critic"
        self.log_mu = torch.tensor(np.log(0.1), dtype=dtype, device=device, requires_grad=True)
        self.mu_optim = torch.optim.Adam([self.log_mu], lr=1e-2)
        self.constr_target = 0.01

    def _make_dynamics(self, correction_scale=1.0):
        return ClosedLoopDynamics(
            self.node.f, self.gain_net, self.manifold,
            correction_scale=correction_scale)

    def predict(self, x0, t_span, correction_scale=1.0):
        dynamics = self._make_dynamics(correction_scale)
        dynamics.eval()
        return euler_integrate(dynamics, x0, t_span)

    def tune_correction_scale(self, x_val, t_span, alphas=None):
        if alphas is None:
            alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
        self.eval()
        best_alpha, best_mse = 1.0, float('inf')
        x0 = x_val[:, 0]
        with torch.no_grad():
            for a in alphas:
                p = self.predict(x0, t_span, correction_scale=a).permute(1, 0, 2)
                mse = (p - x_val).pow(2).mean().item()
                if mse < best_mse:
                    best_mse = mse
                    best_alpha = a
        return best_alpha, best_mse

    def _shooting_loss(self, func, batch_x, t_span, window_size=3):
        """Multiple shooting: predict from each window's true start state."""
        B, T, D = batch_x.shape
        total_loss = 0.0
        n_windows = 0
        for start in range(0, T - 1, max(1, window_size - 1)):
            end = min(start + window_size, T)
            if end - start < 2:
                continue
            x0_w = batch_x[:, start]
            t_w = t_span[start:end]
            x_pred_w = euler_integrate(func, x0_w, t_w).permute(1, 0, 2)
            x_true_w = batch_x[:, start:end]
            total_loss = total_loss + (x_pred_w - x_true_w).pow(2).mean()
            n_windows += 1
        return total_loss / max(n_windows, 1)

    def _gain_loss_onestep(self, batch_x, t_span):
        """One-step lookahead training for GainNet (fully vectorized).

        Flattens all (batch, time) pairs into one batch for parallel computation.
        No Python loop over timesteps.
        """
        B, T, D = batch_x.shape
        if T < 2:
            zero = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            return zero, zero

        # Flatten all states at t=0..T-2 → (B*(T-1), D)
        x_all = batch_x[:, :-1].reshape(-1, D).detach()  # (B*(T-1), D)
        dt_all = (t_span[1:] - t_span[:-1]).unsqueeze(0).expand(B, -1).reshape(-1, 1)  # (B*(T-1), 1)

        # Frozen NODE
        with torch.no_grad():
            f_x = self.node.f(x_all)

        # Constraint gradient + gate (vectorized autograd)
        with torch.enable_grad():
            x_d = x_all.detach().clone().requires_grad_(True)
            kx = self.manifold.k(x_d)
            grad_k = torch.autograd.grad(kx.pow(2).sum(), x_d)[0]

        grad_clipped = grad_k.detach().clamp(-5.0, 5.0)
        k_norm = kx.detach().pow(2).sum(dim=-1, keepdim=True).sqrt()
        gate = torch.tanh(k_norm)

        # Differentiable gain
        gain = self.gain_net(x_all)

        # One-step corrected prediction
        correction = gain * gate * grad_clipped
        x_next = x_all + dt_all * (f_x - correction)

        # Constraint violation at corrected next states
        constr = self.manifold.distance(x_next).mean()
        effort = gain.pow(2).mean()

        return constr, effort

    def train_epoch_warmup(self, dataloader, t_span):
        """Phase 1: Pretrain NODE only (no correction)."""
        self.node.train()
        total_loss, count = 0, 0
        for batch_x in dataloader:
            batch_x = batch_x.to(device=self.device, dtype=self.dtype)
            loss = self._shooting_loss(self.node, batch_x, t_span, window_size=3)
            self.node_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.node.parameters(), 1.0)
            self.node_optim.step()
            total_loss += loss.item() * batch_x.shape[0]
            count += batch_x.shape[0]
        return total_loss / max(count, 1)

    def train_epoch_alternating(self, dataloader, t_span, update_ratio=2,
                                constr_weight=0.1, correction_ramp=1.0):
        """Phase 2: Coupled NODE + GainNet training.

        Step A: NODE trains via shooting on closed-loop dynamics
                (gain is detached; correction acts as regularizer for NODE)
        Step B: GainNet trains via one-step lookahead
                (NODE frozen; direct per-state gradient for gain)
        Step C: Dual ascent on mu
        """
        self.node.train()
        self.gain_net.train()
        total_loss, count = 0, 0

        for batch_x in dataloader:
            batch_x = batch_x.to(device=self.device, dtype=self.dtype)
            B, T, D = batch_x.shape

            mu = self.log_mu.exp().clamp(0.01, 1.0)

            # --- Step A: NODE with closed-loop dynamics ---
            dynamics = self._make_dynamics(correction_scale=correction_ramp)
            traj_loss = self._shooting_loss(dynamics, batch_x, t_span, window_size=3)

            # Constraint from closed-loop trajectory
            with torch.no_grad():
                x_pred = euler_integrate(dynamics, batch_x[:, 0], t_span).permute(1, 0, 2)
            constr_node = self.manifold.distance(x_pred).mean()

            node_loss = traj_loss + mu.detach() * constr_node
            self.node_optim.zero_grad()
            node_loss.backward()
            nn.utils.clip_grad_norm_(self.node.parameters(), 1.0)
            self.node_optim.step()

            # --- Step B: GainNet via one-step lookahead ---
            constr_gain, effort = self._gain_loss_onestep(batch_x, t_span)
            gain_loss = mu.detach() * constr_gain + self.lambda_effort * effort

            self.gain_optim.zero_grad()
            gain_loss.backward()
            nn.utils.clip_grad_norm_(self.gain_net.parameters(), 1.0)
            self.gain_optim.step()

            # --- Step C: Dual ascent ---
            dual_loss = -self.log_mu * (constr_node.detach() - self.constr_target)
            self.mu_optim.zero_grad()
            dual_loss.backward()
            self.mu_optim.step()

            total_loss += traj_loss.item() * B
            count += B

        info = {
            'mu': self.log_mu.exp().item(),
            'constr': constr_gain.item() if count > 0 else 0,
            'effort': effort.item() if count > 0 else 0,
        }
        return total_loss / max(count, 1), info

    @staticmethod
    def _adaptive_schedule(warmup_epochs, alt_epochs, finetune_epochs):
        total = warmup_epochs + alt_epochs + finetune_epochs
        w_frac = min(0.8, 0.2 + 0.004 * total)
        a_frac = max(0.15, 0.5 - 0.002 * total)
        f_frac = 1.0 - w_frac - a_frac
        we = max(3, int(total * w_frac))
        ae = max(3, int(total * a_frac))
        fe = max(2, total - we - ae)
        return we, ae, fe

    def train_full(self, dataloader, t_span,
                   warmup_epochs=20, alt_epochs=60, finetune_epochs=20,
                   update_ratio=2, lr_finetune=3e-4, verbose=True,
                   auto_schedule=True, max_correction=1.0):
        if auto_schedule:
            warmup_epochs, alt_epochs, finetune_epochs = self._adaptive_schedule(
                warmup_epochs, alt_epochs, finetune_epochs)

        history = {'node_loss': [], 'phase': []}

        if verbose:
            total = warmup_epochs + alt_epochs + finetune_epochs
            print(f"=== Phase 1: Warm-up ({warmup_epochs}/{total} ep) ===", flush=True)
        for epoch in range(warmup_epochs):
            loss = self.train_epoch_warmup(dataloader, t_span)
            history['node_loss'].append(loss)
            history['phase'].append(1)
            if verbose and (epoch + 1) % max(1, warmup_epochs // 3) == 0:
                print(f"  Ep {epoch+1}/{warmup_epochs}, Loss: {loss:.6f}", flush=True)

        phase23_total = alt_epochs + finetune_epochs
        node_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.node_optim, T_max=phase23_total, eta_min=lr_finetune)
        gain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.gain_optim, T_max=phase23_total, eta_min=lr_finetune * 0.3)

        if verbose:
            print(f"=== Phase 2: Joint ({alt_epochs}/{total} ep) ===", flush=True)
        for epoch in range(alt_epochs):
            # Gradual correction ramp: 0 → max_correction over first half of Phase 2
            ramp = min(max_correction, max_correction * epoch / max(1, alt_epochs // 2))
            loss, info = self.train_epoch_alternating(
                dataloader, t_span, update_ratio, correction_ramp=ramp)
            node_scheduler.step()
            gain_scheduler.step()
            history['node_loss'].append(loss)
            history['phase'].append(2)
            if verbose and (epoch + 1) % max(1, alt_epochs // 3) == 0:
                print(f"  Ep {epoch+1}/{alt_epochs}, Loss: {loss:.6f}, "
                      f"mu={info['mu']:.3f}, CE={info['constr']:.4f}, "
                      f"effort={info['effort']:.4f}, ramp={ramp:.2f}", flush=True)

        if verbose:
            print(f"=== Phase 3: Fine-tune ({finetune_epochs}/{total} ep) ===", flush=True)
        for epoch in range(finetune_epochs):
            loss, info = self.train_epoch_alternating(
                dataloader, t_span, update_ratio, correction_ramp=max_correction)
            node_scheduler.step()
            gain_scheduler.step()
            history['node_loss'].append(loss)
            history['phase'].append(3)
            if verbose and (epoch + 1) % max(1, finetune_epochs // 2) == 0:
                print(f"  Ep {epoch+1}/{finetune_epochs}, Loss: {loss:.6f}", flush=True)

        return history
