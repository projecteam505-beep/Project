"""
Classical Gaussian-Bernoulli Conditional RBM (CRBM).

Reimplements the model of Taylor & Hinton (2009) exactly as derived in
Hellstern et al. (2026), Section 4.2 (arXiv:2607.24065):

  Energy:      E(v,h|u) = sum_i (v_i - b_eff,i)^2 / (2 sigma^2)
                          - sum_j c_eff,j h_j
                          - sum_ij (v_i / sigma) W_ij h_j
  Effective biases:
      b_eff(u) = b + u @ W_cv
      c_eff(u) = c + u @ W_ch
  Conditionals:
      p(h_j=1 | v,u) = sigmoid( (1/sigma) sum_i v_i W_ij + c_eff,j )
      p(v | h,u)      = N( b_eff + sigma * W^T h,  sigma^2 I )

Trained with Contrastive Divergence-k (CD-k), Eqs. (11)-(16) of the base
paper. Forecasting uses the deterministic mean-field iteration, Eqs.
(17)-(19).

This module is intentionally dependency-light (plain PyTorch tensors, no
autograd needed for the CD updates themselves) so it can serve as the
foundation that QCRBM extends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def xavier_uniform(fan_in: int, fan_out: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return (torch.rand(fan_in, fan_out, generator=generator) * 2 - 1) * a


@dataclass
class CRBMConfig:
    V: int = 1          # visible dimension (univariate return series)
    H: int = 2           # hidden units
    U: int = 10          # context / lag window length
    k: int = 3           # CD steps
    sigma: float = 0.1   # visible Gaussian std (fixed, as in the base paper)
    lr: float = 1e-3     # classical learning rate eta_cl
    weight_decay: float = 0.0
    seed: int = 0


class CRBM:
    """Gaussian-Bernoulli Conditional RBM with CD-k training and mean-field
    autoregressive forecasting."""

    def __init__(self, cfg: CRBMConfig):
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)

        # Plain tensors (not nn.Parameter / no autograd needed): CD-k updates
        # are computed analytically in cd_step() below, exactly following the
        # base paper's closed-form gradients (Eqs. 11-16). We still support
        # in-place `+=` updates via apply_grads().
        self.W = xavier_uniform(cfg.V, cfg.H, g)
        self.Wcv = xavier_uniform(cfg.U, cfg.V, g)
        self.Wch = xavier_uniform(cfg.U, cfg.H, g)
        self.b = torch.zeros(cfg.V)
        self.c = torch.zeros(cfg.H)

    # -- effective biases (Eq. 7) -------------------------------------------------
    def effective_biases(self, u: torch.Tensor):
        b_eff = self.b + u @ self.Wcv     # (batch, V)
        c_eff = self.c + u @ self.Wch     # (batch, H)
        return b_eff, c_eff

    # -- conditionals (Eqs. 8-9) --------------------------------------------------
    def p_h_given_v(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        _, c_eff = self.effective_biases(u)
        logits = (v / self.cfg.sigma) @ self.W + c_eff
        return torch.sigmoid(logits)

    def sample_h(self, v: torch.Tensor, u: torch.Tensor):
        p = self.p_h_given_v(v, u)
        h = torch.bernoulli(p)
        return p, h

    def mean_v_given_h(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        b_eff, _ = self.effective_biases(u)
        return b_eff + self.cfg.sigma * (h @ self.W.T)

    def sample_v(self, h: torch.Tensor, u: torch.Tensor):
        mean = self.mean_v_given_h(h, u)
        v = mean + self.cfg.sigma * torch.randn_like(mean)
        return mean, v

    # -- Contrastive Divergence (Eqs. 11-16) --------------------------------------
    def cd_step(self, v0: torch.Tensor, u0: torch.Tensor):
        """Run CD-k and return per-parameter gradient *ascent* directions
        (i.e. already the direction that increases log-likelihood)."""
        cfg = self.cfg
        batch = v0.shape[0]

        p_h0, h0 = self.sample_h(v0, u0)

        vk, hk = v0, h0
        for _ in range(cfg.k):
            _, vk = self.sample_v(hk, u0)
            p_hk, hk = self.sample_h(vk, u0)

        sigma = cfg.sigma
        dW = ((v0 / sigma).T @ p_h0 - (vk / sigma).T @ p_hk) / batch
        db = (v0 - vk).mean(dim=0) / (sigma ** 2)
        dc = (p_h0 - p_hk).mean(dim=0)
        dWcv = (u0.T @ (v0 - vk)) / (batch * sigma ** 2)
        dWch = (u0.T @ (p_h0 - p_hk)) / batch

        grads = {"W": dW, "b": db, "c": dc, "Wcv": dWcv, "Wch": dWch}
        recon_mse = ((v0 - vk) ** 2).mean().item()
        return grads, recon_mse

    def apply_grads(self, grads: dict):
        cfg = self.cfg
        for name, grad in grads.items():
            param = getattr(self, name)
            # weight decay (L2) is subtracted from the ascent direction
            decayed = grad - cfg.weight_decay * param
            setattr(self, name, param + cfg.lr * decayed)

    def train_step(self, v0: torch.Tensor, u0: torch.Tensor):
        grads, recon_mse = self.cd_step(v0, u0)
        self.apply_grads(grads)
        return recon_mse

    # -- mean-field forecasting (Eqs. 17-19) --------------------------------------
    @torch.no_grad()
    def mean_field_predict(self, u: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
        b_eff, c_eff = self.effective_biases(u)
        v = b_eff.clone()
        for _ in range(n_iter):
            h = torch.sigmoid((v / self.cfg.sigma) @ self.W + c_eff)
            v = b_eff + self.cfg.sigma * (h @ self.W.T)
        return v

    @torch.no_grad()
    def forecast_autoregressive(self, u0: torch.Tensor, horizon: int, n_mf_iter: int = 5):
        """Multi-step forecast: after each step, slide the context window
        forward by dropping the oldest value and appending the prediction
        (Eq. 19). Assumes V=1 (univariate)."""
        preds = []
        u = u0.clone()
        for _ in range(horizon):
            v_star = self.mean_field_predict(u, n_iter=n_mf_iter)
            preds.append(v_star)
            u = torch.cat([u[:, 1:], v_star[:, :1]], dim=1)
        return torch.cat(preds, dim=1)  # (batch, horizon)

    def parameters(self):
        return [self.W, self.Wcv, self.Wch, self.b, self.c]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # quick smoke test on random data
    torch.manual_seed(0)
    cfg = CRBMConfig(V=1, H=2, U=10, k=3, sigma=0.1, lr=1e-3)
    model = CRBM(cfg)

    n = 500
    u = torch.randn(n, cfg.U) * 0.1
    v = torch.randn(n, cfg.V) * 0.1

    for epoch in range(20):
        mse = model.train_step(v, u)
    print("final recon MSE:", mse)
    print("num params:", model.num_params())

    forecast = model.forecast_autoregressive(u[:5], horizon=5)
    print("forecast shape:", forecast.shape)
