"""
Lag-Feature QFeatureQRBM.

Reimplements Section 4.5 of Hellstern et al. (2026), arXiv:2607.24065.

Instead of encoding the current (v, u) state, this model encodes a dedicated
lag vector `l` (the L most recent values) through a small PQC with
n_q = ceil(log2(L)) qubits, producing compact temporal features z(l; theta)
that are projected into the hidden and visible dimensions via learnable
linear heads Q_h, Q_v (Eqs. 34-38).

Per-unit gate logits alpha_h, alpha_v (vectors, not scalars) scale the
quantum contributions with a bounded sigmoid gain, so the gate cannot
exceed magnitude 1 in either direction:

    p(h_j=1 | v,u,l) = sigmoid( v~ W_vh + c + sigmoid(alpha_h,j) * q_h,j )
    E[v_i | h,u,l]   = b_i + sigma_i (h W_vh^T)_i + sigmoid(alpha_v,i) * q_v,i

Training combines the classical CD-k update with a quantum loss that is a
weighted sum of reconstruction MSE, a knowledge-distillation "alignment"
term against the classical-only (no-quantum) reconstruction, and an L2
penalty on the projection heads (Eqs. 39-41).

This implementation uses a fixed scalar visible sigma (rather than the
paper's learnable per-visible softplus parameterization) to keep the
real-data reimplementation tractable within the project's time budget --
documented as a deliberate simplification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

try:
    import pennylane as qml
except ImportError:  # pragma: no cover
    qml = None

from .crbm import xavier_uniform


@dataclass
class QFeatureQRBMConfig:
    V: int = 1
    H: int = 2
    U: int = 10
    L: int = 10           # lag window length (Sec 4.5.1)
    k: int = 3
    sigma: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_layers: int = 1      # PQC depth
    lambda_align: float = 1.0   # alignment loss weight lambda_a
    eta_Q: float = 1e-3
    lambda_Q: float = 1e-3
    n_warmup: int = 2       # epochs before alignment loss is enabled
    seed: int = 0


class QFeatureQRBM:
    def __init__(self, cfg: QFeatureQRBMConfig):
        if qml is None:
            raise ImportError("pennylane is required for QFeatureQRBM")

        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)

        self.n_q = max(1, math.ceil(math.log2(max(cfg.L, 2))))
        self.dim = 2 ** self.n_q

        # classical parameters
        self.Wvh = xavier_uniform(cfg.V, cfg.H, g)
        self.b = torch.zeros(cfg.V)
        self.c = torch.zeros(cfg.H)

        # quantum parameters
        self.theta = torch.nn.Parameter(
            (torch.rand(cfg.n_layers, self.n_q, 3, generator=g) * 2 - 1) * np.pi
        )
        self.Qh = torch.nn.Parameter(torch.randn(self.n_q, cfg.H, generator=g) * 0.1)
        self.Qv = torch.nn.Parameter(torch.randn(self.n_q, cfg.V, generator=g) * 0.1)
        self.alpha_h = torch.nn.Parameter(torch.zeros(cfg.H))
        self.alpha_v = torch.nn.Parameter(torch.zeros(cfg.V))

        self.dev = qml.device("default.qubit", wires=self.n_q)
        self._build_qnode()

        self.q_optimizer = torch.optim.Adam(
            [self.theta, self.Qh, self.Qv, self.alpha_h, self.alpha_v],
            lr=cfg.eta_Q, weight_decay=0.0,  # L2 applied manually only to Qh/Qv per Eq. 41
        )
        self._epoch = 0

    def _build_qnode(self):
        n_q = self.n_q

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(l_padded_normalized, theta):
            qml.AmplitudeEmbedding(
                l_padded_normalized, wires=range(n_q), normalize=True, pad_with=0.0
            )
            qml.StronglyEntanglingLayers(theta, wires=range(n_q))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_q)]

        self.circuit = circuit

    def _pad_lag(self, lag: torch.Tensor) -> torch.Tensor:
        pad_len = self.dim - lag.shape[1]
        if pad_len > 0:
            return torch.nn.functional.pad(lag, (0, pad_len))
        return lag[:, : self.dim]

    def quantum_features(self, lag: torch.Tensor) -> torch.Tensor:
        x = self._pad_lag(lag)
        out = self.circuit(x, self.theta)
        return torch.stack(out, dim=1)  # (batch, n_q)

    def gated_features(self, lag: torch.Tensor):
        z = self.quantum_features(lag)          # (batch, n_q)
        q_h = z @ self.Qh                        # (batch, H)
        q_v = z @ self.Qv                        # (batch, V)
        gate_h = torch.sigmoid(self.alpha_h)     # (H,)
        gate_v = torch.sigmoid(self.alpha_v)     # (V,)
        return gate_h * q_h, gate_v * q_v

    # -- conditionals (Eqs. 37-38, fixed-sigma simplification) -----------
    def p_h_given_v(self, v: torch.Tensor, lag: torch.Tensor):
        q_h, _ = self.gated_features(lag)
        logits = (v / self.cfg.sigma) @ self.Wvh + self.c + q_h
        return torch.sigmoid(logits)

    def mean_v_given_h(self, h: torch.Tensor, lag: torch.Tensor):
        _, q_v = self.gated_features(lag)
        return self.b + self.cfg.sigma * (h @ self.Wvh.T) + q_v

    # -- classical CD path (closed-form, quantum features held fixed) ----
    def cd_step(self, v0: torch.Tensor, lag0: torch.Tensor):
        cfg = self.cfg
        batch = v0.shape[0]

        with torch.no_grad():
            p_h0 = self.p_h_given_v(v0, lag0)
            h0 = torch.bernoulli(p_h0)

            vk, hk = v0, h0
            for _ in range(cfg.k):
                mean_vk = self.mean_v_given_h(hk, lag0)
                vk = mean_vk + cfg.sigma * torch.randn_like(mean_vk)
                p_hk = self.p_h_given_v(vk, lag0)
                hk = torch.bernoulli(p_hk)

            sigma = cfg.sigma
            dWvh = ((v0 / sigma).T @ p_h0 - (vk / sigma).T @ p_hk) / batch
            db = (v0 - vk).mean(dim=0) / (sigma ** 2)
            dc = (p_h0 - p_hk).mean(dim=0)

        recon_mse = ((v0 - vk) ** 2).mean().item()
        return {"Wvh": dWvh, "b": db, "c": dc}, recon_mse

    def apply_grads(self, grads: dict):
        for name, grad in grads.items():
            param = getattr(self, name)
            setattr(self, name, param + self.cfg.lr * (grad - self.cfg.weight_decay * param))

    # -- quantum loss (Eqs. 39-41) ----------------------------------------
    def quantum_loss(self, v0: torch.Tensor, lag0: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        p_h0 = self.p_h_given_v(v0, lag0)               # quantum-augmented
        v_recon_q = self.mean_v_given_h(p_h0, lag0)
        recon_loss = ((v0 - v_recon_q) ** 2).mean()

        loss = recon_loss
        if self._epoch >= cfg.n_warmup:
            with torch.no_grad():
                logits_nq = (v0 / cfg.sigma) @ self.Wvh + self.c
                p_h0_nq = torch.sigmoid(logits_nq)
                v_recon_nq = self.b + cfg.sigma * (p_h0_nq @ self.Wvh.T)
            align_loss = ((v_recon_q - v_recon_nq.detach()) ** 2).mean()
            loss = loss + cfg.lambda_align * align_loss

        l2_penalty = 1e-4 * ((self.Qh ** 2).mean() + (self.Qv ** 2).mean())
        return loss + l2_penalty

    def train_step(self, v0: torch.Tensor, lag0: torch.Tensor):
        grads, recon_mse = self.cd_step(v0, lag0)
        self.apply_grads(grads)

        self.q_optimizer.zero_grad()
        loss_q = self.quantum_loss(v0, lag0)
        loss_q.backward()
        self.q_optimizer.step()

        return recon_mse, loss_q.item()

    def end_epoch(self):
        self._epoch += 1

    @torch.no_grad()
    def mean_field_predict(self, lag: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
        _, q_v = self.gated_features(lag)
        v = self.b + q_v
        for _ in range(n_iter):
            q_h, q_v = self.gated_features(lag)
            h = torch.sigmoid((v / self.cfg.sigma) @ self.Wvh + self.c + q_h)
            v = self.b + self.cfg.sigma * (h @ self.Wvh.T) + q_v
        return v

    def num_params(self) -> int:
        classical = self.Wvh.numel() + self.b.numel() + self.c.numel()
        quantum = (
            self.theta.numel() + self.Qh.numel() + self.Qv.numel()
            + self.alpha_h.numel() + self.alpha_v.numel()
        )
        return classical + quantum


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = QFeatureQRBMConfig(V=1, H=2, L=10, n_layers=1)
    model = QFeatureQRBM(cfg)
    print("n_q qubits:", model.n_q, "num params:", model.num_params())

    n = 64
    lag = torch.randn(n, cfg.L) * 0.1
    v = torch.randn(n, cfg.V) * 0.1

    for epoch in range(5):
        recon_mse, q_loss = model.train_step(v, lag)
        model.end_epoch()
    print("final recon MSE:", recon_mse, "quantum loss:", q_loss)
