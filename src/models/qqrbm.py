"""
Full-register Quantum-Quantum QQRBM.

Reimplements Section 4.4 of Hellstern et al. (2026), arXiv:2607.24065, as a
STRUCTURAL REFERENCE model -- exactly the role it plays in the base paper
itself (Sec 5.5: "QQRBM is retained as a structural reference at its natural
quantum-register size, not as a memory-matched competitor").

Design:
  - Three qubit registers: visible (V qubits), hidden (H qubits), context
    (U_qq qubits), entangled by a single shared variational circuit
    (Sec 4.4.1). The base paper deliberately restricts the context register
    to the last few lag values (U_qq=3) rather than the full context window,
    to keep the qubit count small (Sec 5.5) -- we follow the same design
    choice here and report it explicitly as a known information restriction,
    not a memory-matched competitor to CRBM/QCRBM.
  - Raw Pauli-Z expectations on the visible and hidden registers are
    projected through learnable linear heads Q_v, Q_h (Eq. 29).
  - Context enters ONLY through the circuit (classical context weights are
    disabled in quantum mode, Sec 4.4.2).
  - Training combines classical CD-k updates for {Wvh, b, c} with a quantum
    Adam update on {theta, Qh, Qv, alpha_h, alpha_v} via the positive-phase
    soft reconstruction loss (Eq. 33).

Because this model requires V >= 2 (Sec 4.4.1) and is fixed at its natural
architecture size rather than hyperparameter-matched, it is NOT part of the
iso-parameter (matched-budget) comparison -- only of the main scaling-curve
comparison, exactly mirroring the base paper's Experiment 2 / Experiment 10
treatment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    import pennylane as qml
except ImportError:  # pragma: no cover
    qml = None

from .crbm import xavier_uniform


@dataclass
class QQRBMConfig:
    V: int = 2            # visible register size (must be >= 2, Sec 4.4.1)
    H: int = 2             # hidden register size
    U_qq: int = 3           # restricted context register size (natural size)
    L_Q: int = 1            # PQC depth
    k: int = 3
    sigma: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    eta_Q: float = 1e-3
    lambda_Q: float = 1e-3
    alpha_init: float = 1.0
    seed: int = 0


class QQRBM:
    """Simplified but functional reimplementation of the base paper's
    three-register QQRBM. V here is the padded visible dimension used only
    for the amplitude register; the model still forecasts a scalar return by
    reading out the first visible unit."""

    def __init__(self, cfg: QQRBMConfig):
        if qml is None:
            raise ImportError("pennylane is required for QQRBM")
        if cfg.V < 2:
            raise ValueError("QQRBM requires V >= 2 (Sec 4.4.1)")

        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)

        self.n_wires = cfg.V + cfg.H + cfg.U_qq
        self.v_wires = list(range(cfg.V))
        self.h_wires = list(range(cfg.V, cfg.V + cfg.H))
        self.u_wires = list(range(cfg.V + cfg.H, self.n_wires))

        self.dev = qml.device("default.qubit", wires=self.n_wires)

        # classical CD parameters (single interaction matrix, Remark 4.2)
        self.Wvh = xavier_uniform(cfg.V, cfg.H, g)
        self.b = torch.zeros(cfg.V)
        self.c = torch.zeros(cfg.H)

        # quantum parameters
        self.theta = torch.nn.Parameter(
            (torch.rand(cfg.L_Q, self.n_wires, 3, generator=g) * 2 - 1) * np.pi
        )
        self.Qh = torch.nn.Parameter(torch.randn(cfg.H, cfg.H, generator=g) * 0.1)
        self.Qv = torch.nn.Parameter(torch.randn(cfg.V, cfg.V, generator=g) * 0.1)
        self.alpha_h = torch.nn.Parameter(torch.tensor(cfg.alpha_init, dtype=torch.float32))
        self.alpha_v = torch.nn.Parameter(torch.tensor(cfg.alpha_init, dtype=torch.float32))

        self._build_qnode()
        self.q_optimizer = torch.optim.Adam(
            [self.theta, self.Qh, self.Qv, self.alpha_h, self.alpha_v],
            lr=cfg.eta_Q, weight_decay=cfg.lambda_Q,
        )

    def _build_qnode(self):
        v_wires, h_wires, u_wires = self.v_wires, self.h_wires, self.u_wires
        dim_v = 2 ** len(v_wires)
        dim_u = 2 ** len(u_wires)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(v_amp, u_amp, theta):
            qml.AmplitudeEmbedding(v_amp, wires=v_wires, normalize=True, pad_with=0.0)
            qml.AmplitudeEmbedding(u_amp, wires=u_wires, normalize=True, pad_with=0.0)
            # hidden register starts at |0>^H implicitly
            qml.StronglyEntanglingLayers(theta, wires=range(len(v_wires) + len(h_wires) + len(u_wires)))
            outs_v = [qml.expval(qml.PauliZ(w)) for w in v_wires]
            outs_h = [qml.expval(qml.PauliZ(w)) for w in h_wires]
            return outs_v + outs_h

        self.circuit = circuit
        self.dim_v = dim_v
        self.dim_u = dim_u

    def _pad(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        pad_len = target_dim - x.shape[1]
        if pad_len > 0:
            return torch.nn.functional.pad(x, (0, pad_len))
        return x[:, :target_dim]

    def raw_quantum_outputs(self, v: torch.Tensor, u_context: torch.Tensor):
        v_amp = self._pad(v, self.dim_v)
        u_amp = self._pad(u_context, self.dim_u)
        out = self.circuit(v_amp, u_amp, self.theta)
        n_v = len(self.v_wires)
        q_v_raw = torch.stack(out[:n_v], dim=1)
        q_h_raw = torch.stack(out[n_v:], dim=1)
        return q_v_raw, q_h_raw

    def projected_features(self, v: torch.Tensor, u_context: torch.Tensor):
        q_v_raw, q_h_raw = self.raw_quantum_outputs(v, u_context)
        q_v = q_v_raw @ self.Qv
        q_h = q_h_raw @ self.Qh
        return q_v, q_h

    # -- conditionals (Eqs. 30-31) ---------------------------------------
    def p_h_given_v(self, v: torch.Tensor, u_context: torch.Tensor):
        _, q_h = self.projected_features(v, u_context)
        logits = (v / self.cfg.sigma) @ self.Wvh + self.c + self.alpha_h * q_h
        return torch.sigmoid(logits)

    def mean_v_given_h(self, h: torch.Tensor, u_context: torch.Tensor, v_for_features: torch.Tensor):
        q_v, _ = self.projected_features(v_for_features, u_context)
        return self.b + self.cfg.sigma * (h @ self.Wvh.T) + self.alpha_v * q_v

    # -- heuristic quantum-feedback Gibbs chain (Algorithm 3) -------------
    def cd_step(self, v0: torch.Tensor, u_context: torch.Tensor):
        cfg = self.cfg
        batch = v0.shape[0]

        with torch.no_grad():
            p_h0 = self.p_h_given_v(v0, u_context)
            h0 = torch.bernoulli(p_h0)

            vk, hk = v0, h0
            for _ in range(cfg.k):
                mean_vk = self.mean_v_given_h(hk, u_context, v_for_features=vk)
                vk = mean_vk + cfg.sigma * torch.randn_like(mean_vk)
                p_hk = self.p_h_given_v(vk, u_context)
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

    def quantum_loss(self, v0: torch.Tensor, u_context: torch.Tensor) -> torch.Tensor:
        p_h0 = self.p_h_given_v(v0, u_context)
        v_recon = self.mean_v_given_h(p_h0, u_context, v_for_features=v0)
        return ((v0 - v_recon) ** 2).mean()

    def train_step(self, v0: torch.Tensor, u_context: torch.Tensor):
        grads, recon_mse = self.cd_step(v0, u_context)
        self.apply_grads(grads)

        self.q_optimizer.zero_grad()
        loss_q = self.quantum_loss(v0, u_context)
        loss_q.backward()
        self.q_optimizer.step()

        return recon_mse, loss_q.item()

    @torch.no_grad()
    def mean_field_predict(self, u_context: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
        v = self.b.unsqueeze(0).repeat(u_context.shape[0], 1)
        for _ in range(n_iter):
            h = self.p_h_given_v(v, u_context)
            v = self.mean_v_given_h(h, u_context, v_for_features=v)
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
    cfg = QQRBMConfig(V=2, H=2, U_qq=3, L_Q=1)
    model = QQRBM(cfg)
    print("n_wires:", model.n_wires, "num params:", model.num_params())

    n = 32
    # V=2 visible "padding": real target is the first column; second is a
    # zero-padded dummy dimension required by the V>=2 constraint.
    v = torch.zeros(n, cfg.V)
    v[:, 0] = torch.randn(n) * 0.1
    u_context = torch.randn(n, cfg.U_qq) * 0.1

    for epoch in range(3):
        recon_mse, q_loss = model.train_step(v, u_context)
    print("final recon MSE:", recon_mse, "quantum loss:", q_loss)
