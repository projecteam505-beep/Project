"""
Hybrid Quantum-Classical QCRBM.

Reimplements Section 4.3 of Hellstern et al. (2026), arXiv:2607.24065:

  - The current (v, u) pair is amplitude-encoded into a PQC with
    H_Q = ceil(log2(V+U)) qubits (Eq. 20).
  - A StronglyEntanglingLayers-style variational circuit of depth L_Q
    produces Pauli-Z expectation features q(x; theta) in [-1, 1]^H (Eq. 21).
  - These features additively correct the hidden-unit logits, scaled by a
    learnable scalar alpha (Eq. 22):
        p(h_j=1 | v,u) = sigmoid( (1/sigma) v W + c_eff,j + alpha * q_j )
  - At alpha=0 this exactly recovers the classical CRBM (Remark 4.1) --
    the "conservative extension" property.
  - Classical parameters {W, b, c, Wcv, Wch} are still updated via CD-k,
    identical to the pure CRBM (Eq. 11-16).
  - Quantum parameters {theta, alpha} are updated via a *positive-phase soft
    surrogate reconstruction loss* (Eqs. 23-24), since Bernoulli sampling is
    non-differentiable. This uses standard autograd (PennyLane's torch
    interface), not the parameter-shift rule directly -- backprop through a
    statevector simulator is mathematically equivalent and much faster, as
    the base paper notes in Section 3.4 for n_wires <= 12.

QCRBM always requires H >= H_Q = ceil(log2(V+U)) hidden qubits (Sec 4.3.2),
so at small classical H the model is "floored" up to this size -- this is
intentional and matches the base paper's design.
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

from .crbm import CRBM, CRBMConfig, xavier_uniform


@dataclass
class QCRBMConfig(CRBMConfig):
    L_Q: int = 2          # PQC depth (variational layers)
    alpha_init: float = 1.0
    eta_Q: float = 1e-3   # quantum Adam learning rate
    lambda_Q: float = 1e-3  # quantum weight decay
    k_Q: int = 1           # CD steps used while computing the quantum loss's
                            # positive phase (matches base paper's kQ)


def required_qubits(V: int, U: int) -> int:
    """H_Q = ceil(log2(V+U)) -- amplitude encoding requires 2^H >= V+U (Sec 4.3.2)."""
    return max(1, math.ceil(math.log2(max(V + U, 2))))


class QCRBM(CRBM):
    """Hybrid QCRBM: classical CRBM backbone + additive quantum correction to
    the hidden logits, per Section 4.3 of the base paper."""

    def __init__(self, cfg: QCRBMConfig):
        # Enforce the qubit floor H >= H_Q (Sec 4.3.2)
        H_Q = required_qubits(cfg.V, cfg.U)
        if cfg.H < H_Q:
            cfg.H = H_Q
        super().__init__(cfg)
        self.cfg: QCRBMConfig = cfg
        self.n_qubits = cfg.H
        self.dim = 2 ** self.n_qubits

        if qml is None:
            raise ImportError("pennylane is required for QCRBM")

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Quantum parameters: StronglyEntanglingLayers shape (L_Q, n_qubits, 3)
        g = torch.Generator().manual_seed(cfg.seed + 1)
        self.theta = torch.nn.Parameter(
            (torch.rand(cfg.L_Q, self.n_qubits, 3, generator=g) * 2 - 1) * np.pi
        )
        self.alpha = torch.nn.Parameter(torch.tensor(cfg.alpha_init, dtype=torch.float32))

        self._build_qnode()

        self.q_optimizer = torch.optim.Adam(
            [self.theta, self.alpha], lr=cfg.eta_Q, weight_decay=cfg.lambda_Q
        )

    # ------------------------------------------------------------------
    def _build_qnode(self):
        n_qubits = self.n_qubits

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x_padded_normalized, theta):
            qml.AmplitudeEmbedding(
                x_padded_normalized, wires=range(n_qubits), normalize=True, pad_with=0.0
            )
            qml.StronglyEntanglingLayers(theta, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def _encode_input(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Concatenate + pad [v; u] to length 2^n_qubits (Eq. 20)."""
        x = torch.cat([v, u], dim=1)  # (batch, V+U)
        pad_len = self.dim - x.shape[1]
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, pad_len))
        elif pad_len < 0:
            x = x[:, : self.dim]
        return x

    def quantum_features(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """q(x; theta) in [-1,1]^H, batched (Eq. 21). PennyLane's default.qubit
        broadcasts over a batch dimension when the input tensor has one."""
        x = self._encode_input(v, u)
        out = self.circuit(x, self.theta)  # list of H tensors, each (batch,)
        return torch.stack(out, dim=1)  # (batch, H)

    # -- augmented hidden conditional (Eq. 22) --------------------------
    def p_h_given_v_quantum(self, v: torch.Tensor, u: torch.Tensor, q: torch.Tensor = None):
        _, c_eff = self.effective_biases(u)
        if q is None:
            q = self.quantum_features(v, u)
        logits = (v / self.cfg.sigma) @ self.W + c_eff + self.alpha * q
        return torch.sigmoid(logits)

    # -- classical CD path (still uses the QUANTUM-augmented hidden conditional,
    #    but W/b/c/Wcv/Wch updates remain the closed-form CD rule; Remark 4.1) --
    def cd_step_quantum(self, v0: torch.Tensor, u0: torch.Tensor):
        cfg = self.cfg
        batch = v0.shape[0]

        with torch.no_grad():
            q0 = self.quantum_features(v0, u0)
            p_h0 = self.p_h_given_v_quantum(v0, u0, q=q0)
            h0 = torch.bernoulli(p_h0)

            vk, hk = v0, h0
            for _ in range(cfg.k):
                vk_mean = self.mean_v_given_h(hk, u0)
                vk = vk_mean + cfg.sigma * torch.randn_like(vk_mean)
                qk = self.quantum_features(vk, u0)
                p_hk = self.p_h_given_v_quantum(vk, u0, q=qk)
                hk = torch.bernoulli(p_hk)

            sigma = cfg.sigma
            dW = ((v0 / sigma).T @ p_h0 - (vk / sigma).T @ p_hk) / batch
            db = (v0 - vk).mean(dim=0) / (sigma ** 2)
            dc = (p_h0 - p_hk).mean(dim=0)
            dWcv = (u0.T @ (v0 - vk)) / (batch * sigma ** 2)
            dWch = (u0.T @ (p_h0 - p_hk)) / batch

        grads = {"W": dW, "b": db, "c": dc, "Wcv": dWcv, "Wch": dWch}
        recon_mse = ((v0 - vk) ** 2).mean().item()
        return grads, recon_mse, p_h0

    # -- quantum path: positive-phase soft surrogate loss (Eqs. 23-24) ---
    def quantum_loss(self, v0: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        q0 = self.quantum_features(v0, u0)
        p_h0 = self.p_h_given_v_quantum(v0, u0, q=q0)  # differentiable in theta, alpha
        b_eff, _ = self.effective_biases(u0)
        v_recon = b_eff + self.cfg.sigma * (p_h0 @ self.W.T)
        return ((v0 - v_recon) ** 2).mean()

    def train_step(self, v0: torch.Tensor, u0: torch.Tensor):
        """One QCRBM mini-batch update: classical CD path + quantum Adam path,
        exactly mirroring Algorithm 2 of the base paper."""
        # 1. Classical path (CD-k, quantum-augmented conditional as input only)
        grads, recon_mse, _ = self.cd_step_quantum(v0, u0)
        self.apply_grads(grads)

        # 2. Quantum path (soft surrogate loss, Adam)
        self.q_optimizer.zero_grad()
        loss_q = self.quantum_loss(v0, u0)
        loss_q.backward()
        self.q_optimizer.step()

        return recon_mse, loss_q.item()

    # -- forecasting: same mean-field iteration, but hidden step uses the
    #    quantum-augmented conditional (alpha, theta held fixed at inference) --
    @torch.no_grad()
    def mean_field_predict(self, u: torch.Tensor, n_iter: int = 5) -> torch.Tensor:
        b_eff, c_eff = self.effective_biases(u)
        v = b_eff.clone()
        for _ in range(n_iter):
            q = self.quantum_features(v, u)
            h = torch.sigmoid((v / self.cfg.sigma) @ self.W + c_eff + self.alpha * q)
            v = b_eff + self.cfg.sigma * (h @ self.W.T)
        return v

    def num_params(self) -> int:
        classical = sum(p.numel() for p in [self.W, self.Wcv, self.Wch, self.b, self.c])
        quantum = self.theta.numel() + self.alpha.numel()
        return classical + quantum


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = QCRBMConfig(V=1, H=4, U=10, k=3, sigma=0.1, lr=1e-3, L_Q=2, eta_Q=1e-3)
    model = QCRBM(cfg)
    print("n_qubits (H, floored):", model.n_qubits)
    print("num params:", model.num_params())

    n = 64
    u = torch.randn(n, cfg.U) * 0.1
    v = torch.randn(n, cfg.V) * 0.1

    for epoch in range(5):
        recon_mse, q_loss = model.train_step(v, u)
    print("final recon MSE:", recon_mse, "quantum loss:", q_loss)

    pred = model.mean_field_predict(u[:5])
    print("prediction shape:", pred.shape)
