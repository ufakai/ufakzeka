"""Muon optimizer (Jordan 2024) with the additions used in 2025-2026 practice:
Nesterov momentum, Newton-Schulz orthogonalisation in bfloat16, per-parameter update
scaling by sqrt(max(1, rows/cols)) (Moonshot "Muon is scalable"), decoupled weight decay,
and cautious weight decay (only decay where update and weight agree in sign, arXiv 2510.12402).

Used for every 2D weight inside the transformer blocks. Embeddings, norms and the head
use AdamW. Single GPU only."""

from __future__ import annotations

import torch


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.mT
    return X


if torch.cuda.is_available():
    newton_schulz = torch.compile(newton_schulz)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 weight_decay: float = 0.0, cautious_wd: bool = True, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, cautious_wd=cautious_wd, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=mom)
                else:
                    g = buf
                u = newton_schulz(g, group["ns_steps"])
                u = u * max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd > 0:
                    if group["cautious_wd"]:
                        mask = (u * p).sign() > 0  # decay only where the update pushes toward zero anyway
                        p.mul_(torch.where(mask, 1 - lr * wd, 1.0))
                    else:
                        p.mul_(1 - lr * wd)
                p.add_(u.type_as(p), alpha=-lr)
