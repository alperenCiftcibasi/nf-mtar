"""Bulanık motorun PyTorch eşdeğeri: aynı matematik, öğrenilebilir θ = {c, σ, y_k} (CLAUDE.md §5.14, §5.16).

Uygulanan kararlar:
- §5.14 "Eğitim için PyTorch eşdeğeri (aynı matematik)": Denk. 9 üyelik, Denk. 11 çarpım t-normu, Denk. 12 merkez-ortalama,
  Denk. 13a f_norm, Denk. 13b güven — fuzzy.py'deki NumPy motoruyla birebir (testte 1e-6 içinde, float64).
- §5.16 parametreleştirme: σ = softplus(s) + σ_min (σ_min = config training.sigma_min, 1e-3) -> σ > 0 her zaman;
  y_k = y_min + y_span · sigmoid(u_k) -> y_k ∈ (0,1; 1). Uzman başlangıcından ters dönüşümle s ve u kurulur
  (s = softplus⁻¹(σ − σ_min), u = logit((y − y_min)/y_span)); y = 1,0 olan kural (bütün girdiler en iyi terimde)
  logit'in sonsuz olmaması için (0,1) içine 1e-12 ile kırpılır (y hatası ≤ 1e-12).
- Kural tablosu (don't care dahil) sabittir (buffer); öğrenilmez.
- Eğitim döngüsü, kayıp (çapalı L2) ve optimizasyon BURADA YOK: engine/trainer.py (M8).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nfmtar.engine.fuzzy import DONT_CARE, FuzzyParams, FuzzySpec

_Y_EPS = 1e-12


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """softplus⁻¹(x) = x + log(−expm1(−x)), x > 0 (sayısal kararlı)."""
    return x + torch.log(-torch.expm1(-x))


def torch_infer(
    X: torch.Tensor,
    centers: list[torch.Tensor],
    sigmas: list[torch.Tensor],
    y: torch.Tensor,
    cols: list[torch.Tensor],
    y_min: float,
    y_span: float,
) -> dict[str, torch.Tensor]:
    """Saf fonksiyonel çıkarım (autograd testleri için): X (B, 5) -> f, f_norm, confidence, w_norm, top_rule.

    cols[i]: kural başına girdi i'nin sütunu (don't care -> n_i, log üyeliği 0 olan ek sütun).
    NumPy motoruyla aynı biçimde log uzayında normalize edilir: w_k/Σw = softmax(log w) (küçük σ'da 0/0 olmaz).
    """
    logw = torch.zeros(X.shape[0], y.shape[0], dtype=X.dtype)
    for i, (c, s) in enumerate(zip(centers, sigmas)):
        z = -(((X[:, i:i + 1] - c[None, :]) / s[None, :]) ** 2)                       # log μ, Denk. 9
        z1 = torch.cat([z, torch.zeros(X.shape[0], 1, dtype=X.dtype)], dim=1)
        logw = logw + z1[:, cols[i]]                                                   # log Π μ, Denk. 11
    w_norm = torch.softmax(logw, dim=1)                                                # w_k / Σ w
    f = w_norm @ y                                                                     # Denk. 12
    f_norm = torch.clamp((f - y_min) / y_span, 0.0, 1.0)                               # Denk. 13a
    conf, top = w_norm.max(dim=1)                                                      # Denk. 13b
    return {"f": f, "f_norm": f_norm, "confidence": conf, "w_norm": w_norm, "top_rule": top}


class TorchFuzzyEngine(nn.Module):
    """Öğrenilebilir bulanık motor (§5.16): parametreler girdi başına c_i, s_i ve kural başına u_k."""

    def __init__(self, spec: FuzzySpec, params: FuzzyParams | None = None, sigma_min: float = 1e-3,
                 dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        params = params or FuzzyParams.expert(spec)
        self.spec = spec
        self.sigma_min = float(sigma_min)
        self.y_min, self.y_span = spec.y_min, spec.y_span
        self.centers = nn.ParameterList([nn.Parameter(torch.as_tensor(c, dtype=dtype).clone()) for c in params.centers])
        if any((s <= sigma_min).any() for s in params.sigmas):
            raise ValueError(f"σ > σ_min = {sigma_min} olmalı (σ = softplus(s) + σ_min)")
        self.s = nn.ParameterList([nn.Parameter(inverse_softplus(torch.as_tensor(sg, dtype=dtype) - sigma_min))
                                   for sg in params.sigmas])
        p = ((torch.as_tensor(params.y, dtype=dtype) - self.y_min) / self.y_span).clamp(_Y_EPS, 1.0 - _Y_EPS)
        self.u = nn.Parameter(torch.logit(p))
        table = torch.as_tensor(params.table, dtype=torch.long)
        self.register_buffer("table", table)
        for i, n in enumerate(spec.n_terms):
            self.register_buffer(f"cols_{i}", torch.where(table[:, i] == DONT_CARE, torch.tensor(n), table[:, i]))

    @classmethod
    def from_config(cls, cfg, params: FuzzyParams | None = None) -> "TorchFuzzyEngine":
        return cls(FuzzySpec.from_config(cfg), params, sigma_min=float(cfg.training["sigma_min"]))

    def sigmas(self) -> list[torch.Tensor]:
        """σ = softplus(s) + σ_min (§5.16)."""
        return [F.softplus(s) + self.sigma_min for s in self.s]

    def consequents(self) -> torch.Tensor:
        """y_k = y_min + y_span · sigmoid(u_k) (§5.16)."""
        return self.y_min + self.y_span * torch.sigmoid(self.u)

    def cols(self) -> list[torch.Tensor]:
        return [getattr(self, f"cols_{i}") for i in range(len(self.spec.n_terms))]

    def forward(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        X = torch.as_tensor(X, dtype=self.u.dtype)
        return torch_infer(X, list(self.centers), self.sigmas(), self.consequents(), self.cols(), self.y_min, self.y_span)

    def to_params(self) -> FuzzyParams:
        """Eğitilmiş θ -> NumPy motorunun FuzzyParams'ı (results/models/<run_id>.json, §5.16)."""
        with torch.no_grad():
            return FuzzyParams(
                centers=[c.detach().cpu().numpy().copy() for c in self.centers],
                sigmas=[s.detach().cpu().numpy().copy() for s in self.sigmas()],
                y=self.consequents().detach().cpu().numpy().copy(),
                table=self.table.cpu().numpy().copy(),
            )
