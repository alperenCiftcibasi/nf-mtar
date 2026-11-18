"""ANFIS tarzı eğitim (CLAUDE.md §5.16; E9-lite ölçeği): θ = {c, σ, y_k}, kayıp ve düzenlileştirme modları, erken durdurma.

- Model: engine/fuzzy_torch.TorchFuzzyEngine (σ = softplus(s) + 1e-3, y_k = 0,1 + 0,9·sigmoid(u_k)); uzman başlangıcı θ0.
- Kayıp: E = (1/2N) Σ (y_oracle − f)² + (λ/2) R(θ);
    anchored: R = Σ (θ − θ0)²  (θ0 = §5.14 uzman parametreleri; c, σ, y üzerinde)
    zero_l2:  R = Σ θ²
    none:     R = 0
  (θ doğal parametreler c, σ, y üzerinden; ham s, u üzerinden değil.)
- Optimizasyon: tam yığın; Adam (lr 0,01; E9-lite süre kısıtı nedeniyle, §5.16'daki SGD-momentum yerine — config'e bağlı
  değil, fonksiyon argümanı) veya SGD momentum 0,9 (η = 0,01·exp(−epoch/500)); en çok max_epochs; doğrulama MSE'si
  `patience` epoch iyileşmezse durur; en iyi doğrulama durumu geri yüklenir.
- Değerlendirme (`evaluate`): MSE; grup (aynı durum × topoloji) içinde Spearman (ortalama), top-1 uyumu
  (argmax f = argmax kâhin), pişmanlık (kâhin_en_iyi − kâhin_seçilen).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr

from nfmtar.engine.fuzzy import FuzzyParams, FuzzySpec
from nfmtar.engine.fuzzy_torch import TorchFuzzyEngine

REG_MODES = ("anchored", "zero_l2", "none")


@dataclass
class TrainResult:
    params: FuzzyParams
    epochs: int
    best_val_mse: float
    history: list[tuple[int, float, float]]   # (epoch, train loss, val mse)


def _theta(model: TorchFuzzyEngine) -> list[torch.Tensor]:
    return [*model.centers, *model.sigmas(), model.consequents()]


def train(spec: FuzzySpec, X_tr, y_tr, X_val, y_val, reg_mode: str = "anchored", lam: float = 1e-2,
          max_epochs: int = 200, patience: int = 20, optimizer: str = "adam", lr: float = 0.01,
          sigma_min: float = 1e-3, seed: int = 0, batch_size: int | None = None,
          eval_chunk: int = 20_000) -> TrainResult:
    """§5.16 kaybıyla eğitim; uzman parametrelerinden başlar.

    batch_size None: tam yığın (belirlenimci). Verilirse her epoch'ta `seed` ile karıştırılan mini yığınlar (E9 tekrarları
    farklı karıştırma seed'iyle koşulur); düzenlileştirme terimi her mini yığına aynen eklenir.
    Doğrulama MSE'si eval_chunk satırlık parçalarla hesaplanır (aynı değer; büyük doğrulama kümesinde bellek sınırı).
    """
    if reg_mode not in REG_MODES:
        raise ValueError(f"reg_mode {reg_mode!r}: {REG_MODES}")
    torch.manual_seed(seed)
    model = TorchFuzzyEngine(spec, sigma_min=sigma_min)
    theta0 = [t.detach().clone() for t in _theta(model)]
    Xt, yt = torch.as_tensor(X_tr, dtype=torch.float64), torch.as_tensor(y_tr, dtype=torch.float64)
    Xv, yv = torch.as_tensor(X_val, dtype=torch.float64), torch.as_tensor(y_val, dtype=torch.float64)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = None
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: float(np.exp(-e / 500)))
    best = (float("inf"), copy.deepcopy(model.state_dict()), 0)
    history: list[tuple[int, float, float]] = []
    since_best = 0
    gen = torch.Generator().manual_seed(seed)
    n = len(yt)
    for epoch in range(1, max_epochs + 1):
        order = torch.randperm(n, generator=gen) if batch_size else torch.arange(n)
        step = batch_size or n
        for start in range(0, n, step):
            idx = order[start:start + step]
            opt.zero_grad()
            f = model(Xt[idx])["f"]
            loss = 0.5 * torch.mean((yt[idx] - f) ** 2)
            if reg_mode == "anchored":
                loss = loss + 0.5 * lam * sum(((a - b) ** 2).sum() for a, b in zip(_theta(model), theta0))
            elif reg_mode == "zero_l2":
                loss = loss + 0.5 * lam * sum((a ** 2).sum() for a in _theta(model))
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        with torch.no_grad():
            sse = sum(float(((yv[i:i + eval_chunk] - model(Xv[i:i + eval_chunk])["f"]) ** 2).sum())
                      for i in range(0, len(yv), eval_chunk))
            val = sse / max(len(yv), 1)
        history.append((epoch, float(loss.detach()), val))
        if val < best[0] - 1e-9:
            best = (val, copy.deepcopy(model.state_dict()), epoch)
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best[1])
    return TrainResult(model.to_params(), best[2], best[0], history)


def evaluate(f: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """MSE, grup içi Spearman (ortalama), top-1 uyumu, pişmanlık (§5.16).

    Gruplar bir kez sıralanıp dilimlenir (büyük veride grup başına tam maske O(G·n) yerine O(n log n)); grup içinde
    satır sırası korunur (kararlı sıralama), bu yüzden argmax eşitlikleri eski uygulamayla aynı kırılır."""
    f, y, groups = np.asarray(f), np.asarray(y), np.asarray(groups)
    out = {"mse": float(np.mean((y - f) ** 2))}
    order = np.argsort(groups, kind="stable")
    fs, ys = f[order], y[order]
    _, starts = np.unique(groups[order], return_index=True)
    bounds = [*starts.tolist(), len(order)]
    rhos, top1, regret = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        fg, yg = fs[a:b], ys[a:b]
        if len(fg) < 2:
            continue
        if np.ptp(fg) > 0 and np.ptp(yg) > 0:
            rhos.append(spearmanr(fg, yg).statistic)
        chosen = int(np.argmax(fg))
        top1.append(float(chosen == int(np.argmax(yg))))
        regret.append(float(yg.max() - yg[chosen]))
    out["spearman"] = float(np.mean(rhos)) if rhos else float("nan")
    out["top1"] = float(np.mean(top1))
    out["regret"] = float(np.mean(regret))
    return out
