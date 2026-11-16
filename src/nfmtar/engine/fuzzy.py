"""Sıfırıncı dereceden Takagi-Sugeno bulanık çıkarım motoru ve girdi çıkarımı (CLAUDE.md §5.14; makale Denk. 9, 11, 12, 13a/b).

Uygulanan kararlar (§5.14):
- Sistem: sıfırıncı dereceden Takagi-Sugeno. Makalenin Denk. 12'sinde μ_Bk(y_k) = 1 olduğundan çıkış merkez-ortalama
  durulaştırıcıya indirgenir: f = Σ w_k y_k / Σ w_k.
- Üyelik (Denk. 9; MATLAB gaussmf DEĞİL): μ(x) = exp(−((x − c)/σ)²). Merkezler [0,1]'de eşit aralıklı, komşu terimler
  0,5'te kesişir: σ = (Δ/2)/sqrt(ln 2). 5 terim: c = 0, .25, .5, .75, 1, σ ≈ 0,150; 3 terim σ ≈ 0,300; 2 terim σ ≈ 0,601.
- Kural tabanı: 5×5×5×2×3 = 750 kuralın TAMAMI. Kural-terim indeks tablosu (750×5) itertools.product sırasıyla:
  x1 en yavaş, x5 en hızlı değişen sütun; kural indeksi 0 tabanlıdır.
- Kural sonucu (uzman fonksiyonu): terim indeksi j -> iyilik g = j/(n−1) ("yüksek = kötü" girdilerde 1 − j/(n−1));
  y_k = 0,1 + 0,9 × Σ_i a_i · g_i, a = {x1 .15, x2 .20, x3 .30, x4 .25, x5 .10} (config fuzzy.consequent).
- Çıkarım: w_k = Π_i μ_{i,term(k,i)}(x_i) (çarpım t-normu, Denk. 11); f = Σ w_k y_k / Σ w_k (Denk. 12);
  f_norm = clip((f − 0,1)/0,9, 0, 1) (Denk. 13a); confidence = max_k w_k / Σ w_k ∈ [1/750, 1] (Denk. 13b).
- Uygulama: NumPy vektörize; (B, 5) girdi matrisi tek seferde değerlendirilir (B aday × 750 kural).
- Opsiyonel pruned=True: bir girdinin TÜM terimleri için aynı sonucu veren (diğer girdileri aynı) kurallar tek "don't
  care" kuralında birleştirilir; don't-care girdinin üyeliği 1'dir. Not: Gauss üyelikler terimler üzerinde toplamı 1
  olmayan (bölümleme olmayan) fonksiyonlar olduğundan birleştirilmiş tabanın çıkışı tam taban ile birebir aynı DEĞİLDİR
  (birleşen grubun ağırlığı Σ_j μ_ij yerine 1 olur); yorumlanabilirlik/özet içindir. Uzman parametrelerinde bütün a_i > 0
  olduğundan hiçbir kural birleşmez; birleşme ancak eğitilmiş y_k'lerde bir girdi etkisizleşirse görülür.
- Normalizasyon (§5.14 "Normalizasyon fuzzy.py içinde"): `extract_features` ağ durumundan sabit normalizasyonlu
  x1–x5'i üretir (saf fonksiyon; ağ nesnesini değiştirmez). Kuyruk doluluğu ve trafik hızları dışarıdan verilir
  (simülasyon durumu). CLAUDE.md'de açık olmayan ve burada seçilen ayrıntılar:
    * "topoloji içi komşu": topolojinin CANLI sensörleri; edge router komşu sayılmaz (edge router'a yakınlık rank'taki
      H ve C ile taşınır, §5.17; iki kez sayılmasın).
    * x1'in 2-atlama komşuluğu düğümün kendisini de içerir (kapalı komşuluk); payda topolojinin canlı sensörlerinin
      toplam hızıdır. Toplam hız 0 ise x1 = 0.
    * x4: komşusu olmayan düğümde 0 (bağlantısı olmayan düğümün link kalitesi yok).
- PyTorch eşdeğeri (aynı matematik, öğrenilebilir c, σ, y_k) engine/fuzzy_torch.py'dedir; torch'u yalnızca eğitim
  yükler, simülasyon yüklemez.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from nfmtar.config import Config
from nfmtar.engine.base import FEATURE_LABELS, FEATURE_NAMES, NodeFeatures, ScoreOutput, features_matrix

TERM_LABELS: dict[int, tuple[str, ...]] = {
    5: ("VeryLow", "Low", "Medium", "High", "VeryHigh"),
    3: ("Low", "Medium", "High"),
    2: ("Low", "High"),
}
DONT_CARE = -1          # pruned kural tablosunda "herhangi bir terim" (üyelik 1)
HIGH_GOOD, HIGH_BAD = "high_good", "high_bad"


# --------------------------------------------------------------------------- #
# Yapı: girdi tanımları ve uzman parametreleri (§5.14)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FuzzySpec:
    """Motorun sabit yapısı (config fuzzy bölümünden): girdi başına terim sayısı ve yönü, sonuç ağırlıkları a_i, y aralığı."""

    n_terms: tuple[int, ...]
    directions: tuple[str, ...]
    weights: tuple[float, ...]
    y_min: float
    y_span: float
    neighborhood_hops: int        # x1 komşuluk yarıçapı (2)
    max_neighbors: int            # x5 normalizasyonu (15)

    @classmethod
    def from_config(cls, cfg: Config) -> "FuzzySpec":
        """configs/default.yaml fuzzy bölümünü okur; girdi anahtarları 'x1_demand' gibi, sıra x1..x5 (§5.14)."""
        fz = cfg.fuzzy
        by_name: dict[str, dict[str, Any]] = {}
        for key, spec in fz["inputs"].items():
            by_name[key.split("_", 1)[0]] = spec
        if tuple(sorted(by_name)) != FEATURE_NAMES:
            raise ValueError(f"fuzzy.inputs x1..x5 olmalı, bulunan: {sorted(by_name)}")
        cons = fz["consequent"]
        return cls(
            n_terms=tuple(int(by_name[n]["n_terms"]) for n in FEATURE_NAMES),
            directions=tuple(str(by_name[n]["direction"]) for n in FEATURE_NAMES),
            weights=tuple(float(cons["weights"][n]) for n in FEATURE_NAMES),
            y_min=float(cons["y_min"]),
            y_span=float(cons["y_span"]),
            neighborhood_hops=int(by_name["x1"].get("neighborhood_hops", 2)),
            max_neighbors=int(by_name["x5"].get("max_neighbors", 15)),
        )

    def __post_init__(self) -> None:
        if len(self.n_terms) != len(FEATURE_NAMES) or any(n not in TERM_LABELS for n in self.n_terms):
            raise ValueError(f"terim sayıları {self.n_terms}: girdi başına 2, 3 veya 5 bekleniyordu (§5.14)")
        if any(d not in (HIGH_GOOD, HIGH_BAD) for d in self.directions):
            raise ValueError(f"yönler {self.directions}: high_good / high_bad")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-9) or min(self.weights) < 0:
            raise ValueError(f"sonuç ağırlıkları a_i negatif olmamalı ve toplamı 1 olmalı (§5.14): {self.weights}")

    @property
    def n_rules(self) -> int:
        return int(np.prod(self.n_terms))


def equally_spaced_centers(n_terms: int) -> np.ndarray:
    """[0,1] aralığında eşit aralıklı merkezler (§5.14): 5 terim -> 0, .25, .5, .75, 1."""
    return np.linspace(0.0, 1.0, n_terms)


def half_crossing_sigma(n_terms: int) -> float:
    """Komşu terimler 0,5'te kesişsin: exp(−((Δ/2)/σ)²) = 0,5 -> σ = (Δ/2)/sqrt(ln 2) (§5.14)."""
    delta = 1.0 / (n_terms - 1)
    return (delta / 2.0) / math.sqrt(math.log(2.0))


def gaussian_mf(x: np.ndarray, c: np.ndarray | float, sigma: np.ndarray | float) -> np.ndarray:
    """Makale Denk. 9: μ(x) = exp(−((x − c)/σ)²) (MATLAB gaussmf'teki 2σ² paydası YOK)."""
    return np.exp(-(((x - c) / sigma) ** 2))


def full_rule_table(n_terms: Sequence[int]) -> np.ndarray:
    """(K, 5) kural-terim indeks tablosu, K = Π n_i (§5.14: 750); x1 en yavaş, x5 en hızlı değişen sütun."""
    return np.array(list(itertools.product(*(range(n) for n in n_terms))), dtype=np.int64)


def term_goodness(n_terms: int, direction: str) -> np.ndarray:
    """Terim indeksi -> iyilik g ∈ [0,1], eşit aralıklı; 'yüksek = kötü' girdilerde ters (§5.14)."""
    g = np.linspace(0.0, 1.0, n_terms)
    return g if direction == HIGH_GOOD else g[::-1].copy()


def expert_consequents(spec: FuzzySpec, table: np.ndarray) -> np.ndarray:
    """Uzman kural sonuçları: y_k = y_min + y_span × Σ_i a_i · g_i(term(k, i)) ∈ [0,1; 1] (§5.14)."""
    total = np.zeros(len(table))
    for i, (n, d, a) in enumerate(zip(spec.n_terms, spec.directions, spec.weights)):
        total += a * term_goodness(n, d)[table[:, i]]
    return spec.y_min + spec.y_span * total


@dataclass
class FuzzyParams:
    """Öğrenilebilir parametreler θ = {c, σ, y_k} (§5.16) ve kural tablosu; uzman başlangıcı θ0 = `expert`."""

    centers: list[np.ndarray]     # girdi başına (n_i,)
    sigmas: list[np.ndarray]      # girdi başına (n_i,), > 0
    y: np.ndarray                 # (K,)
    table: np.ndarray             # (K, 5); DONT_CARE yalnızca pruned tabloda

    @classmethod
    def expert(cls, spec: FuzzySpec) -> "FuzzyParams":
        """§5.14 uzman parametreleri: eşit aralıklı merkezler, yarı-kesişim σ, uzman y_k, 750 kuralın tamamı."""
        table = full_rule_table(spec.n_terms)
        return cls(
            centers=[equally_spaced_centers(n) for n in spec.n_terms],
            sigmas=[np.full(n, half_crossing_sigma(n)) for n in spec.n_terms],
            y=expert_consequents(spec, table),
            table=table,
        )

    def copy(self) -> "FuzzyParams":
        return FuzzyParams([c.copy() for c in self.centers], [s.copy() for s in self.sigmas], self.y.copy(), self.table.copy())

    def to_dict(self) -> dict[str, Any]:
        """JSON'a yazılabilir sözlük (results/models/<run_id>.json, §5.16)."""
        return {
            "centers": [c.tolist() for c in self.centers],
            "sigmas": [s.tolist() for s in self.sigmas],
            "y": self.y.tolist(),
            "table": self.table.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FuzzyParams":
        return cls(
            centers=[np.asarray(c, dtype=float) for c in d["centers"]],
            sigmas=[np.asarray(s, dtype=float) for s in d["sigmas"]],
            y=np.asarray(d["y"], dtype=float),
            table=np.asarray(d["table"], dtype=np.int64),
        )

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "FuzzyParams":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def prune_rules(table: np.ndarray, y: np.ndarray, n_terms: Sequence[int], tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """§5.14 "don't care" birleştirme: girdi i'nin bütün terimleri için diğer sütunları aynı ve sonuçları (tol içinde) eşit
    olan kural grubu, i sütunu DONT_CARE olan tek kurala (sonuç = grup ortalaması) indirgenir. Değişiklik kalmayana dek
    bütün girdiler üzerinde tekrarlanır. Dönüş belirlenimci sırada (tablo satırlarına göre sözlük sırası)."""
    rules: list[tuple[tuple[int, ...], float]] = [(tuple(int(v) for v in row), float(yk)) for row, yk in zip(table, y)]
    changed = True
    while changed:
        changed = False
        for i, n in enumerate(n_terms):
            groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
            for idx, (row, _) in enumerate(rules):
                if row[i] != DONT_CARE:
                    groups[row[:i] + row[i + 1:]].append(idx)
            merged: set[int] = set()
            new_rules: list[tuple[tuple[int, ...], float]] = []
            for key, idxs in groups.items():
                terms = {rules[j][0][i] for j in idxs}
                ys = [rules[j][1] for j in idxs]
                if len(idxs) == n and len(terms) == n and max(ys) - min(ys) <= tol:
                    new_rules.append((key[:i] + (DONT_CARE,) + key[i:], float(np.mean(ys))))
                    merged.update(idxs)
            if merged:
                rules = [r for j, r in enumerate(rules) if j not in merged] + new_rules
                changed = True
    rules.sort(key=lambda r: r[0])
    return np.array([r[0] for r in rules], dtype=np.int64), np.array([r[1] for r in rules], dtype=float)


# --------------------------------------------------------------------------- #
# Çıkarım (§5.14 Denk. 11, 12, 13a, 13b)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FuzzyOutput:
    """Tek düğüm çıktısı (§5.14): score = f_norm (Denk. 13a), confidence (Denk. 13b), en baskın kural; f ham çıkış."""

    score: float
    confidence: float
    top_rule_index: int
    f: float

    def to_score_output(self) -> ScoreOutput:
        return ScoreOutput(score=self.score, confidence=self.confidence, top_rule=self.top_rule_index)


@dataclass(frozen=True)
class FuzzyBatchOutput:
    """Toplu çıktı: B aday için diziler; w_norm (B, K) normalize ateşleme güçleri w_k/Σw (analiz için)."""

    score: np.ndarray
    confidence: np.ndarray
    top_rule_index: np.ndarray
    f: np.ndarray
    w_norm: np.ndarray

    def __getitem__(self, b: int) -> FuzzyOutput:
        return FuzzyOutput(float(self.score[b]), float(self.confidence[b]), int(self.top_rule_index[b]), float(self.f[b]))


class FuzzyEngine:
    """§5.14 bulanık çıkarım motoru (NumPy, vektörize). params verilmezse uzman parametreleri (B3, §5.15)."""

    def __init__(self, spec: FuzzySpec, params: FuzzyParams | None = None, pruned: bool = False, prune_tol: float = 1e-9) -> None:
        self.spec = spec
        params = (params or FuzzyParams.expert(spec)).copy()
        if pruned:
            table, y = prune_rules(params.table, params.y, spec.n_terms, prune_tol)
            params = FuzzyParams(params.centers, params.sigmas, y, table)
        if params.table.shape[1] != len(FEATURE_NAMES) or len(params.y) != len(params.table):
            raise ValueError("kural tablosu (K, 5) ve y (K,) uyumlu olmalı")
        if any(np.any(s <= 0) for s in params.sigmas):
            raise ValueError("σ > 0 olmalı")
        self.params = params
        self.pruned = pruned
        # DONT_CARE (−1) -> üyelik matrisinde ek "hep 1" sütununun indeksi (n_i)
        self._cols = [np.where(params.table[:, i] == DONT_CARE, n, params.table[:, i]) for i, n in enumerate(spec.n_terms)]

    @classmethod
    def from_config(cls, cfg: Config, params: FuzzyParams | None = None, pruned: bool | None = None) -> "FuzzyEngine":
        """Config'ten motor; pruned verilmezse config fuzzy.rules.pruned (§5.14)."""
        if pruned is None:
            pruned = bool(cfg.fuzzy["rules"].get("pruned", False))
        return cls(FuzzySpec.from_config(cfg), params, pruned)

    @property
    def n_rules(self) -> int:
        return len(self.params.y)

    # ------------------------------------------------------------------ çekirdek
    def memberships(self, X: np.ndarray) -> list[np.ndarray]:
        """Girdi başına (B, n_i) üyelik dereceleri, Denk. 9."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return [gaussian_mf(X[:, [i]], self.params.centers[i][None, :], self.params.sigmas[i][None, :])
                for i in range(len(FEATURE_NAMES))]

    def log_firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """(B, K) log w_k = Σ_i log μ_{i,term(k,i)}(x_i) = −Σ_i ((x_i − c)/σ)² (Denk. 9 + 11); don't care -> 0."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        logw = np.zeros((X.shape[0], self.n_rules))
        for i in range(len(FEATURE_NAMES)):
            z = -(((X[:, [i]] - self.params.centers[i][None, :]) / self.params.sigmas[i][None, :]) ** 2)
            z1 = np.concatenate([z, np.zeros((z.shape[0], 1))], axis=1)
            logw += z1[:, self._cols[i]]
        return logw

    def firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """(B, K) ateşleme güçleri w_k = Π_i μ_{i,term(k,i)}(x_i) (Denk. 11, çarpım t-normu); don't-care üyeliği 1."""
        return np.exp(self.log_firing_strengths(X))

    def evaluate_batch(self, X: np.ndarray | Sequence[NodeFeatures]) -> FuzzyBatchOutput:
        """(B, 5) girdi -> skor, güven, en baskın kural (Denk. 11, 12, 13a, 13b).

        Sayısal kararlılık: normalize ağırlıklar w_k/Σw log uzayında hesaplanır (softmax(log w)); matematik aynıdır,
        ama küçük σ'larda (eğitim) bütün w_k'lerin 0'a taşması ve 0/0 oluşması önlenir.
        """
        if len(X) and isinstance(X[0], NodeFeatures):
            X = features_matrix(list(X))
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"girdi (B, 5) olmalı, {X.shape} verildi")
        logw = self.log_firing_strengths(X)
        top = np.argmax(logw, axis=1)
        e = np.exp(logw - logw[np.arange(len(logw)), top][:, None])          # en büyük terim = 1
        s = e.sum(axis=1)
        w_norm = e / s[:, None]                                              # w_k / Σ w
        f = w_norm @ self.params.y                                           # Denk. 12
        f_norm = np.clip((f - self.spec.y_min) / self.spec.y_span, 0.0, 1.0)  # Denk. 13a
        confidence = 1.0 / s                                                 # Denk. 13b: max w / Σ w
        return FuzzyBatchOutput(f_norm, confidence, top, f, w_norm)

    def evaluate(self, features: NodeFeatures | Sequence[float] | np.ndarray) -> FuzzyOutput:
        """Tek düğüm: evaluate(features) -> FuzzyOutput(score, confidence, top_rule_index) (§5.14)."""
        x = features.as_array() if isinstance(features, NodeFeatures) else np.asarray(features, dtype=float)
        return self.evaluate_batch(x.reshape(1, -1))[0]

    # ------------------------------------------------------------------ yorumlanabilirlik
    def rule_terms(self, k: int) -> tuple[str | None, ...]:
        """Kural k'nin girdi başına terim adı (don't care -> None)."""
        row = self.params.table[k]
        return tuple(None if j == DONT_CARE else TERM_LABELS[n][j] for j, n in zip(row, self.spec.n_terms))

    def rule_to_text(self, k: int, with_labels: bool = True) -> str:
        """İnsan okunur kural: 'IF x1 (demand) is High AND ... THEN 0.82' (don't-care girdiler yazılmaz)."""
        parts = []
        for name, term in zip(FEATURE_NAMES, self.rule_terms(k)):
            if term is None:
                continue
            label = f"{name} ({FEATURE_LABELS[name]})" if with_labels else name
            parts.append(f"{label} is {term}")
        cond = " AND ".join(parts) if parts else "TRUE"
        return f"R{k}: IF {cond} THEN {self.params.y[k]:.2f}"

    def rule_index(self, terms: Sequence[int]) -> int:
        """Terim indeksleri (x1..x5) -> kural indeksi (tam tabloda)."""
        hits = np.flatnonzero((self.params.table == np.asarray(terms)).all(axis=1))
        if len(hits) != 1:
            raise KeyError(f"terimler {tuple(terms)} için tek kural bulunamadı")
        return int(hits[0])

    def term_centers_input(self, terms: Sequence[int]) -> np.ndarray:
        """Verilen terimlerin merkezlerinde girdi vektörü (kalibrasyon raporu: kural girdileri terim merkezlerinde)."""
        return np.array([self.params.centers[i][j] for i, j in enumerate(terms)])


# --------------------------------------------------------------------------- #
# Girdi çıkarımı ve sabit normalizasyon (§5.14 girdi tablosu)
# --------------------------------------------------------------------------- #


def nominal_rates_pps(cfg: Config, net: Any) -> dict[int, float]:
    """Düğüm başına beklenen üretim hızı (§5.6 expected_rate_pps; epizot ağırlıklı). x1 için varsayılan hız kaynağı."""
    from nfmtar.traffic import expected_rate_pps

    return {v: expected_rate_pps(cfg.traffic, net.nodes[v].traffic_class) for v in net.sensor_ids}


def extract_features(
    spec: FuzzySpec,
    net: Any,
    part: Any,
    topology: Any,
    rates_pps: Mapping[int, float],
    prr: Callable[[float], float],
    queue_occupancy: Mapping[int, float] | None = None,
    hops_to_edge: Mapping[int, int] | None = None,
) -> dict[int, NodeFeatures]:
    """Topolojinin canlı sensörleri için NodeFeatures (§5.14); saf fonksiyon (ağı değiştirmez).

    net: PhysicalNetwork; part: Partition; rates_pps: düğüm -> üretim hızı (ölçülen veya `nominal_rates_pps`);
    prr: mesafe -> PRR (RadioModel.prr); queue_occupancy: düğüm -> kuyruk doluluk oranı [0,1] (simülasyondan; yoksa 0);
    hops_to_edge: düğüm -> edge router'a en az atlama (router.EdgeReach.hops; yoksa None).
      x1 = Σ hız(kapalı 2-atlama komşuluğu) / Σ hız(topolojinin canlı sensörleri)
      x2 = kuyruk doluluğu, [0,1]'e kırpılır
      x3 = 1 − kalan/başlangıç enerji (NodeState.depletion)
      x4 = topoloji içi canlı komşulara ortalama PRR (komşu yoksa 0)
      x5 = min(1, topoloji içi canlı komşu sayısı / max_neighbors)
    """
    import networkx as nx

    sub = part.subgraph(net, topology, alive_only=True).copy()
    edge_id = part.edge_router_ids[topology]
    if edge_id in sub:
        sub.remove_node(edge_id)          # topoloji içi komşu = canlı sensörler (edge router komşu sayılmaz)
    members = sorted(sub.nodes)
    total_rate = sum(float(rates_pps.get(v, 0.0)) for v in members)
    occupancy = queue_occupancy or {}
    hops = hops_to_edge or {}
    out: dict[int, NodeFeatures] = {}
    for v in members:
        near = nx.single_source_shortest_path_length(sub, v, cutoff=spec.neighborhood_hops)
        x1 = sum(float(rates_pps.get(u, 0.0)) for u in near) / total_rate if total_rate > 0 else 0.0
        nbrs = list(sub.neighbors(v))
        x4 = float(np.mean([prr(net.distance(v, u)) for u in nbrs])) if nbrs else 0.0
        out[v] = NodeFeatures(
            node_id=v,
            x1=min(1.0, max(0.0, x1)),
            x2=min(1.0, max(0.0, float(occupancy.get(v, 0.0)))),
            x3=float(net.nodes[v].depletion),
            x4=min(1.0, max(0.0, x4)),
            x5=min(1.0, len(nbrs) / spec.max_neighbors),
            hops_to_edge=hops.get(v),
        )
    return out
