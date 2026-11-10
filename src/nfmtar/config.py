"""YAML -> dataclass yapılandırma yükleyici (CLAUDE.md §3, §7).

Uygulanan kararlar:
- §7: Tüm parametreler configs/default.yaml'dan okunur; koda sabit değer gömülmez.
  Bilinmeyen, eksik veya yanlış tipli anahtarlar yükleme anında ConfigError üretir
  (yazım hataları sessizce varsayılana düşmesin diye).
- §3: Seed listesi ve tekrarlanabilirlik bayrakları config'dedir (§5.19).
- Her bölüm (network §5.1, placement §5.2, partition §5.3, edge_routers §5.4,
  radio/energy §5.5, traffic §5.6, policy §5.7) kendi dataclass'ına eşlenir.
  router (§5.9) yalnızca relay_mode için erken dönüştürüldü. Henüz kodlanmamış bölümler
  (controller, engine, fuzzy, oracle, training, rank, metrics, experiments, contribution)
  ham sözlük olarak taşınır ve ilgili modül yazılırken dataclass'a dönüştürülür (§7).
"""

from __future__ import annotations

import copy
import types
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

TOPOLOGY_NAMES: tuple[str, ...] = ("E", "L", "R")  # §5.2: Eliptik, Doğrusal, Rastgele


class ConfigError(ValueError):
    """Yapılandırmadaki eksik / fazla / yanlış tipli anahtar hatası (§7)."""


# --------------------------------------------------------------------------- #
# Bölüm dataclass'ları
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PacketLogConfig:
    """Opsiyonel paket logu ayarları (§5.19)."""

    enabled: bool
    format: str


@dataclass(frozen=True)
class RunConfig:
    """Genel koşu düzeni: seed listesi, ısınma süresi, çıktı yolları (§5.19, §7)."""

    seeds: tuple[int, ...]
    warm_up_s: float
    results_dir: str
    runs_table: str
    figures_dir: str
    models_dir: str
    data_dir: str
    packet_log: PacketLogConfig


@dataclass(frozen=True)
class SimulationConfig:
    """Ayrık olay simülasyonu ayarları (§3, §5.10)."""

    duration_s: float
    event_queue: str
    reproducible: bool


@dataclass(frozen=True)
class NetworkConfig:
    """Fiziksel ağ parametreleri: alan, düğüm sayısı, menzil, bağlılık (§5.1)."""

    area_m: tuple[float, float]
    n_nodes: int
    n_nodes_range: tuple[int, int]
    radio_range_m: float
    topology_model: str
    enforce_partition_connectivity: bool  # §5.1 yeniden üretim ölçütü; false = E0, yalnızca ölç
    max_regen_attempts: int


@dataclass(frozen=True)
class EllipticPlacement:
    """Eliptik (kritik) bölge: alan oranı cinsinden merkez ve yarı eksenler (§5.2)."""

    center_frac: tuple[float, float]
    semi_axes_frac: tuple[float, float]


@dataclass(frozen=True)
class LinearPlacement:
    """Doğrusal (düzenli) koridor: eksen, konum ve şerit genişliği (§5.2)."""

    axis: str
    center_frac: float
    width_m: float

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ConfigError(f"placement.linear.axis 'x' veya 'y' olmalı, {self.axis!r} geldi")


@dataclass(frozen=True)
class RandomPlacement:
    """Rastgele (kritik olmayan) yerleşim (§5.2): region (varsayılan) veya uniform (yalnızca E0).

    region: Edge-R'ye bitişik dikdörtgen, alanı A_R = n_R · π · R² / r_target_degree.
    """

    r_placement: str
    r_target_degree: float
    region_aspect: float

    def __post_init__(self) -> None:
        if self.r_placement not in ("region", "uniform"):
            raise ConfigError(
                f"placement.random.r_placement 'region' veya 'uniform' olmalı, {self.r_placement!r} geldi"
            )
        if self.r_target_degree <= 0 or self.region_aspect <= 0:
            raise ConfigError("placement.random.r_target_degree ve region_aspect pozitif olmalı")


@dataclass(frozen=True)
class PlacementConfig:
    """Yerleşim: sınıf oranları ve üç bölge tanımı (§5.2)."""

    class_ratio: dict[str, float]
    elliptic: EllipticPlacement
    linear: LinearPlacement
    random: RandomPlacement

    def __post_init__(self) -> None:
        if set(self.class_ratio) != set(TOPOLOGY_NAMES):
            raise ConfigError(
                f"placement.class_ratio anahtarları {TOPOLOGY_NAMES} olmalı, "
                f"{sorted(self.class_ratio)} geldi"
            )
        total = sum(self.class_ratio.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"placement.class_ratio toplamı 1 olmalı, {total} geldi")


@dataclass(frozen=True)
class PartitionConfig:
    """Bölümleme kuralları ve bağlantısallık kontrolü bayrakları (§5.3)."""

    assignment: str
    single_class_per_node: bool
    disjoint: bool
    overlap_capable_structure: bool
    check_edge_connectivity: bool
    log_disconnected_count: bool


@dataclass(frozen=True)
class EdgeRouterConfig:
    """Sabit edge router'lar ve root ayrımı (§5.4)."""

    fixed: bool
    positions_m: dict[str, tuple[float, float]]
    root_separate_from_edge: bool
    two_stage_path: bool
    root_aggregation: bool

    def __post_init__(self) -> None:
        if set(self.positions_m) != set(TOPOLOGY_NAMES):
            raise ConfigError(
                f"edge_routers.positions_m anahtarları {TOPOLOGY_NAMES} olmalı, "
                f"{sorted(self.positions_m)} geldi"
            )


@dataclass(frozen=True)
class CsmaConfig:
    """CSMA/CA geri çekilme parametreleri (§5.5)."""

    backoff_slot_s: float
    max_backoff_slots: int


@dataclass(frozen=True)
class PrrConfig:
    """Mesafeye bağlı lojistik PRR modeli (§5.5) ve PRR tabanı (§5.9)."""

    model: str
    prr_max: float
    d_mid_m: float
    steepness_per_m: float
    prr_min: float


@dataclass(frozen=True)
class RadioConfig:
    """IEEE 802.15.4 iletim parametreleri (§5.5)."""

    bitrate_bps: int
    max_frame_bytes: int
    payload_per_frame_bytes: int
    fragment_loss_drops_packet: bool
    csma: CsmaConfig
    processing_delay_s: float
    shared_channel: bool
    delay_components: tuple[str, ...]
    prr: PrrConfig


@dataclass(frozen=True)
class EnergyConfig:
    """Birinci dereceden radyo enerji modeli ve başlangıç enerjisi (§5.5)."""

    e_elec_nJ_per_bit: float
    eps_amp_pJ_per_bit_m2: float
    initial_J: float
    dead_threshold_J: float


@dataclass(frozen=True)
class TrafficClassConfig:
    """Bir trafik sınıfının topolojisi, paket boyutu (sabit veya aralık) ve hızı (§5.6)."""

    topology: str
    packet_bytes: int | tuple[int, int]
    rate_pps: float


@dataclass(frozen=True)
class TrafficConfig:
    """Trafik sınıfları ve yük ölçeği (§5.6); sınıf -> topoloji eşlemesi one-hot (§5.3)."""

    load_multiplier: float
    arrival_process: str
    classes: dict[str, TrafficClassConfig]

    def __post_init__(self) -> None:
        topologies = [c.topology for c in self.classes.values()]
        if sorted(topologies) != sorted(TOPOLOGY_NAMES):
            raise ConfigError(
                "traffic.classes her topolojiye tam bir sınıf atamalı (§5.3 one-hot); "
                f"{topologies} geldi"
            )

    def class_for_topology(self, topology: str) -> str:
        """Topolojinin tek trafik sınıfını döndürür (§5.3: bire bir eşleme)."""
        for name, cls in self.classes.items():
            if cls.topology == topology:
                return name
        raise KeyError(topology)


@dataclass(frozen=True)
class TopologyPolicy:
    """Tablo 1 politika satırı: π1–π6, π9–π12 (§5.7; π7/π8 kapsam dışı)."""

    bandwidth_share_pct: float
    priority: int
    max_delay_ms: float
    min_throughput_pct: float
    loss_tolerance_pct: float
    energy_weight: float
    buffer_packets: int
    retransmit_limit: int
    route_update_s: float
    qos_monitor_ms: float


@dataclass(frozen=True)
class PolicyConfig:
    """Üç topolojinin Tablo 1 politikaları ve QoS memnuniyet eşik listesi (§5.7)."""

    E: TopologyPolicy
    L: TopologyPolicy
    R: TopologyPolicy
    qos_satisfaction_thresholds: tuple[str, ...]

    def for_topology(self, topology: Any) -> TopologyPolicy:
        """Topoloji adına (str veya .value taşıyan Enum) göre politika satırı (§5.7)."""
        name = getattr(topology, "value", topology)
        if name not in TOPOLOGY_NAMES:
            raise KeyError(f"bilinmeyen topoloji {topology!r}")
        return getattr(self, name)


@dataclass(frozen=True)
class B1RouterConfig:
    """B1 yol maliyeti: atlama sayısı (§5.9)."""

    edge_cost: str


@dataclass(frozen=True)
class B2RouterConfig:
    """B2 yol maliyeti: ETX + depletion, d² terimi yok (§5.9)."""

    link_metric: str
    lambda1: float


@dataclass(frozen=True)
class RouterConfig:
    """Yol seçimi parametreleri (§5.9): λ ağırlıkları, PRR tabanı, algoritma ve relay_mode.

    relay_mode: strict (varsayılan; yol yalnızca kendi topolojisinin alt çizgesinde) veya shared
    (yol fiziksel ağda, diğer topolojiler yalnızca aktarıcı; sadece ablasyon/katkı ve E0).
    router.py yazılana kadar yalnızca §5.1 yeniden üretim ölçütünü etkiler.
    """

    lambda1: float
    lambda2: float
    lambda3_from_policy: bool
    normalize_lambdas: bool
    prr_min: float
    algorithm: str
    relay_mode: str
    b1: B1RouterConfig
    b2: B2RouterConfig

    def __post_init__(self) -> None:
        if self.relay_mode not in ("strict", "shared"):
            raise ConfigError(f"router.relay_mode 'strict' veya 'shared' olmalı, {self.relay_mode!r} geldi")


@dataclass(frozen=True)
class Config:
    """Tüm yapılandırma; configs/default.yaml'ın üst düzey bölümleri (§7)."""

    run: RunConfig
    simulation: SimulationConfig
    network: NetworkConfig
    placement: PlacementConfig
    partition: PartitionConfig
    edge_routers: EdgeRouterConfig
    radio: RadioConfig
    energy: EnergyConfig
    traffic: TrafficConfig
    policy: PolicyConfig
    router: RouterConfig
    # Henüz dataclass'a dönüştürülmemiş bölümler (ilgili modül yazılırken dönüştürülür, §7):
    controller: dict[str, Any]
    engine: dict[str, Any]
    fuzzy: dict[str, Any]
    oracle: dict[str, Any]
    training: dict[str, Any]
    rank: dict[str, Any]
    metrics: dict[str, Any]
    experiments: dict[str, Any]
    contribution: dict[str, Any]
    source_path: str | None = field(default=None, metadata={"internal": True})


# --------------------------------------------------------------------------- #
# Sözlük -> dataclass dönüştürücü
# --------------------------------------------------------------------------- #


def _build(tp: Any, value: Any, path: str) -> Any:
    """Tip ipucuna göre YAML değerini doğrulayıp dönüştürür (§7: hatalar erken yakalanır)."""
    if tp is Any:
        return value
    if isinstance(tp, type) and is_dataclass(tp):
        return _build_dataclass(tp, value, path)
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        errors: list[str] = []
        for alt in args:
            if alt is type(None):
                if value is None:
                    return None
                continue
            try:
                return _build(alt, value, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: hiçbir tip uymadı ({' | '.join(errors)})")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: liste bekleniyordu, {type(value).__name__} geldi")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{path}: {len(args)} elemanlı liste bekleniyordu, {len(value)} geldi")
        return tuple(_build(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: sözlük bekleniyordu, {type(value).__name__} geldi")
        key_t, val_t = args
        return {
            _build(key_t, k, f"{path}.{k}"): _build(val_t, v, f"{path}.{k}") for k, v in value.items()
        }
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: true/false bekleniyordu, {value!r} geldi")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: tam sayı bekleniyordu, {value!r} geldi")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: sayı bekleniyordu, {value!r} geldi")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: metin bekleniyordu, {value!r} geldi")
        return value
    if tp is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: sözlük bekleniyordu, {type(value).__name__} geldi")
        return copy.deepcopy(value)
    raise ConfigError(f"{path}: desteklenmeyen tip ipucu {tp!r}")


def _build_dataclass(cls: type, value: Any, path: str) -> Any:
    """Bir YAML bölümünü dataclass'a çevirir; fazla/eksik anahtarda ConfigError (§7)."""
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: bölüm (sözlük) bekleniyordu, {type(value).__name__} geldi")
    hints = typing.get_type_hints(cls)
    yaml_fields = [f for f in fields(cls) if not f.metadata.get("internal")]
    known = {f.name for f in yaml_fields}
    unknown = set(value) - known
    if unknown:
        raise ConfigError(
            f"{path}: bilinmeyen anahtar(lar) {sorted(unknown)}; izin verilenler {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for f in yaml_fields:
        if f.name not in value:
            if f.default is not MISSING or f.default_factory is not MISSING:
                continue
            raise ConfigError(f"{path}.{f.name}: zorunlu anahtar eksik")
        kwargs[f.name] = _build(hints[f.name], value[f.name], f"{path}.{f.name}")
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Yükleme ve geçersiz kılma
# --------------------------------------------------------------------------- #


def parse_override(text: str) -> tuple[str, Any]:
    """'network.n_nodes=200' biçimindeki komut satırı argümanını (yol, değer) çiftine çevirir.

    Değer YAML olarak ayrıştırılır (200 -> int, 1.5 -> float, true -> bool, "[1,2]" -> liste),
    böylece deney scriptleri default.yaml'ı değiştirmeden parametre tarayabilir (§5.19, §7).
    """
    if "=" not in text:
        raise ConfigError(f"geçersiz override {text!r}; 'bolum.anahtar=deger' bekleniyor")
    key, raw_value = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw_value)


def apply_overrides(raw: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Noktalı yol -> değer eşlemesini ham YAML sözlüğüne uygular (kopya döndürür).

    Var olmayan bir yol ConfigError verir: deney scriptlerindeki yazım hataları sessizce
    yeni anahtar oluşturmasın (§7).
    """
    result = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node: Any = result
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"override yolu bulunamadı: {dotted!r} ({part!r} yok)")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ConfigError(f"override yolu bulunamadı: {dotted!r} ({parts[-1]!r} yok)")
        node[parts[-1]] = value
    return result


def config_from_dict(raw: Mapping[str, Any], source_path: str | None = None) -> Config:
    """Ham sözlükten Config üretir; tüm doğrulamalar burada çalışır (§7)."""
    cfg = _build_dataclass(Config, dict(raw), "config")
    if source_path is not None:
        object.__setattr__(cfg, "source_path", source_path)
    return cfg


def load_raw(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """YAML dosyasını ham sözlük olarak okur (UTF-8)."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: üst düzeyde sözlük bekleniyordu")
    return raw


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """configs/default.yaml (veya verilen dosya) + opsiyonel override'lar -> Config (§7).

    Örnek: load_config(overrides={"network.n_nodes": 200, "energy.initial_J": 0.5})
    """
    raw = load_raw(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return config_from_dict(raw, source_path=str(path))
