"""Typed configuration for the whole pipeline.

Every experiment is fully described by a `Config` object, which is serialised
alongside its results so any run can be reproduced exactly.

Load order (later wins):
    dataclass defaults  ->  YAML file  ->  dotted CLI overrides (--set a.b=c)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"
MODEL_DIR = RESULTS_DIR / "models"


# ───────────────────────────── digital twin ──────────────────────────────────
@dataclass
class TwinConfig:
    """MQTT digital-twin simulation (Phase 1 / Step 2)."""

    broker_host: str = "127.0.0.1"
    broker_port: int = 1883
    topic_root: str = "hospital"
    n_hospitals: int = 4
    devices_per_hospital: int = 5
    publish_hz: float = 5.0          # telemetry messages per device per second
    duration_s: float = 60.0         # 0 => run forever
    attack_probability: float = 0.0  # background chance an attack fires per tick
    seed: int = 42


# ────────────────────────────── dataset ──────────────────────────────────────
@dataclass
class DataConfig:
    """Dataset ingestion, feature engineering and non-IID sharding (Step 3)."""

    source: str = "auto"             # auto | cic | synthetic | twin
    raw_dir: str = str(DATA_DIR / "raw")
    processed_dir: str = str(DATA_DIR / "processed")
    hospital_dir: str = str(DATA_DIR / "hospitals")
    n_samples: int = 24_000          # rows kept for the experiment
    n_hospitals: int = 4
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    # group -> whole (hospital, time-window) graphs go to one split (inductive,
    #          required for an honest GNN score)
    # row   -> independent per-row stratification (tabular models only)
    split_mode: str = "group"
    # Non-IID control: lower alpha => more skewed attack mix per hospital.
    dirichlet_alpha: float = 0.4
    # native    -> keep the twin's own hospital structure (device-level non-IID)
    # dirichlet -> re-shard by a Dirichlet draw over attack classes
    partition: str = "native"
    # Twin capture settings, used when source resolves to "twin".
    twin_duration_s: float = 600.0
    # 0 => derive the attack count from twin_attack_fraction so that attack
    # density stays constant no matter how long the capture runs.
    twin_attacks_per_device: int = 0
    twin_attack_fraction: float = 0.25
    # Benign confounder events per device (code blue, imaging, reboots).
    twin_events_per_device: float = 7.0
    window_size: int = 8             # traffic windows per graph / transformer seq
    binary_labels: bool = True       # benign vs attack (multi-class also stored)
    seed: int = 42


# ────────────────────────────── detector ─────────────────────────────────────
@dataclass
class DetectorConfig:
    """GNN / Transformer anomaly detector (Phase 2 / Step 4)."""

    arch: str = "gnn"                # gnn | transformer | mlp
    hidden_dim: int = 64
    n_layers: int = 2
    heads: int = 4                   # transformer / GAT attention heads
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    local_epochs: int = 2            # epochs per federated round
    central_epochs: int = 30         # epochs for the centralised baseline
    graph_knn: int = 4               # device-graph connectivity (edges per node)
    seed: int = 42


# ──────────────────────────── adversarial GAN ────────────────────────────────
@dataclass
class GANConfig:
    """Adversarial perturbation engine (Phase 2 / Step 6)."""

    enabled: bool = True
    latent_dim: int = 16
    hidden_dim: int = 64
    lr_g: float = 2e-4
    lr_d: float = 1e-4
    epochs: int = 20                 # generator epochs per federated round
    batch_size: int = 64
    # Perturbation budget: attacks must stay *functional*, so only non-critical
    # features may move, and only by a bounded L-inf amount.
    # L-inf budget in normalised (median/IQR) feature space. 0.5 == half an
    # inter-quartile range per feature, which is a realistic amount of slack for
    # an attacker padding packets or pacing a flood. 0.15 was calibrated away:
    # it is far below the noise floor of these features and produced <1% evasion
    # against every architecture, which measures the budget, not the defence.
    epsilon: float = 0.5
    mutable_fraction: float = 0.5    # fraction of features the attacker may touch
    adv_train_ratio: float = 0.5     # adversarial samples per real attack sample
    warmup_rounds: int = 2           # rounds of clean FL before the GAN engages
    seed: int = 42


# ────────────────────────── differential privacy ─────────────────────────────
@dataclass
class PrivacyConfig:
    """Opacus DP-SGD configuration (Phase 3 / Step 7)."""

    enabled: bool = False
    max_grad_norm: float = 1.0       # L2 clipping threshold C
    noise_multiplier: float = 0.8    # sigma
    target_delta: float = 1e-5
    target_epsilon: float | None = None   # if set, sigma is calibrated to hit it
    accountant: str = "rdp"          # rdp | prv
    seed: int = 42


# ─────────────────────────── federated learning ──────────────────────────────
@dataclass
class FederatedConfig:
    """Flower orchestration + Byzantine-robust aggregation (Phase 4 / Steps 5,8)."""

    n_rounds: int = 15
    n_clients: int = 4
    clients_per_round: float = 1.0   # participation fraction
    aggregator: str = "fedavg"       # fedavg | krum | multikrum | trimmed_mean | median
    trim_ratio: float = 0.25         # trimmed-mean: fraction cut from each tail
    byzantine_clients: int = 0       # number of malicious hospitals
    byzantine_attack: str = "sign_flip"  # sign_flip | gauss | scale | label_flip
    byzantine_scale: float = 10.0    # magnitude for scale / gauss attacks
    krum_f: int | None = None        # assumed #malicious; defaults to byzantine_clients
    seed: int = 42


# ────────────────────────────── evaluation ───────────────────────────────────
@dataclass
class EvalConfig:
    """Metric logging and figure generation."""

    run_name: str = "default"
    log_dir: str = str(LOG_DIR)
    figure_dir: str = str(FIGURE_DIR)
    model_dir: str = str(MODEL_DIR)
    save_models: bool = True
    plot: bool = True


# ──────────────────────────────── root ───────────────────────────────────────
@dataclass
class Config:
    """Root configuration object passed through the entire pipeline."""

    seed: int = 42
    device: str = "auto"             # auto | cpu | cuda | mps
    twin: TwinConfig = field(default_factory=TwinConfig)
    data: DataConfig = field(default_factory=DataConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls()
        _merge_into(cfg, raw)
        cfg.propagate_seed()
        return cfg

    def override(self, dotted: str) -> "Config":
        """Apply a single ``a.b.c=value`` override, parsing the value as YAML."""
        key, _, value = dotted.partition("=")
        if not _:
            raise ValueError(f"override must look like a.b=value, got {dotted!r}")
        target: Any = self
        parts = key.strip().split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise AttributeError(f"unknown config key: {key}")
        setattr(target, leaf, yaml.safe_load(value))
        return self

    def propagate_seed(self) -> "Config":
        """Push the root seed into every sub-config that did not set its own."""
        for f in fields(self):
            sub = getattr(self, f.name)
            if is_dataclass(sub) and hasattr(sub, "seed"):
                sub.seed = self.seed
        # Keep client/hospital counts consistent across layers.
        self.data.n_hospitals = self.federated.n_clients
        self.twin.n_hospitals = self.federated.n_clients
        if self.federated.krum_f is None:
            self.federated.krum_f = self.federated.byzantine_clients
        return self

    # ── serialisation ────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
        return path

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return json.dumps(self.to_dict(), indent=2)


def _merge_into(obj: Any, raw: dict[str, Any]) -> None:
    """Recursively overlay a plain dict onto a dataclass instance."""
    for key, value in raw.items():
        if not hasattr(obj, key):
            raise AttributeError(f"unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_into(current, value)
        else:
            setattr(obj, key, value)


def ensure_dirs(cfg: Config) -> None:
    """Create every output directory the run will write into."""
    for p in (
        cfg.data.raw_dir, cfg.data.processed_dir, cfg.data.hospital_dir,
        cfg.eval.log_dir, cfg.eval.figure_dir, cfg.eval.model_dir,
    ):
        Path(p).mkdir(parents=True, exist_ok=True)
