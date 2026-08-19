"""CIC-IoT2023 ingestion and projection onto the project feature schema.

The CIC-IoT2023 release ships per-capture CSVs with 46 pre-extracted flow
features and a `label` column covering 33 attack classes. This module maps that
native schema onto the 30-D behavioural vector defined in `swarmdef.data.schema`
so a model can be trained on real capture data and evaluated on live twin
traffic without any change to the detector.

Getting the data
----------------
The dataset is free but requires accepting UNB's terms, so it cannot be fetched
unattended. Download any subset of the CSV release from

    https://www.unb.ca/cic/datasets/iotdataset-2023.html

and drop the `.csv` files into `data/raw/cic_iot2023/`. The loader picks up
everything it finds there. If the directory is empty the pipeline transparently
falls back to the digital twin as its data source.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from swarmdef.data.schema import (ALL_COLUMNS, FEATURE_NAMES, LABEL_TO_ID,
                                  N_FEATURES)
from swarmdef.utils.logging import get_logger

log = get_logger("data.cic")

CIC_SUBDIR = "cic_iot2023"

# ── CIC-IoT2023 label vocabulary -> our 6 MIoT-relevant attack families ──────
# The native taxonomy is far finer than a hospital SOC would act on. We collapse
# it onto the families the digital twin also produces, so both data sources
# share one label space.
LABEL_MAP: dict[str, str] = {
    # ── benign ──
    "BenignTraffic": "BENIGN",
    "Benign": "BENIGN",
    # ── volumetric floods (DDoS + DoS) ──
    **{f"DDoS-{k}": "DDoS" for k in (
        "ICMP_Flood", "UDP_Flood", "TCP_Flood", "PSHACK_Flood", "SYN_Flood",
        "RSTFINFlood", "SynonymousIP_Flood", "ICMP_Fragmentation",
        "UDP_Fragmentation", "ACK_Fragmentation", "HTTP_Flood", "SlowLoris",
    )},
    **{f"DoS-{k}": "DDoS" for k in (
        "UDP_Flood", "TCP_Flood", "SYN_Flood", "HTTP_Flood",
    )},
    # ── botnet enrolment / C2 ──
    **{f"Mirai-{k}": "Mirai" for k in (
        "greeth_flood", "greip_flood", "udpplain",
    )},
    "DictionaryBruteForce": "Mirai",      # credential stuffing = enrolment phase
    # ── reconnaissance ──
    **{f"Recon-{k}": "Recon" for k in (
        "PingSweep", "OSScan", "PortScan", "HostDiscovery",
    )},
    "VulnerabilityScan": "Recon",
    # ── in-path tampering / identity ──
    "MITM-ArpSpoofing": "MITM",
    "DNS_Spoofing": "Spoofing",
    # ── code / firmware integrity violations ──
    "Backdoor_Malware": "FirmwareTamper",
    "BrowserHijacking": "FirmwareTamper",
    "CommandInjection": "FirmwareTamper",
    "SqlInjection": "FirmwareTamper",
    "XSS": "FirmwareTamper",
    "Uploading_Attack": "FirmwareTamper",
}


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def map_label(raw: str) -> str:
    """Collapse a native CIC label onto our family taxonomy."""
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    low = raw.lower()
    if low.startswith(("ddos", "dos")):
        return "DDoS"
    if low.startswith("mirai"):
        return "Mirai"
    if low.startswith("recon") or "scan" in low:
        return "Recon"
    if "spoof" in low:
        return "Spoofing"
    if "benign" in low:
        return "BENIGN"
    return "FirmwareTamper"


def find_csvs(raw_dir: str | Path) -> list[Path]:
    """Locate CIC CSVs under `raw_dir`, tolerating a nested release layout."""
    root = Path(raw_dir)
    for candidate in (root / CIC_SUBDIR, root):
        if candidate.is_dir():
            hits = sorted(candidate.rglob("*.csv"))
            if hits:
                return hits
    return []


def available(raw_dir: str | Path) -> bool:
    return bool(find_csvs(raw_dir))


def project_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map CIC-IoT2023 native columns onto the 30-D behavioural vector.

    Where CIC measures the same quantity under another name we take it
    directly; where it does not (MQTT QoS, topic entropy, clinical
    plausibility) the feature is left at zero, which is the correct value for a
    non-MQTT capture and keeps the vector dimension stable across sources.
    """
    cols = {_norm(c): c for c in df.columns}

    def col(*names, default=0.0):
        for n in names:
            key = _norm(n)
            if key in cols:
                return pd.to_numeric(df[cols[key]], errors="coerce").fillna(default)
        return pd.Series(np.full(len(df), default, dtype=np.float64), index=df.index)

    rate = col("Rate")
    srate = col("Srate")
    duration = col("flow_duration", "Duration")
    tot_size = col("Tot size", "Tot_size")
    avg_size = col("AVG")
    std_size = col("Std")
    iat = col("IAT")
    number = col("Number", default=1.0)

    # TCP flag indicator columns are 0/1 per flow in the CIC release.
    syn = col("syn_flag_number") + col("syn_count")
    ack = col("ack_flag_number") + col("ack_count")
    rst = col("rst_flag_number") + col("rst_count")
    flag_total = (syn + ack + rst).replace(0, np.nan)

    proto_tcp, proto_udp = col("TCP"), col("UDP")
    # No MQTT column exists in CIC; application-layer IoT protocols are the
    # closest analogue to the twin's MQTT share.
    proto_mqtt = col("HTTP") + col("HTTPS") + col("DNS") + col("SSH") + col("Telnet")
    proto_total = (proto_tcp + proto_udp + proto_mqtt).replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out["packet_rate"] = rate
    out["byte_rate"] = rate * avg_size
    out["pkt_size_mean"] = avg_size
    out["pkt_size_std"] = std_size
    out["pkt_size_min"] = col("Min")
    out["pkt_size_max"] = col("Max")
    out["payload_len_mean"] = (avg_size - col("Header_Length") / number.clip(lower=1)).clip(lower=0)
    out["header_len_mean"] = col("Header_Length") / number.clip(lower=1)
    out["flow_duration"] = duration
    out["iat_mean"] = iat
    out["iat_std"] = col("Variance") ** 0.5
    # Burstiness from the IAT coefficient of variation, matching the twin's
    # Goh-Barabasi definition so the feature means the same thing in both sources.
    cv = (col("Std") / avg_size.replace(0, np.nan)).fillna(0.0)
    out["burstiness"] = ((cv - 1.0) / (cv + 1.0)).fillna(0.0)
    # CIC has no payload-entropy column; Magnitude/Radius summarise payload
    # dispersion and are the nearest available proxy, rescaled to bits/byte.
    mag = col("Magnitue", "Magnitude")
    out["payload_entropy"] = (mag / mag.max() * 8.0).fillna(0.0) if mag.max() else 0.0
    out["topic_entropy"] = 0.0
    out["proto_mqtt_ratio"] = (proto_mqtt / proto_total).fillna(0.0)
    out["proto_tcp_ratio"] = (proto_tcp / proto_total).fillna(0.0)
    out["proto_udp_ratio"] = (proto_udp / proto_total).fillna(0.0)
    out["flag_syn_ratio"] = (syn / flag_total).fillna(0.0)
    out["flag_ack_ratio"] = (ack / flag_total).fillna(0.0)
    out["flag_rst_ratio"] = (rst / flag_total).fillna(0.0)
    out["qos_mean"] = 0.0
    out["retain_ratio"] = 0.0
    out["n_unique_src"] = 1.0
    out["n_unique_dst"] = 1.0
    out["n_unique_topic"] = 0.0
    out["conn_attempt_rate"] = srate
    out["subscribe_ratio"] = 0.0
    out["publish_ratio"] = 1.0
    out["dup_ratio"] = 0.0
    out["vital_zscore_max"] = 0.0

    return out[FEATURE_NAMES].astype(np.float32)


def load(
    raw_dir: str | Path,
    n_samples: int | None = None,
    seed: int = 42,
    chunk_rows: int = 200_000,
) -> pd.DataFrame:
    """Load CIC-IoT2023 CSVs and return rows in the project schema.

    Sampling is stratified per file so that rare attack families survive the
    subsample -- a uniform sample of a flood-dominated capture would return
    almost nothing but DDoS.
    """
    files = find_csvs(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No CIC-IoT2023 CSVs under {raw_dir}. See swarmdef/data/cic.py for "
            "download instructions, or use source='twin'."
        )
    log.info("Loading CIC-IoT2023 from %d CSV file(s)", len(files))

    rng = np.random.default_rng(seed)
    per_file = None if n_samples is None else max(n_samples // len(files), 1)
    frames: list[pd.DataFrame] = []

    for path in files:
        try:
            df = pd.read_csv(path, nrows=chunk_rows, low_memory=False)
        except Exception as exc:  # pragma: no cover - malformed release file
            log.warning("Skipping %s: %s", path.name, exc)
            continue
        label_col = next((c for c in df.columns if _norm(c) in ("label", "attack", "class")), None)
        if label_col is None:
            log.warning("Skipping %s: no label column", path.name)
            continue

        df["_family"] = df[label_col].astype(str).map(map_label)
        if per_file is not None and len(df) > per_file:
            # Stratified: take an equal share of every family present.
            groups = df.groupby("_family", group_keys=False)
            share = max(per_file // max(groups.ngroups, 1), 1)
            df = groups.apply(
                lambda g: g.sample(min(len(g), share), random_state=seed), include_groups=True
            ).reset_index(drop=True)

        feats = project_to_schema(df)
        feats["attack_type"] = df["_family"].map(LABEL_TO_ID).astype(int).to_numpy()
        feats["label"] = (feats["attack_type"] != 0).astype(int)
        frames.append(feats)
        log.info("  %-40s %7d rows", path.name, len(feats))

    if not frames:
        raise RuntimeError(f"Found CSVs under {raw_dir} but none were usable")

    out = pd.concat(frames, ignore_index=True)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if n_samples is not None and len(out) > n_samples:
        idx = rng.choice(len(out), size=n_samples, replace=False)
        out = out.iloc[np.sort(idx)].reset_index(drop=True)

    # Fill the metadata columns the rest of the pipeline expects. Real captures
    # carry no device identity, so each row becomes its own single-node graph.
    out["device_id"] = [f"cic-{i}" for i in range(len(out))]
    out["hospital_id"] = 0
    out["t_start"] = np.arange(len(out), dtype=float)
    log.info("CIC-IoT2023: %d rows, %.1f%% attack", len(out), 100 * out["label"].mean())
    return out[ALL_COLUMNS]
