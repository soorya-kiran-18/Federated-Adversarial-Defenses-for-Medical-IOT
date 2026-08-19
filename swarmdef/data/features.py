"""Flow-feature engineering: packets -> the 30-D behavioural vector (Step 3).

Raw packets are not a useful detector input: a single MQTT PUBLISH tells you
almost nothing, and a DDoS is only visible as a *rate*. So we bin packets into
per-device time windows and summarise each window with the fixed schema in
`swarmdef.data.schema`.

Windowing rule
--------------
Packets are attributed to the device they concern: the source if the source is
a known device, otherwise the destination. Flood traffic aimed at an infusion
pump therefore lands in that pump's window -- which is what lets the graph
detector reason about *which* device is under pressure, not just that the
segment is noisy.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from swarmdef.data.schema import (ALL_COLUMNS, FEATURE_NAMES, LABEL_TO_ID,
                                  N_FEATURES)
from swarmdef.twin.packet import CONNECT, PUBLISH, SUBSCRIBE, Packet

EXTERNAL = "external"


def _entropy(counts) -> float:
    """Shannon entropy in bits over a collection of counts."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def attribute_device(pkt: Packet, known: set[str], gateway: str | None = None) -> str:
    """Which device's flow does this packet belong to?

    Resolution order: an explicit device endpoint wins; failing that, a device
    named in the MQTT topic (this is what catches *spoofed* publishes, whose
    source address is forged but whose topic still names the impersonated
    device); and anything left over is segment traffic observed at the gateway
    -- subnet scans, service sweeps -- which is where reconnaissance lives.
    """
    if pkt.src in known:
        return pkt.src
    if pkt.dst in known:
        return pkt.dst
    if pkt.topic:
        for part in pkt.topic.split("/"):
            if part in known:
                return part
    return gateway if gateway is not None else EXTERNAL


def window_features(packets: list[Packet], t_start: float, t_end: float) -> np.ndarray:
    """Summarise one device-window as the 30-D behavioural vector."""
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    if not packets:
        return vec

    duration = max(t_end - t_start, 1e-3)
    n = len(packets)

    sizes = np.array([p.size for p in packets], dtype=np.float64)
    payloads = np.array([p.payload_len for p in packets], dtype=np.float64)
    headers = np.array([p.header_len for p in packets], dtype=np.float64)
    times = np.sort(np.array([p.t for p in packets], dtype=np.float64))

    # ── timing ───────────────────────────────────────────────────────────────
    iats = np.diff(times) if n > 1 else np.array([duration])
    iat_mean = float(iats.mean()) if iats.size else duration
    iat_std = float(iats.std()) if iats.size else 0.0
    # Goh-Barabasi burstiness B in [-1, 1]: -1 periodic, 0 Poisson, +1 bursty.
    denom = iat_std + iat_mean
    burstiness = float((iat_std - iat_mean) / denom) if denom > 1e-12 else 0.0

    # ── protocol / flag ratios ───────────────────────────────────────────────
    proto = Counter(p.protocol for p in packets)
    flags = Counter(p.flags for p in packets)
    mtypes = Counter(p.msg_type for p in packets)

    # ── topology ─────────────────────────────────────────────────────────────
    topics = Counter(p.topic for p in packets if p.topic)
    srcs = {p.src for p in packets}
    dsts = {p.dst for p in packets}

    # ── payload entropy (mean over packets that carry one) ───────────────────
    ent = [p.payload_entropy() for p in packets]
    ent = [e for e in ent if e > 0]

    values = {
        "packet_rate":      n / duration,
        "byte_rate":        float(sizes.sum()) / duration,
        "pkt_size_mean":    float(sizes.mean()),
        "pkt_size_std":     float(sizes.std()),
        "pkt_size_min":     float(sizes.min()),
        "pkt_size_max":     float(sizes.max()),
        "payload_len_mean": float(payloads.mean()),
        "header_len_mean":  float(headers.mean()),
        "flow_duration":    float(times[-1] - times[0]) if n > 1 else 0.0,
        "iat_mean":         iat_mean,
        "iat_std":          iat_std,
        "burstiness":       burstiness,
        "payload_entropy":  float(np.mean(ent)) if ent else 0.0,
        "topic_entropy":    _entropy(topics.values()),
        "proto_mqtt_ratio": proto.get("MQTT", 0) / n,
        "proto_tcp_ratio":  proto.get("TCP", 0) / n,
        "proto_udp_ratio":  proto.get("UDP", 0) / n,
        "flag_syn_ratio":   flags.get("SYN", 0) / n,
        "flag_ack_ratio":   flags.get("ACK", 0) / n,
        "flag_rst_ratio":   flags.get("RST", 0) / n,
        "qos_mean":         float(np.mean([p.qos for p in packets])),
        "retain_ratio":     sum(p.retain for p in packets) / n,
        "n_unique_src":     float(len(srcs)),
        "n_unique_dst":     float(len(dsts)),
        "n_unique_topic":   float(len(topics)),
        "conn_attempt_rate": mtypes.get(CONNECT, 0) / duration,
        "subscribe_ratio":  mtypes.get(SUBSCRIBE, 0) / n,
        "publish_ratio":    mtypes.get(PUBLISH, 0) / n,
        "dup_ratio":        sum(p.dup for p in packets) / n,
        "vital_zscore_max": max((p.vital_z for p in packets), default=0.0),
    }

    for i, name in enumerate(FEATURE_NAMES):
        vec[i] = values[name]
    return vec


class FlowFeatureExtractor:
    """Bins a packet stream into labelled per-device flow windows."""

    def __init__(self, device_ids: list[str], hospital_id: int, window_s: float = 1.0,
                 gateway_id: str | None = None) -> None:
        self.device_ids = list(device_ids)
        self.known = set(device_ids)
        self.hospital_id = hospital_id
        self.window_s = window_s
        self.gateway_id = gateway_id
        if gateway_id is not None:
            self.known.add(gateway_id)
        # device -> window index -> packets
        self._buffer: dict[str, dict[int, list[Packet]]] = defaultdict(lambda: defaultdict(list))

    def add(self, packets: list[Packet]) -> None:
        for pkt in packets:
            dev = attribute_device(pkt, self.known, self.gateway_id)
            idx = int(pkt.t // self.window_s)
            self._buffer[dev][idx].append(pkt)

    def to_frame(self, drop_external: bool = True, min_packets: int = 2) -> pd.DataFrame:
        """Materialise every buffered window as a labelled dataframe row."""
        rows: list[np.ndarray] = []
        meta: list[tuple] = []

        for dev, windows in self._buffer.items():
            if drop_external and dev == EXTERNAL:
                continue
            for idx, pkts in sorted(windows.items()):
                if len(pkts) < min_packets:
                    continue
                t0 = idx * self.window_s
                vec = window_features(pkts, t0, t0 + self.window_s)
                # A window is malicious if it contains any attack packet.
                labels = [p.label for p in pkts if p.label != "BENIGN"]
                attack_type = Counter(labels).most_common(1)[0][0] if labels else "BENIGN"
                rows.append(vec)
                meta.append((
                    0 if attack_type == "BENIGN" else 1,
                    LABEL_TO_ID[attack_type],
                    dev,
                    self.hospital_id,
                    t0,
                ))

        if not rows:
            return pd.DataFrame(columns=ALL_COLUMNS)

        df = pd.DataFrame(np.vstack(rows), columns=FEATURE_NAMES)
        meta_df = pd.DataFrame(
            meta, columns=["label", "attack_type", "device_id", "hospital_id", "t_start"]
        )
        out = pd.concat([df, meta_df], axis=1)
        return out.sort_values(["device_id", "t_start"]).reset_index(drop=True)

    def reset(self) -> None:
        self._buffer.clear()
