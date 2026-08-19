"""The canonical MIoT flow-feature schema.

Both data paths -- the live MQTT digital twin (Step 2) and the CIC-IoT2023
capture files (Step 3) -- are projected onto *this* 30-dimensional behavioural
feature vector. Sharing one schema is what lets a detector trained on
CIC-IoT2023 be evaluated on live twin traffic and vice-versa.

Feature groups (mirroring Section 3, Phase 1 of the report):
    volume   : packet / byte rates                     -> DDoS, flooding
    size     : packet-size statistics                  -> fragmentation, firmware push
    timing   : flow duration, inter-arrival, burstiness -> beaconing, recon
    entropy  : payload randomness                      -> encrypted/obfuscated payloads
    protocol : MQTT / TCP / UDP + TCP flag ratios      -> spoofing, scanning
    topology : distinct sources / destinations / topics -> reconnaissance, lateral movement
    semantic : physiological plausibility of the vitals -> MITM value tampering
"""
from __future__ import annotations

# ── feature names, in fixed order (never reorder: models are trained on it) ──
FEATURE_NAMES: list[str] = [
    # volume
    "packet_rate", "byte_rate",
    # size
    "pkt_size_mean", "pkt_size_std", "pkt_size_min", "pkt_size_max",
    "payload_len_mean", "header_len_mean",
    # timing
    "flow_duration", "iat_mean", "iat_std", "burstiness",
    # entropy
    "payload_entropy", "topic_entropy",
    # protocol
    "proto_mqtt_ratio", "proto_tcp_ratio", "proto_udp_ratio",
    "flag_syn_ratio", "flag_ack_ratio", "flag_rst_ratio",
    "qos_mean", "retain_ratio",
    # topology
    "n_unique_src", "n_unique_dst", "n_unique_topic",
    "conn_attempt_rate", "subscribe_ratio", "publish_ratio", "dup_ratio",
    # semantic (MIoT-specific: is the *clinical* value plausible?)
    "vital_zscore_max",
]

N_FEATURES = len(FEATURE_NAMES)

# Features an adversary may NOT touch without breaking the attack's function.
# The GAN is only allowed to perturb the complement of this set (Step 6): an
# attacker can pad packets or slow a flood down, but cannot fake the fact that
# it opened many connections, nor un-tamper a vital it deliberately altered.
IMMUTABLE_FEATURES: list[str] = [
    "conn_attempt_rate",
    "n_unique_dst",
    "vital_zscore_max",
    "flag_rst_ratio",
]

# ── label vocabulary ─────────────────────────────────────────────────────────
BENIGN = "BENIGN"

ATTACK_CLASSES: list[str] = [
    "DDoS",              # flooding a device / broker off the network
    "MITM",              # in-transit tampering of clinical values
    "FirmwareTamper",    # forged OTA firmware push to a pump / monitor
    "Spoofing",          # impersonating a legitimate device identity
    "Recon",             # topic enumeration / port + service scanning
    "Mirai",             # botnet enrolment and C2 beaconing
]

LABEL_CLASSES: list[str] = [BENIGN] + ATTACK_CLASSES
LABEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(LABEL_CLASSES)}
ID_TO_LABEL: dict[int, str] = {i: name for name, i in LABEL_TO_ID.items()}

# Column names in every processed CSV, after the feature columns.
META_COLUMNS: list[str] = [
    "label",        # 0 = benign, 1 = attack (binary target)
    "attack_type",  # multi-class id into LABEL_CLASSES
    "device_id",    # which simulated device the flow belongs to
    "hospital_id",  # which federated client owns the row
    "t_start",      # flow window start, seconds since run start
]

ALL_COLUMNS: list[str] = FEATURE_NAMES + META_COLUMNS


def mutable_mask() -> list[bool]:
    """True where the adversarial generator is permitted to perturb."""
    return [name not in IMMUTABLE_FEATURES for name in FEATURE_NAMES]


def feature_index(name: str) -> int:
    return FEATURE_NAMES.index(name)
