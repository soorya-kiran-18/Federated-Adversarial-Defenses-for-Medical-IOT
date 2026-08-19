"""The wire-level event record produced by the digital twin.

A `Packet` is the single unit of observable network activity. Devices emit
them, attacks inject or mutate them, and `swarmdef.data.features` aggregates
windows of them into the 30-dimensional flow vectors the detector consumes.

Keeping this record separate from MQTT itself means the twin can run in two
modes with *identical* semantics:
    online  -- packets are actually published through a Mosquitto broker
    offline -- packets are generated in-process (fast, deterministic, CI-safe)
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# MQTT control-packet types we model.
PUBLISH = "PUBLISH"
SUBSCRIBE = "SUBSCRIBE"
CONNECT = "CONNECT"
PINGREQ = "PINGREQ"
DISCONNECT = "DISCONNECT"


@dataclass
class Packet:
    """One observed network event on the hospital MIoT segment."""

    t: float                      # seconds since capture start
    src: str                      # source device / host identity
    dst: str                      # destination (broker, gateway, external C2)
    topic: str                    # MQTT topic ("" for non-MQTT traffic)
    msg_type: str = PUBLISH       # MQTT control packet type
    protocol: str = "MQTT"        # MQTT | TCP | UDP
    payload_len: int = 90         # application payload bytes
    header_len: int = 40          # transport + MQTT fixed/variable header bytes
    qos: int = 0
    retain: bool = False
    dup: bool = False
    flags: str = "ACK"            # TCP flag summary: SYN | ACK | RST | PSH
    payload: dict[str, Any] | None = None   # decoded telemetry, if any
    vital_z: float = 0.0          # clinical implausibility of the payload
    label: str = "BENIGN"         # ground truth for this packet
    entropy_override: float | None = None   # for encrypted / random payloads

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def size(self) -> int:
        return self.payload_len + self.header_len

    def payload_entropy(self) -> float:
        """Shannon entropy (bits/byte) of the serialised payload.

        Structured JSON telemetry sits around 4-5 bits/byte; encrypted or
        randomised exfiltration/firmware payloads approach 8.
        """
        if self.entropy_override is not None:
            return self.entropy_override
        if not self.payload:
            return 0.0
        raw = json.dumps(self.payload, separators=(",", ":")).encode()
        if not raw:
            return 0.0
        counts = Counter(raw)
        n = len(raw)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    def as_mqtt(self) -> tuple[str, str]:
        """Serialise to a (topic, json-payload) pair for the real broker."""
        body = dict(self.payload or {})
        body.setdefault("_meta", {})
        body["_meta"].update({"t": round(self.t, 4), "src": self.src, "seq_type": self.msg_type})
        return self.topic, json.dumps(body, separators=(",", ":"))


@dataclass
class FlowWindow:
    """A time-bounded batch of packets belonging to one device's flow."""

    device_id: str
    hospital_id: int
    t_start: float
    t_end: float
    packets: list[Packet] = field(default_factory=list)

    def add(self, pkt: Packet) -> None:
        self.packets.append(pkt)

    @property
    def duration(self) -> float:
        return max(self.t_end - self.t_start, 1e-3)

    def dominant_label(self) -> str:
        """A window is labelled malicious if *any* attack packet appears in it.

        This is the standard convention for window-level IDS labelling: a single
        injected packet compromises the window, so the detector must be able to
        spot a minority of hostile packets inside otherwise benign traffic.
        """
        malicious = [p.label for p in self.packets if p.label != "BENIGN"]
        if not malicious:
            return "BENIGN"
        return Counter(malicious).most_common(1)[0][0]

    def __len__(self) -> int:
        return len(self.packets)
