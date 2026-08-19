"""Benign clinical and operational events that *look* like attacks.

Why this module exists
----------------------
A first version of the twin produced a dataset a plain logistic regression
solved at 99.9% accuracy. That is a defective benchmark: if the detector is
already perfect there is no headroom for the GAN to degrade, no visible cost to
differential-privacy noise, and no measurable damage from a poisoned client --
every downstream plot would be a flat line at 1.0.

Real hospital networks are hard precisely because benign traffic regularly
mimics attack signatures:

    a code-blue resuscitation      looks like MITM value tampering
    an authorised firmware rollout looks like a forged OTA push
    a PACS imaging transfer        looks like high-entropy exfiltration
    a ward-wide device reboot      looks like a SYN flood
    shift handover                 looks like a volumetric spike
    network congestion             looks like an interception proxy (DUP flags)

These events are labelled BENIGN. They create genuine class overlap, so the
detector must learn *context* rather than a single threshold -- which is the
entire argument for using a graph model over per-packet rules.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any

from swarmdef.twin.devices import Device
from swarmdef.twin.packet import CONNECT, PUBLISH, Packet


@dataclass
class BenignEvent:
    """A timed, legitimate event. Structurally like an attack, labelled BENIGN."""

    name: str = "EVENT"
    start_t: float = 0.0
    duration: float = 8.0
    intensity: float = 1.0
    target: Device | None = None
    rng: random.Random = field(default_factory=random.Random)
    hospital_id: int = 0
    topic_root: str = "hospital"

    def active(self, t: float) -> bool:
        return self.start_t <= t < self.start_t + self.duration

    def mutate(self, pkt: Packet, t: float) -> Packet:
        return pkt

    def inject(self, t: float, dt: float) -> list[Packet]:
        return []

    def _n_events(self, rate_hz: float, dt: float) -> int:
        expected = rate_hz * dt
        base = int(expected)
        return base + (1 if self.rng.random() < (expected - base) else 0)


class CodeBlue(BenignEvent):
    """Cardiac arrest / rapid-response: vitals swing hard, and legitimately.

    This is the direct confounder for MITM detection. A tampered heart rate and
    a real crashing patient both produce a large `vital_zscore_max`; only the
    surrounding context (no DUP artefacts, the whole monitor escalates together,
    the ward's other devices react) separates them.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "CodeBlue"})
        # Populated by HospitalTwin so the event can generate the ward-wide
        # response described in `inject`.
        self.fleet: list[Device] = []
        self.workstations: list[str] = []

    def mutate(self, pkt: Packet, t: float) -> Packet:
        if not self.active(t) or pkt.payload is None:
            return pkt
        if self.target is not None and pkt.src != self.target.device_id:
            return pkt
        readings = pkt.payload.get("readings")
        if not readings or self.target is None:
            return pkt

        progress = (t - self.start_t) / max(self.duration, 1e-6)
        swing = self.intensity * (1.0 + 2.5 * progress)
        new = dict(readings)
        z = 0.0
        for key in ("heart_rate", "spo2", "resp_rate", "systolic_bp"):
            if key not in new:
                continue
            v = self.target.vitals[key]
            direction = -1.0 if key in ("spo2", "systolic_bp") else 1.0
            val = float(new[key]) + direction * v.sigma * swing * self.rng.uniform(4, 9)
            val = min(v.hi, max(v.lo, val))
            new[key] = round(val, 2)
            z = max(z, v.zscore(val))
        pkt.payload = {**pkt.payload, "readings": new, "alarm": "RAPID_RESPONSE"}
        pkt.vital_z = max(pkt.vital_z, z)
        return pkt

    def inject(self, t: float, dt: float) -> list[Packet]:
        """The whole ward reacts -- and that is the point.

        A real resuscitation is a *correlated, ward-wide* event: the monitor
        alarms, the rapid-response team is paged over the gateway, staff chart
        against the EHR, and the other devices at the bedside are handled and
        report more often. An MITM that fakes the same implausible vital
        produces none of this: one device looks wrong while its neighbours stay
        completely calm.

        That contrast is deliberately the only reliable way to separate the two,
        because it is the discrimination a per-window model *cannot* make and a
        graph model can. It is the empirical justification for the GNN in
        Section 4.1.2 of the report.
        """
        if not self.active(t) or self.target is None:
            return []
        out: list[Packet] = []

        # 1. The alarming device escalates its own reporting.
        for _ in range(self._n_events(8.0 * self.intensity, dt)):
            out.append(Packet(
                t=t + self.rng.random() * dt,
                src=self.target.device_id, dst="broker",
                topic=f"{self.topic_root}/{self.hospital_id}/alarm/{self.target.device_id}",
                msg_type=PUBLISH, protocol="MQTT",
                payload_len=self.rng.randint(80, 180), header_len=42, qos=1,
                flags="ACK", payload={"alarm": "RAPID_RESPONSE", "priority": "HIGH"},
                label="BENIGN",
            ))

        # 2. Rapid-response paging and EHR charting, seen at the gateway.
        ws = self.workstations or [f"10.{self.hospital_id}.1.10"]
        for _ in range(self._n_events(14.0 * self.intensity, dt)):
            out.append(Packet(
                t=t + self.rng.random() * dt,
                src=self.rng.choice(ws), dst=f"10.{self.hospital_id}.0.20",
                topic="", msg_type=PUBLISH, protocol="TCP",
                payload_len=self.rng.randint(300, 1200), header_len=40,
                flags="ACK", entropy_override=self.rng.uniform(4.5, 6.2),
                label="BENIGN",
            ))

        # 3. Bedside neighbours are handled and report more often.
        peers = [d for d in self.fleet if d.device_id != self.target.device_id]
        for _ in range(self._n_events(10.0 * self.intensity, dt)):
            if not peers:
                break
            dev = self.rng.choice(peers)
            out.append(Packet(
                t=t + self.rng.random() * dt,
                src=dev.device_id, dst="broker",
                topic=dev.topic(self.topic_root), msg_type=PUBLISH, protocol="MQTT",
                payload_len=dev.base_payload_bytes + self.rng.randint(-6, 20),
                header_len=42, qos=1, flags="ACK",
                payload=dev.read(), label="BENIGN",
            ))
        return out


class AuthorisedFirmwareUpdate(BenignEvent):
    """A scheduled, signed firmware rollout from the biomed management server.

    Byte-for-byte this looks like `FirmwareTamper`: large, high-entropy chunks
    on the OTA topic with retain set. The difference is provenance -- a known
    management host and an ordinary QoS/retain pattern.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "AuthorisedFirmwareUpdate"})

    def inject(self, t: float, dt: float) -> list[Packet]:
        if not self.active(t) or self.target is None:
            return []
        if self.rng.random() > 0.4 * max(self.intensity, 0.1):
            return []
        return [Packet(
            t=t + self.rng.random() * dt,
            src=f"10.{self.hospital_id}.0.30",          # biomed management server
            dst=self.target.device_id,
            topic=f"{self.topic_root}/{self.hospital_id}/{self.target.device_type}/"
                  f"{self.target.device_id}/ota/firmware",
            msg_type=PUBLISH, protocol="MQTT",
            payload_len=self.rng.randint(8_000, 30_000), header_len=48,
            qos=1, retain=True, flags="PSH",
            entropy_override=self.rng.uniform(7.5, 7.98),
            payload={"op": "ota_write", "ver": "v2.5.0",
                     "sig": "".join(self.rng.choices(string.hexdigits, k=12))},
            label="BENIGN",
        )]


class ImagingTransfer(BenignEvent):
    """A radiology study moving to PACS: sustained high byte rate, big packets."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "ImagingTransfer"})

    def inject(self, t: float, dt: float) -> list[Packet]:
        if not self.active(t):
            return []
        out = []
        src = f"10.{self.hospital_id}.1.{self.rng.randint(10, 15)}"
        for _ in range(self._n_events(60.0 * self.intensity, dt)):
            out.append(Packet(
                t=t + self.rng.random() * dt,
                src=src, dst=f"10.{self.hospital_id}.0.21",
                topic="", msg_type=PUBLISH, protocol="TCP",
                payload_len=self.rng.randint(1200, 1460), header_len=40,
                flags="ACK", entropy_override=self.rng.uniform(7.2, 7.9),
                label="BENIGN",
            ))
        return out


class DeviceReboot(BenignEvent):
    """A device power-cycles: CONNECT storm then normal service resumes.

    Confounds Mirai's credential brute force and a SYN flood's connection churn.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "DeviceReboot"})

    def inject(self, t: float, dt: float) -> list[Packet]:
        if not self.active(t) or self.target is None:
            return []
        out = []
        for _ in range(self._n_events(25.0 * self.intensity, dt)):
            out.append(Packet(
                t=t + self.rng.random() * dt,
                src=self.target.device_id, dst="broker", topic="",
                msg_type=CONNECT, protocol="TCP",
                payload_len=self.rng.randint(20, 60), header_len=40,
                flags=self.rng.choices(["SYN", "RST", "ACK"], [0.6, 0.25, 0.15])[0],
                label="BENIGN",
            ))
        return out


class ShiftHandover(BenignEvent):
    """Ward-wide sync at shift change: every device reconnects and backfills."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "ShiftHandover"})
        self.fleet: list[Device] = []

    def inject(self, t: float, dt: float) -> list[Packet]:
        if not self.active(t) or not self.fleet:
            return []
        out = []
        for _ in range(self._n_events(40.0 * self.intensity, dt)):
            dev = self.rng.choice(self.fleet)
            out.append(Packet(
                t=t + self.rng.random() * dt,
                src=dev.device_id, dst="broker",
                topic=f"{self.topic_root}/{self.hospital_id}/{dev.device_type}/"
                      f"{dev.device_id}/backfill",
                msg_type=self.rng.choices([PUBLISH, CONNECT], [0.85, 0.15])[0],
                protocol="MQTT", payload_len=self.rng.randint(150, 700),
                header_len=42, qos=1, flags="ACK",
                payload={"op": "backfill", "n": self.rng.randint(5, 60)},
                label="BENIGN",
            ))
        return out


class NetworkCongestion(BenignEvent):
    """Switch congestion: retransmits, DUP flags and jitter across the fleet.

    Confounds the MITM interception artefact, which also shows elevated DUP.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "NetworkCongestion"})

    def mutate(self, pkt: Packet, t: float) -> Packet:
        if not self.active(t):
            return pkt
        if self.rng.random() < 0.45 * self.intensity:
            pkt.dup = True
        if self.rng.random() < 0.2 * self.intensity:
            pkt.flags = "RST"
        return pkt


EVENTS: dict[str, type[BenignEvent]] = {
    "CodeBlue": CodeBlue,
    "AuthorisedFirmwareUpdate": AuthorisedFirmwareUpdate,
    "ImagingTransfer": ImagingTransfer,
    "DeviceReboot": DeviceReboot,
    "ShiftHandover": ShiftHandover,
    "NetworkCongestion": NetworkCongestion,
}

# Which device kinds each benign event can involve.
EVENT_TARGETS: dict[str, tuple[str, ...]] = {
    "CodeBlue": ("patient_monitor", "ventilator"),
    "AuthorisedFirmwareUpdate": ("infusion_pump", "patient_monitor", "ventilator"),
    "ImagingTransfer": ("gateway",),
    "DeviceReboot": ("infusion_pump", "patient_monitor", "wearable", "env_sensor"),
    "ShiftHandover": ("gateway",),
    "NetworkCongestion": ("gateway",),
}


def build_event(kind: str, **kw: Any) -> BenignEvent:
    if kind not in EVENTS:
        raise KeyError(f"unknown benign event {kind!r}; choose from {sorted(EVENTS)}")
    return EVENTS[kind](**kw)
