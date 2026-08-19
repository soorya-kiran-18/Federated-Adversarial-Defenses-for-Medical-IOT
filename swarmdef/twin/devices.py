"""Virtual medical devices for the hospital digital twin (Step 2).

Each device is a small stochastic physiological model that emits MQTT-shaped
telemetry. The point is *realism of dynamics*, not medical accuracy: values
must drift smoothly, stay inside clinical bounds, and occasionally show benign
excursions -- otherwise "detect the anomaly" would be trivial thresholding.

Every device publishes to:
    hospital/<hospital_id>/<device_type>/<device_id>/telemetry
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Vital:
    """One clinical signal following a bounded Ornstein-Uhlenbeck random walk."""

    name: str
    baseline: float
    sigma: float              # per-step noise magnitude
    lo: float                 # clinical floor
    hi: float                 # clinical ceiling
    reversion: float = 0.05   # pull back toward baseline
    unit: str = ""
    decimals: int = 1
    value: float = field(init=False)

    def __post_init__(self) -> None:
        self.value = self.baseline

    def step(self, rng: random.Random) -> float:
        """Advance one tick: mean-reverting drift + gaussian noise, then clamp."""
        drift = self.reversion * (self.baseline - self.value)
        self.value += drift + rng.gauss(0.0, self.sigma)
        self.value = min(self.hi, max(self.lo, self.value))
        return round(self.value, self.decimals)

    def zscore(self, value: float) -> float:
        """How implausible a reported value is, in units of the vital's noise.

        The twin's MITM detector feature (`vital_zscore_max`) is built from this:
        a tampered heart rate of 210 bpm is many sigma from where the true
        physiological process could have walked in one tick.
        """
        spread = max(self.sigma * 6.0, 1e-6)
        return abs(value - self.value) / spread


class Device:
    """Base class for every simulated MIoT endpoint."""

    device_type = "generic"
    # Nominal MQTT publish characteristics, used to build flow features.
    base_payload_bytes = 90
    base_qos = 0

    def __init__(self, device_id: str, hospital_id: int, rng: random.Random) -> None:
        self.device_id = device_id
        self.hospital_id = hospital_id
        self.rng = rng
        self.vitals: dict[str, Vital] = {}
        self.seq = 0
        self.firmware = "v2.4.1"
        self._build()

    def _build(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # ── telemetry ────────────────────────────────────────────────────────────
    def topic(self, root: str = "hospital") -> str:
        return f"{root}/{self.hospital_id}/{self.device_type}/{self.device_id}/telemetry"

    def read(self) -> dict[str, Any]:
        """Produce one telemetry payload."""
        self.seq += 1
        readings = {v.name: v.step(self.rng) for v in self.vitals.values()}
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "hospital_id": self.hospital_id,
            "seq": self.seq,
            "firmware": self.firmware,
            "readings": readings,
            "status": "OK",
        }

    def plausibility(self, readings: dict[str, float]) -> float:
        """Max z-score across reported vitals -- the MITM tell-tale."""
        scores = [
            self.vitals[name].zscore(float(val))
            for name, val in readings.items()
            if name in self.vitals and isinstance(val, (int, float))
        ]
        return max(scores) if scores else 0.0

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.device_id}@H{self.hospital_id}>"


class PatientMonitor(Device):
    """Bedside multi-parameter monitor: ECG, SpO2, respiration, blood pressure."""

    device_type = "patient_monitor"
    base_payload_bytes = 140

    def _build(self) -> None:
        self.vitals = {
            "heart_rate":  Vital("heart_rate", self.rng.uniform(62, 88), 1.6, 35, 180, unit="bpm"),
            "spo2":        Vital("spo2", self.rng.uniform(95, 99), 0.35, 80, 100, unit="%"),
            "resp_rate":   Vital("resp_rate", self.rng.uniform(12, 18), 0.7, 5, 45, unit="brpm"),
            "systolic_bp": Vital("systolic_bp", self.rng.uniform(105, 130), 2.0, 70, 200, unit="mmHg"),
            "temp_c":      Vital("temp_c", self.rng.uniform(36.4, 37.2), 0.05, 34, 42, unit="C", decimals=2),
        }


class InfusionPump(Device):
    """Smart infusion pump: the canonical high-consequence MIoT target.

    A tampered infusion rate is directly patient-harming, which is why this
    device's telemetry is weighted most heavily in the twin's attack scenarios.
    """

    device_type = "infusion_pump"
    base_payload_bytes = 110
    base_qos = 1  # dosing commands are delivered at-least-once

    def _build(self) -> None:
        self.vitals = {
            "flow_rate_ml_h":  Vital("flow_rate_ml_h", self.rng.uniform(20, 120), 0.8, 0, 1200, unit="mL/h"),
            "reservoir_pct":   Vital("reservoir_pct", self.rng.uniform(40, 95), 0.25, 0, 100, unit="%", reversion=0.0),
            "line_pressure":   Vital("line_pressure", self.rng.uniform(8, 14), 0.4, 0, 60, unit="psi"),
            "battery_pct":     Vital("battery_pct", self.rng.uniform(55, 100), 0.15, 0, 100, unit="%", reversion=0.0),
        }


class Ventilator(Device):
    """Mechanical ventilator: tidal volume, PEEP, FiO2, airway pressure."""

    device_type = "ventilator"
    base_payload_bytes = 160
    base_qos = 1

    def _build(self) -> None:
        self.vitals = {
            "tidal_volume_ml": Vital("tidal_volume_ml", self.rng.uniform(400, 520), 6.0, 100, 900, unit="mL", decimals=0),
            "peep_cmh2o":      Vital("peep_cmh2o", self.rng.uniform(4, 8), 0.2, 0, 25, unit="cmH2O"),
            "fio2_pct":        Vital("fio2_pct", self.rng.uniform(28, 45), 0.6, 21, 100, unit="%"),
            "airway_pressure": Vital("airway_pressure", self.rng.uniform(14, 22), 0.7, 0, 60, unit="cmH2O"),
        }


class WearableSensor(Device):
    """Ambulatory patient wearable: low power, chatty, weak crypto in practice."""

    device_type = "wearable"
    base_payload_bytes = 70

    def _build(self) -> None:
        self.vitals = {
            "heart_rate":  Vital("heart_rate", self.rng.uniform(65, 95), 2.2, 35, 190, unit="bpm"),
            "steps":       Vital("steps", self.rng.uniform(0, 40), 6.0, 0, 400, unit="count", reversion=0.02, decimals=0),
            "skin_temp_c": Vital("skin_temp_c", self.rng.uniform(32.5, 34.5), 0.08, 28, 40, unit="C", decimals=2),
        }


class EnvironmentSensor(Device):
    """Ward environmental sensor: cold-chain and air-quality compliance."""

    device_type = "env_sensor"
    base_payload_bytes = 80

    def _build(self) -> None:
        self.vitals = {
            "room_temp_c": Vital("room_temp_c", self.rng.uniform(20.5, 23.5), 0.09, 12, 35, unit="C", decimals=2),
            "humidity_pct": Vital("humidity_pct", self.rng.uniform(40, 55), 0.5, 10, 90, unit="%"),
            "co2_ppm":     Vital("co2_ppm", self.rng.uniform(430, 720), 8.0, 350, 3000, unit="ppm", decimals=0),
        }


# Clinical devices only -- the Gateway is added separately by HospitalTwin.
DEVICE_TYPES: list[type[Device]] = [
    PatientMonitor,
    InfusionPump,
    Ventilator,
    WearableSensor,
    EnvironmentSensor,
]


def build_fleet(hospital_id: int, n_devices: int, rng: random.Random) -> list[Device]:
    """Instantiate a plausible device mix for one hospital.

    A patient monitor and an infusion pump are always present (they are the
    devices the attack scenarios target); the rest of the fleet is sampled so
    that hospitals differ from one another -- part of what makes the federated
    data non-IID at the *source*, not just in the label distribution.
    """
    fleet: list[Device] = []
    guaranteed = [PatientMonitor, InfusionPump]
    for i, cls in enumerate(guaranteed[:n_devices]):
        fleet.append(cls(f"H{hospital_id}-{cls.device_type}-{i:02d}", hospital_id, rng))
    remaining = [c for c in DEVICE_TYPES]
    while len(fleet) < n_devices:
        cls = rng.choice(remaining)
        idx = sum(1 for d in fleet if d.device_type == cls.device_type)
        fleet.append(cls(f"H{hospital_id}-{cls.device_type}-{idx:02d}", hospital_id, rng))
    return fleet


class Gateway(Device):
    """The ward's network gateway / IDS span port.

    Not a clinical device: it is the vantage point from which segment-wide
    traffic is observed. Attacks that address no specific device -- subnet
    reconnaissance, identity spoofing against the broker -- are only visible
    here, so the gateway must carry its own realistic benign baseline
    (EHR synchronisation, DNS/NTP, nurse workstations, broker keepalives).
    Without that baseline the gateway node would be trivially separable: every
    packet on it would be hostile.
    """

    device_type = "gateway"
    base_payload_bytes = 300

    def _build(self) -> None:
        self.vitals = {}
        h = self.hospital_id
        # A stable set of legitimate hosts on this ward's subnet.
        self.workstations = [f"10.{h}.1.{i}" for i in range(10, 10 + self.rng.randint(3, 6))]
        self.services = {
            "ehr":  f"10.{h}.0.20",
            "pacs": f"10.{h}.0.21",
            "dns":  f"10.{h}.0.2",
            "ntp":  f"10.{h}.0.3",
        }

    def background(self, t: float, dt: float, rate_hz: float = 12.0) -> list:
        """Emit benign, non-device segment traffic for the interval [t, t+dt)."""
        from swarmdef.twin.packet import CONNECT, PINGREQ, PUBLISH, Packet

        expected = rate_hz * dt
        n = int(expected) + (1 if self.rng.random() < expected - int(expected) else 0)
        out = []
        for _ in range(n):
            kind = self.rng.choices(
                ["ehr", "pacs", "dns", "ntp", "keepalive"],
                [0.38, 0.17, 0.18, 0.12, 0.15],
            )[0]
            src = self.rng.choice(self.workstations)
            if kind == "ehr":
                pkt = Packet(t=t + self.rng.random() * dt, src=src, dst=self.services["ehr"],
                             topic="", msg_type=PUBLISH, protocol="TCP",
                             payload_len=self.rng.randint(200, 1400), header_len=40,
                             flags=self.rng.choices(["ACK", "SYN"], [0.85, 0.15])[0],
                             entropy_override=self.rng.uniform(4.2, 6.0), label="BENIGN")
            elif kind == "pacs":
                # Imaging transfers: large, high-entropy but from a known host.
                pkt = Packet(t=t + self.rng.random() * dt, src=src, dst=self.services["pacs"],
                             topic="", msg_type=PUBLISH, protocol="TCP",
                             payload_len=self.rng.randint(1000, 1460), header_len=40,
                             flags="ACK", entropy_override=self.rng.uniform(7.0, 7.8),
                             label="BENIGN")
            elif kind in ("dns", "ntp"):
                pkt = Packet(t=t + self.rng.random() * dt, src=src, dst=self.services[kind],
                             topic="", msg_type=PUBLISH, protocol="UDP",
                             payload_len=self.rng.randint(40, 120), header_len=28,
                             flags="PSH", entropy_override=self.rng.uniform(3.5, 5.5),
                             label="BENIGN")
            else:
                pkt = Packet(t=t + self.rng.random() * dt, src=src, dst="broker",
                             topic=f"hospital/{self.hospital_id}/$SYS/keepalive",
                             msg_type=self.rng.choices([PINGREQ, CONNECT], [0.8, 0.2])[0],
                             protocol="MQTT", payload_len=self.rng.randint(2, 30),
                             header_len=38, flags="ACK", label="BENIGN")
            out.append(pkt)
        return out
