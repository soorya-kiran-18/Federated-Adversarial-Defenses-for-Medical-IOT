"""Attack scenarios injected into the hospital digital twin (Step 2).

Each scenario is a finite-duration state machine with two hooks:

    mutate(pkt)   -- tamper with a packet the fleet was going to send anyway
    inject(t, dt) -- emit *additional* hostile packets for this tick

This split matters. A MITM only rewrites existing telemetry (no new flows, so
volume features look normal -- only the clinical plausibility feature betrays
it), whereas a DDoS adds thousands of new packets (volume explodes, but every
payload is individually well-formed). A detector that only looks at one feature
family will miss one of the two, which is exactly the point of the GNN.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any

from swarmdef.twin.devices import Device
from swarmdef.twin.packet import CONNECT, PINGREQ, PUBLISH, SUBSCRIBE, Packet


@dataclass
class AttackScenario:
    """Base class: a timed attack against one hospital's device fleet."""

    name: str = "BASE"
    start_t: float = 0.0
    duration: float = 10.0
    intensity: float = 1.0          # 0..1 scaling of how aggressive the attack is
    target: Device | None = None
    rng: random.Random = field(default_factory=random.Random)
    hospital_id: int = 0
    topic_root: str = "hospital"

    def active(self, t: float) -> bool:
        return self.start_t <= t < self.start_t + self.duration

    def mutate(self, pkt: Packet, t: float) -> Packet:
        """Tamper with an outbound legitimate packet. Default: leave it alone."""
        return pkt

    def inject(self, t: float, dt: float) -> list[Packet]:
        """Emit extra hostile packets for the interval [t, t+dt)."""
        return []

    # ── helpers ──────────────────────────────────────────────────────────────
    def _rand_ip(self) -> str:
        return ".".join(str(self.rng.randint(1, 254)) for _ in range(4))

    def _n_events(self, rate_hz: float, dt: float) -> int:
        """Poisson-ish count of events in dt at the given rate."""
        expected = rate_hz * dt
        base = int(expected)
        return base + (1 if self.rng.random() < (expected - base) else 0)


class DDoSFlood(AttackScenario):
    """Volumetric flood aimed at the MQTT broker / a specific device.

    Signature: packet_rate and byte_rate spike by orders of magnitude, inter-
    arrival times collapse, SYN ratio climbs as half-open connections pile up,
    and source addresses fan out across a spoofed range.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "DDoS"})
        self.peak_rate = 400.0 * max(self.intensity, 0.05)   # packets/second
        # A botnet controls a finite number of hosts. Drawing a fresh random
        # source for every packet would make `n_unique_src` a perfect give-away
        # (unique sources == packet count, which nothing benign ever does), so
        # the flood is emitted from a bounded pool that scales with intensity:
        # a small internal reflector at low intensity, a wide botnet at high.
        pool_size = max(2, int(3 + 180 * self.intensity ** 2))
        self.botnet = [self._rand_ip() for _ in range(pool_size)]

    def inject(self, t: float, dt: float) -> list[Packet]:
        if not self.active(t):
            return []
        # Ramp up over the first 20% of the attack, then hold: real floods are
        # not instantaneous, and the ramp is what a rate-based detector sees.
        progress = (t - self.start_t) / max(self.duration, 1e-6)
        rate = self.peak_rate * min(1.0, progress / 0.2 + 0.15)
        pkts: list[Packet] = []
        dst = self.target.device_id if self.target else "broker"
        for _ in range(self._n_events(rate, dt)):
            pkts.append(Packet(
                t=t + self.rng.random() * dt,
                src=self.rng.choice(self.botnet),  # bounded botnet source pool
                dst=dst,
                topic=f"{self.topic_root}/{self.hospital_id}/{dst}/ingest",
                msg_type=self.rng.choices([CONNECT, PUBLISH, PINGREQ], [0.55, 0.3, 0.15])[0],
                protocol=self.rng.choices(["TCP", "MQTT", "UDP"], [0.5, 0.35, 0.15])[0],
                payload_len=self.rng.randint(0, 60),
                header_len=40,
                flags=self.rng.choices(["SYN", "ACK", "RST"], [0.7, 0.2, 0.1])[0],
                label=self.name,
            ))
        return pkts


class MITMTamper(AttackScenario):
    """Man-in-the-middle rewriting of clinical values in transit.

    Signature: traffic volume is *unchanged* -- this attack is invisible to any
    rate-based IDS. The only tells are the clinical implausibility of the
    reported vital and a small DUP-flag artefact from the interception proxy.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "MITM"})
        self.offset_scale = 1.0 + 3.0 * self.intensity

    def mutate(self, pkt: Packet, t: float) -> Packet:
        if not self.active(t) or pkt.payload is None:
            return pkt
        if self.target is not None and pkt.src != self.target.device_id:
            return pkt
        readings = pkt.payload.get("readings")
        if not readings:
            return pkt

        device = self.target
        key = self.rng.choice(list(readings.keys()))
        original = float(readings[key])
        # Drive the value toward a clinically dangerous direction: suppress an
        # alarm (report a normal HR while the patient crashes) or force an
        # overdose (inflate an infusion rate).
        #
        # A competent attacker keeps the forged value inside the range a real
        # deterioration could produce -- a heart rate of 900 would be rejected by
        # the device's own range check. So the magnitude here deliberately
        # overlaps `CodeBlue`, and the implausibility feature alone can no longer
        # separate attack from emergency. What separates them is that a real
        # emergency is corroborated by the rest of the ward.
        # Matched to CodeBlue: same direction convention, same sigma-scaled
        # magnitude range, so `vital_zscore_max` alone cannot separate them.
        v = device.vitals[key] if device is not None and key in device.vitals else None
        if v is not None:
            direction = -1.0 if key in ("spo2", "systolic_bp", "reservoir_pct") else 1.0
            swing = self.intensity * (1.0 + 2.5 * self.rng.random())
            tampered = original + direction * v.sigma * swing * self.rng.uniform(4, 9)
            tampered = min(v.hi, max(v.lo, tampered))
        else:
            tampered = original * (1.0 + 0.35 * self.offset_scale)

        z = device.plausibility({key: tampered}) if device is not None else 8.0
        pkt.payload = {**pkt.payload, "readings": {**readings, key: round(tampered, 2)},
                       "alarm": "RAPID_RESPONSE"}
        pkt.payload["_tampered_field"] = key
        pkt.vital_z = max(pkt.vital_z, z)
        # A transparent proxy leaks only an occasional re-send; congestion
        # produces far more, so DUP on its own is not evidence either.
        pkt.dup = self.rng.random() < 0.12      # interception proxy re-send artefact
        pkt.label = self.name
        return pkt

    def inject(self, t: float, dt: float) -> list[Packet]:
        """The victim device alarms on the forged value -- and nothing else does.

        A monitor cannot tell that its reading was rewritten in transit: it sees
        a crashing patient and escalates exactly as it would in a genuine
        emergency. So the *device-local* traffic signature of an MITM is, by
        construction, the same as `CodeBlue`'s.

        What an attacker cannot forge is the ward's reaction. A real
        resuscitation pages the rapid-response team, drives EHR charting at the
        gateway and brings staff to the neighbouring devices. A tampered vital
        produces an alarming device surrounded by a completely calm ward.

        This is the discrimination the GNN exists to make, and no per-window
        model can make it -- the evidence simply is not in the victim's own
        feature vector.
        """
        if not self.active(t) or self.target is None:
            return []
        out: list[Packet] = []
        for _ in range(self._n_events(8.0 * self.intensity, dt)):
            out.append(Packet(
                t=t + self.rng.random() * dt,
                src=self.target.device_id, dst="broker",
                topic=f"{self.topic_root}/{self.hospital_id}/alarm/{self.target.device_id}",
                msg_type=PUBLISH, protocol="MQTT",
                payload_len=self.rng.randint(80, 180), header_len=42, qos=1,
                flags="ACK", payload={"alarm": "RAPID_RESPONSE", "priority": "HIGH"},
                label=self.name,
            ))
        return out


class FirmwareTamper(AttackScenario):
    """Forged over-the-air firmware push to a pump / monitor.

    Signature: a handful of very large, high-entropy PUBLISH packets on an OTA
    topic the device never normally sees, with retain set so the payload sticks
    for every future subscriber.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "FirmwareTamper"})

    def inject(self, t: float, dt: float) -> list[Packet]:
        if not self.active(t):
            return []
        # Firmware images arrive as a slow burst of large chunks.
        if self.rng.random() > 0.35 * max(self.intensity, 0.1):
            return []
        dev = self.target.device_id if self.target else "unknown"
        dtype = self.target.device_type if self.target else "device"
        return [Packet(
            t=t + self.rng.random() * dt,
            src=self._rand_ip(),
            dst=dev,
            topic=f"{self.topic_root}/{self.hospital_id}/{dtype}/{dev}/ota/firmware",
            msg_type=PUBLISH,
            protocol="MQTT",
            payload_len=self.rng.randint(8_000, 32_000),   # firmware chunk
            header_len=48,
            qos=1,
            retain=True,
            flags="PSH",
            entropy_override=self.rng.uniform(7.6, 7.99),  # compressed/encrypted blob
            payload={"op": "ota_write", "ver": "v9.9.9-rogue",
                     "chunk": "".join(self.rng.choices(string.printable, k=16))},
            label=self.name,
        )]


class Spoofing(AttackScenario):
    """Identity spoofing: a rogue host publishes as a legitimate device.

    Signature: duplicate device identity appearing from a new source address,
    a fresh CONNECT storm as the impostor fights the real client for the
    session, and telemetry that does not continue the true physiological walk.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "Spoofing"})
        # The impostor is typically an already-compromised host on the same
        # subnet, not a fresh external address per packet.
        self.impostors = [f"10.{self.hospital_id}.1.{self.rng.randint(10, 60)}"
                          for _ in range(max(1, int(1 + 3 * self.intensity)))]

    def inject(self, t: float, dt: float) -> list[Packet]:
        if not self.active(t):
            return []
        pkts: list[Packet] = []
        dev = self.target
        rate = 6.0 * max(self.intensity, 0.1)
        for _ in range(self._n_events(rate, dt)):
            fake_readings: dict[str, float] = {}
            z = 0.0
            if dev is not None:
                key = self.rng.choice(list(dev.vitals.keys()))
                v = dev.vitals[key]
                # The impostor guesses a plausible-looking constant, but it does
                # not track the real signal -- so it lands far from the true walk.
                guess = v.baseline + self.rng.choice([-1, 1]) * v.sigma * self.rng.uniform(8, 20)
                fake_readings = {key: round(guess, 2)}
                z = v.zscore(guess)
            pkts.append(Packet(
                t=t + self.rng.random() * dt,
                src=self.rng.choice(self.impostors),       # NOT the device's real address
                dst="broker",
                topic=(dev.topic(self.topic_root) if dev else f"{self.topic_root}/{self.hospital_id}/spoof"),
                msg_type=self.rng.choices([PUBLISH, CONNECT], [0.7, 0.3])[0],
                protocol="MQTT",
                payload_len=self.rng.randint(60, 150),
                header_len=42,
                qos=self.rng.choice([0, 1]),
                dup=self.rng.random() < 0.2,
                flags=self.rng.choices(["ACK", "SYN"], [0.6, 0.4])[0],
                payload={"device_id": dev.device_id if dev else "?", "readings": fake_readings},
                vital_z=z,
                label=self.name,
            ))
        return pkts


class Recon(AttackScenario):
    """Reconnaissance: topic enumeration and service scanning.

    Signature: low volume but very high *topic* entropy and destination fan-out
    -- the scanner touches many endpoints once each, the opposite shape of a
    flood. RST ratio climbs as closed ports refuse connections.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "Recon"})

    def inject(self, t: float, dt: float) -> list[Packet]:
        if not self.active(t):
            return []
        pkts: list[Packet] = []
        rate = 12.0 * max(self.intensity, 0.1)
        if not hasattr(self, "_scan_src"):
            # One compromised host sweeping a bounded address range -- a real
            # scan does not forge a new source for every probe.
            self._scan_src = self._rand_ip()
            self._scan_range = [self._rand_ip() for _ in range(max(3, int(8 + 60 * self.intensity)))]
        src = self._scan_src
        for _ in range(self._n_events(rate, dt)):
            probe = "".join(self.rng.choices(string.ascii_lowercase, k=self.rng.randint(3, 9)))
            pkts.append(Packet(
                t=t + self.rng.random() * dt,
                src=src,
                dst=self.rng.choice(self._scan_range),  # sweeping the subnet
                topic=f"{self.topic_root}/{self.hospital_id}/{probe}/status",
                msg_type=self.rng.choices([SUBSCRIBE, CONNECT], [0.75, 0.25])[0],
                protocol=self.rng.choices(["TCP", "MQTT"], [0.45, 0.55])[0],
                payload_len=self.rng.randint(0, 24),
                header_len=40,
                flags=self.rng.choices(["SYN", "RST", "ACK"], [0.45, 0.4, 0.15])[0],
                label=self.name,
            ))
        return pkts


class MiraiBotnet(AttackScenario):
    """Mirai-style enrolment: credential brute force then periodic C2 beacons.

    Signature: two phases. A brute-force burst of CONNECT attempts with varying
    credentials, then a very regular low-rate beacon to an external host --
    low burstiness, high periodicity, small encrypted payloads.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**{**kw, "name": "Mirai"})
        self.c2 = self._rand_ip()
        self.bruteforce_frac = 0.35     # first third of the window is the brute force
        self._next_beacon = self.start_t

    def inject(self, t: float, dt: float) -> list[Packet]:
        if not self.active(t):
            return []
        elapsed = t - self.start_t
        pkts: list[Packet] = []

        if elapsed < self.duration * self.bruteforce_frac:
            rate = 45.0 * max(self.intensity, 0.1)
            for _ in range(self._n_events(rate, dt)):
                pkts.append(Packet(
                    t=t + self.rng.random() * dt,
                    src=self.c2,
                    dst=self.target.device_id if self.target else "broker",
                    topic="",
                    msg_type=CONNECT,
                    protocol="TCP",
                    payload_len=self.rng.randint(16, 48),
                    header_len=40,
                    flags=self.rng.choices(["SYN", "RST"], [0.65, 0.35])[0],
                    label=self.name,
                ))
        else:
            # C2 beacon: machine-regular, which is itself the anomaly.
            period = 0.75
            while self._next_beacon < t + dt:
                if self._next_beacon >= t:
                    pkts.append(Packet(
                        t=self._next_beacon,
                        src=self.target.device_id if self.target else "compromised",
                        dst=self.c2,
                        topic="",
                        msg_type=PUBLISH,
                        protocol="UDP",
                        payload_len=self.rng.randint(28, 40),
                        header_len=28,
                        flags="PSH",
                        entropy_override=self.rng.uniform(7.4, 7.95),
                        label=self.name,
                    ))
                self._next_beacon += period
        return pkts


SCENARIOS: dict[str, type[AttackScenario]] = {
    "DDoS": DDoSFlood,
    "MITM": MITMTamper,
    "FirmwareTamper": FirmwareTamper,
    "Spoofing": Spoofing,
    "Recon": Recon,
    "Mirai": MiraiBotnet,
}


def build_attack(kind: str, **kw: Any) -> AttackScenario:
    """Factory: ``build_attack("DDoS", start_t=10, duration=5, target=pump)``."""
    if kind not in SCENARIOS:
        raise KeyError(f"unknown attack {kind!r}; choose from {sorted(SCENARIOS)}")
    return SCENARIOS[kind](**kw)
