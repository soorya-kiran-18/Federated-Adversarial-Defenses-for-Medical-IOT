"""One simulated hospital: a device fleet, an attack scheduler, and a clock.

`HospitalTwin` is deliberately transport-agnostic. It advances simulated time
and yields `Packet` objects; whether those packets are then published to a real
Mosquitto broker (`swarmdef.twin.runner`) or consumed in-process to build a
dataset (`swarmdef.data.synth`) is the caller's choice. The physics, the attack
behaviour and the ground-truth labels are identical either way -- which is what
makes the offline dataset a faithful stand-in for the live demo.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from swarmdef.twin.attacks import AttackScenario, build_attack
from swarmdef.twin.devices import Device, Gateway, build_fleet
from swarmdef.twin.events import (EVENT_TARGETS, EVENTS, BenignEvent, CodeBlue,
                                  ShiftHandover, build_event)
from swarmdef.twin.packet import PUBLISH, Packet


@dataclass
class ScheduledAttack:
    """A declarative attack booking, resolved into a scenario at build time."""

    kind: str
    start_t: float
    duration: float = 8.0
    intensity: float = 0.8
    target_type: str | None = None    # e.g. "infusion_pump"; None => random device
    target_id: str | None = None      # pin an exact device (used by the planner)


class HospitalTwin:
    """A digital twin of one hospital's MIoT segment."""

    def __init__(
        self,
        hospital_id: int,
        n_devices: int = 5,
        publish_hz: float = 5.0,
        topic_root: str = "hospital",
        seed: int = 42,
    ) -> None:
        self.hospital_id = hospital_id
        self.publish_hz = publish_hz
        self.topic_root = topic_root
        # Per-hospital RNG streams keep hospitals independent but reproducible.
        self.rng = random.Random(seed * 1000 + hospital_id)
        self.attack_rng = random.Random(seed * 2000 + hospital_id)
        self.devices: list[Device] = build_fleet(hospital_id, n_devices, self.rng)
        # The gateway is an observation point, not a clinical device: it is
        # excluded from attack targeting but carries the segment's background
        # traffic and absorbs any packet not addressed to a specific device.
        self.gateway = Gateway(f"H{hospital_id}-gateway", hospital_id, self.rng)
        self.background_hz = 12.0
        self.attacks: list[AttackScenario] = []
        # Legitimate-but-attack-shaped activity (code blue, imaging transfer,
        # authorised firmware rollout). Labelled BENIGN; exists to make the
        # detection problem non-trivial.
        self.events: list[BenignEvent] = []
        self.t = 0.0
        self._emit_carry: dict[str, float] = {d.device_id: 0.0 for d in self.devices}

    @property
    def all_nodes(self) -> list[Device]:
        """Clinical devices plus the gateway -- the node set of the device graph."""
        return [*self.devices, self.gateway]

    # ── attack scheduling ────────────────────────────────────────────────────
    def device_of_type(self, device_type: str) -> Device | None:
        matches = [d for d in self.devices if d.device_type == device_type]
        return self.rng.choice(matches) if matches else None

    def schedule(self, booking: ScheduledAttack) -> AttackScenario:
        """Book an attack against this hospital."""
        if booking.target_id is not None:
            target = next(
                (d for d in self.all_nodes if d.device_id == booking.target_id), None
            )
        elif booking.target_type:
            target = self.device_of_type(booking.target_type)
        else:
            # No explicit target: pick one this attack class could plausibly
            # hit, so a live-triggered Recon lands on the gateway rather than
            # on an infusion pump.
            pool = [d for t in ATTACK_TARGETS.get(booking.kind, ()) for d in self.all_nodes
                    if d.device_type == t]
            target = self.rng.choice(pool or self.devices)
        scenario = build_attack(
            booking.kind,
            start_t=booking.start_t,
            duration=booking.duration,
            intensity=booking.intensity,
            target=target,
            rng=random.Random(self.attack_rng.randrange(1 << 30)),
            hospital_id=self.hospital_id,
            topic_root=self.topic_root,
        )
        self.attacks.append(scenario)
        return scenario

    def schedule_many(self, bookings: list[ScheduledAttack]) -> None:
        for b in bookings:
            self.schedule(b)

    def schedule_events(self, duration: float, rng: random.Random,
                        events_per_device: float = 4.0) -> list[BenignEvent]:
        """Book benign confounder events across the fleet's timeline.

        These are spread independently of the attack schedule -- they may and
        should overlap with attacks, because in a real ward a code blue during
        a scan is exactly the ambiguous case the detector has to survive.
        """
        kinds = list(EVENTS)
        total = max(1, int(round(events_per_device * len(self.all_nodes))))
        lo, hi = duration * 0.05, duration * 0.97
        for _ in range(total):
            kind = rng.choice(kinds)
            pool = [d for t in EVENT_TARGETS.get(kind, ()) for d in self.all_nodes
                    if d.device_type == t]
            target = rng.choice(pool) if pool else rng.choice(self.devices)
            dur = rng.uniform(4.0, 14.0)
            ev = build_event(
                kind,
                start_t=rng.uniform(lo, max(lo, hi - dur)),
                duration=dur,
                intensity=rng.uniform(0.35, 1.0),
                target=target,
                rng=random.Random(self.attack_rng.randrange(1 << 30)),
                hospital_id=self.hospital_id,
                topic_root=self.topic_root,
            )
            if isinstance(ev, (ShiftHandover, CodeBlue)):
                ev.fleet = list(self.devices)
            if isinstance(ev, CodeBlue):
                ev.workstations = list(self.gateway.workstations)
            self.events.append(ev)
        return self.events

    def trigger_now(self, kind: str, duration: float = 8.0, intensity: float = 0.8,
                    target_type: str | None = None) -> AttackScenario:
        """Fire an attack starting at the current simulated time (live demo hook)."""
        return self.schedule(ScheduledAttack(kind, self.t, duration, intensity, target_type))

    def active_attacks(self, t: float | None = None) -> list[AttackScenario]:
        t = self.t if t is None else t
        return [a for a in self.attacks if a.active(t)]

    # ── simulation ───────────────────────────────────────────────────────────
    def tick(self, dt: float) -> list[Packet]:
        """Advance the twin by `dt` seconds and return everything observed."""
        t = self.t
        packets: list[Packet] = []

        # 1. Legitimate telemetry from every device, at its own publish rate.
        for dev in self.devices:
            self._emit_carry[dev.device_id] += self.publish_hz * dt
            n_emit = int(self._emit_carry[dev.device_id])
            self._emit_carry[dev.device_id] -= n_emit
            for k in range(n_emit):
                reading = dev.read()
                pkt = Packet(
                    t=t + (k + 0.5) * dt / max(n_emit, 1),
                    src=dev.device_id,
                    dst="broker",
                    topic=dev.topic(self.topic_root),
                    msg_type=PUBLISH,
                    protocol="MQTT",
                    payload_len=dev.base_payload_bytes + self.rng.randint(-8, 8),
                    header_len=40 + self.rng.randint(0, 6),
                    qos=dev.base_qos,
                    flags="ACK",
                    payload=reading,
                    label="BENIGN",
                )
                # 2a. Benign events that alter legitimate traffic (code blue,
                #     congestion). Applied first so an attack can still tamper
                #     on top of them -- the genuinely ambiguous case.
                for ev in self.events:
                    pkt = ev.mutate(pkt, t)
                # 2b. Attacks that rewrite legitimate traffic (MITM).
                for atk in self.attacks:
                    pkt = atk.mutate(pkt, t)
                packets.append(pkt)

        # 3. Benign segment traffic seen at the gateway (EHR, DNS/NTP, admin)
        #    plus any active clinical/operational events.
        packets.extend(self.gateway.background(t, dt, self.background_hz))
        for ev in self.events:
            packets.extend(ev.inject(t, dt))

        # 4. Attacks that inject additional hostile traffic.
        for atk in self.attacks:
            packets.extend(atk.inject(t, dt))

        self.t += dt
        packets.sort(key=lambda p: p.t)
        return packets

    def run(self, duration: float, dt: float = 0.2) -> Iterator[list[Packet]]:
        """Yield one batch of packets per tick for `duration` simulated seconds."""
        n_ticks = int(round(duration / dt))
        for _ in range(n_ticks):
            yield self.tick(dt)

    def __repr__(self) -> str:
        return (f"<HospitalTwin id={self.hospital_id} devices={len(self.devices)} "
                f"attacks={len(self.attacks)} t={self.t:.1f}s>")


# Which device kinds each attack class can plausibly target. Reconnaissance
# addresses no single device, so it is only ever observed at the gateway.
ATTACK_TARGETS: dict[str, tuple[str, ...]] = {
    "DDoS":           ("patient_monitor", "infusion_pump", "ventilator", "gateway"),
    "MITM":           ("infusion_pump", "patient_monitor", "ventilator"),
    "FirmwareTamper": ("infusion_pump", "patient_monitor", "ventilator"),
    "Spoofing":       ("patient_monitor", "wearable", "infusion_pump"),
    "Recon":          ("gateway",),
    "Mirai":          ("wearable", "env_sensor", "patient_monitor"),
}


def plan_attacks(
    twin: "HospitalTwin",
    duration: float,
    rng: random.Random,
    attacks_per_device: int = 6,
    non_iid: bool = True,
) -> list[ScheduledAttack]:
    """Build a per-hospital attack schedule pinned to specific devices.

    Three properties matter for the downstream experiments:

    1. **Coverage.** Every attack class appears at least once at every hospital,
       so the 6-way confusion matrix is well defined everywhere.
    2. **Skew.** *Frequencies* still differ sharply per hospital -- hospital 0 is
       dominated by DDoS, hospital 1 by MITM, and so on. This is the source of
       the non-IID label distribution the federated experiments rely on: no
       single hospital sees a representative sample of the threat landscape.
    3. **Unambiguous labels.** Windows are per-device, so two attacks may run
       concurrently as long as they hit *different* devices. Within any one
       device's timeline the slots stay disjoint, otherwise a window containing
       both a flood and a tampered vital could only carry one of the two labels
       and the multi-class metrics would silently degrade.
    """
    from swarmdef.data.schema import ATTACK_CLASSES

    kinds = list(ATTACK_CLASSES)
    if non_iid:
        dominant = kinds[twin.hospital_id % len(kinds)]
        secondary = kinds[(twin.hospital_id + 1) % len(kinds)]
        weights = [
            0.45 if k == dominant else 0.25 if k == secondary else 0.30 / (len(kinds) - 2)
            for k in kinds
        ]
    else:
        weights = [1.0 / len(kinds)] * len(kinds)

    nodes = twin.all_nodes
    by_type: dict[str, list] = {}
    for node in nodes:
        by_type.setdefault(node.device_type, []).append(node)

    def eligible(kind: str) -> list:
        allowed = ATTACK_TARGETS.get(kind, ())
        pool = [d for t in allowed for d in by_type.get(t, [])]
        return pool or list(twin.devices)

    # Draw the attack mix: one guaranteed of each class, remainder from the prior.
    total = max(attacks_per_device * len(nodes), len(kinds))
    chosen = list(kinds) + rng.choices(kinds, weights, k=total - len(kinds))

    # Assign each attack to an eligible device, balancing load across devices.
    per_device: dict[str, list[str]] = {d.device_id: [] for d in nodes}
    for kind in chosen:
        pool = eligible(kind)
        target = min(pool, key=lambda d: (len(per_device[d.device_id]), rng.random()))
        per_device[target.device_id].append(kind)

    # Lay each device's attacks out in disjoint slots on its own timeline.
    lo, hi = duration * 0.12, duration * 0.97
    plan: list[ScheduledAttack] = []
    for device_id, device_kinds in per_device.items():
        if not device_kinds:
            continue
        rng.shuffle(device_kinds)
        slot = (hi - lo) / len(device_kinds)
        guard = min(1.5, slot * 0.3)
        for i, kind in enumerate(device_kinds):
            usable = max(slot - guard, 1.0)
            dur = min(rng.uniform(4.0, 10.0), usable)
            start = lo + i * slot + rng.uniform(0.0, max(usable - dur, 0.0))
            plan.append(ScheduledAttack(
                kind=kind, start_t=start, duration=dur,
                # Squaring a uniform draw skews the mix toward *stealthy*
                # attacks. Loud attacks are the easy case; a benchmark made
                # mostly of them overstates how good any detector is.
                intensity=0.08 + 0.92 * rng.uniform(0.0, 1.0) ** 2,
                target_id=device_id,
            ))
    plan.sort(key=lambda b: b.start_t)
    return plan
