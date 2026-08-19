"""Drive the hospital digital twins -- online (real MQTT) or offline (in-process).

Two entry points, one simulation core:

    capture_offline()  -- run the twins headless and return labelled flow windows.
                          Used by the dataset builder; deterministic and fast.
    TwinRunner         -- publish the same packets through a live Mosquitto
                          broker in wall-clock time. Used for the demo, and it
                          accepts attack triggers on a control topic so the
                          presenter can toggle attacks mid-stream.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass

import pandas as pd

from swarmdef.data.features import FlowFeatureExtractor
from swarmdef.twin.broker import start_broker
from swarmdef.twin.hospital import HospitalTwin, ScheduledAttack, plan_attacks
from swarmdef.twin.packet import Packet
from swarmdef.utils.logging import get_logger

log = get_logger("twin.runner")

CONTROL_TOPIC = "swarmdef/control"

# Mean attack duration in seconds (plan_attacks draws uniform(4, 10)); used
# to convert a target attack density into an attack count.
MEAN_ATTACK_S = 7.0


# ───────────────────────────── offline capture ───────────────────────────────
@dataclass
class CaptureResult:
    """Labelled flow windows plus the packet-level ground truth behind them."""

    frame: pd.DataFrame
    n_packets: int
    duration_s: float

    def summary(self) -> str:
        from swarmdef.data.schema import ID_TO_LABEL
        counts = self.frame["attack_type"].value_counts().sort_index()
        parts = [f"{ID_TO_LABEL[int(k)]}={int(v)}" for k, v in counts.items()]
        return (f"{len(self.frame)} windows from {self.n_packets} packets "
                f"over {self.duration_s:.0f}s | " + " ".join(parts))


def capture_offline(
    n_hospitals: int = 4,
    devices_per_hospital: int = 5,
    duration_s: float = 240.0,
    publish_hz: float = 5.0,
    dt: float = 0.2,
    window_s: float = 1.0,
    attacks_per_device: int = 0,
    attack_fraction: float = 0.25,
    events_per_device: float = 4.0,
    non_iid: bool = True,
    seed: int = 42,
) -> CaptureResult:
    """Run every hospital twin headless and return one labelled flow table.

    If `attacks_per_device` is 0 the count is derived from `attack_fraction`, so
    that attack density stays constant regardless of capture length. Each attack
    occupies roughly `MEAN_ATTACK_S` one-second windows on its target device, and
    each device contributes about `duration_s` windows in total.
    """
    if attacks_per_device <= 0:
        attacks_per_device = max(1, round(attack_fraction * duration_s / MEAN_ATTACK_S))
        log.info("Deriving attacks_per_device=%d for a %.0f%% target attack density",
                 attacks_per_device, 100 * attack_fraction)

    frames: list[pd.DataFrame] = []
    total_packets = 0

    for h in range(n_hospitals):
        twin = HospitalTwin(h, devices_per_hospital, publish_hz, seed=seed)
        plan = plan_attacks(
            twin, duration_s, random.Random(seed * 31 + h),
            attacks_per_device=attacks_per_device, non_iid=non_iid,
        )
        twin.schedule_many(plan)
        twin.schedule_events(
            duration_s, random.Random(seed * 57 + h), events_per_device=events_per_device
        )
        extractor = FlowFeatureExtractor(
            [d.device_id for d in twin.devices], hospital_id=h, window_s=window_s,
            gateway_id=twin.gateway.device_id,
        )
        for batch in twin.run(duration_s, dt=dt):
            total_packets += len(batch)
            extractor.add(batch)
        frame = extractor.to_frame()
        log.info("Hospital %d: %d windows, %d devices, %d attacks, %d benign events",
                 h, len(frame), len(twin.devices), len(plan), len(twin.events))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return CaptureResult(combined, total_packets, duration_s)


# ────────────────────────────── live MQTT run ────────────────────────────────
class TwinRunner:
    """Publishes twin traffic to a real broker in wall-clock time."""

    def __init__(
        self,
        n_hospitals: int = 2,
        devices_per_hospital: int = 5,
        publish_hz: float = 5.0,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
        topic_root: str = "hospital",
        speed: float = 1.0,          # >1 runs the simulation faster than real time
        seed: int = 42,
    ) -> None:
        self.twins = [
            HospitalTwin(h, devices_per_hospital, publish_hz, topic_root, seed)
            for h in range(n_hospitals)
        ]
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_root = topic_root
        self.speed = speed
        self.client = None
        self._stop = threading.Event()
        self.published = 0
        self.published_attack = 0

    # ── mqtt plumbing ────────────────────────────────────────────────────────
    def _connect(self):
        import paho.mqtt.client as mqtt

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=f"swarmdef-twin-{random.randrange(1<<20)}"
        )

        def on_connect(cli, userdata, flags, reason_code, properties=None):
            log.info("Twin publisher connected to %s:%d (rc=%s)",
                     self.broker_host, self.broker_port, reason_code)
            cli.subscribe(CONTROL_TOPIC, qos=1)

        def on_message(cli, userdata, msg):
            self._handle_control(msg.payload)

        client.on_connect = on_connect
        client.on_message = on_message
        client.max_queued_messages_set(0)
        client.connect(self.broker_host, self.broker_port, keepalive=30)
        client.loop_start()
        self.client = client
        return client

    def _handle_control(self, payload: bytes) -> None:
        """Live attack toggle: publish JSON to `swarmdef/control`.

            {"cmd":"attack","kind":"DDoS","hospital":0,"duration":8,"intensity":0.9}
            {"cmd":"stop"}
        """
        try:
            msg = json.loads(payload.decode())
        except Exception:
            log.warning("Ignoring malformed control message: %r", payload[:120])
            return

        cmd = msg.get("cmd")
        if cmd == "stop":
            log.warning("Control: stop requested")
            self._stop.set()
        elif cmd == "attack":
            kind = msg.get("kind", "DDoS")
            hid = int(msg.get("hospital", 0)) % len(self.twins)
            twin = self.twins[hid]
            try:
                scenario = twin.trigger_now(
                    kind,
                    duration=float(msg.get("duration", 8.0)),
                    intensity=float(msg.get("intensity", 0.9)),
                    target_type=msg.get("target_type"),
                )
            except KeyError as exc:
                log.error("Control: %s", exc)
                return
            target = scenario.target.device_id if scenario.target else "fleet"
            log.warning("*** ATTACK TRIGGERED: %s on hospital %d -> %s (%.0fs) ***",
                        kind, hid, target, scenario.duration)
        else:
            log.warning("Control: unknown cmd %r", cmd)

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self, duration_s: float = 60.0, dt: float = 0.2, schedule_attacks: bool = False,
            manage_broker: bool = True) -> None:
        handle = start_broker(self.broker_host, self.broker_port) if manage_broker else None
        self._connect()

        if schedule_attacks:
            for twin in self.twins:
                twin.schedule_many(plan_attacks(
                    twin, duration_s, random.Random(1234 + twin.hospital_id),
                    attacks_per_device=2,
                ))

        log.info("Streaming %d hospitals x %d devices at %.1f Hz for %.0fs "
                 "(speed x%.1f). Trigger attacks on topic '%s'.",
                 len(self.twins), len(self.twins[0].devices),
                 self.twins[0].publish_hz, duration_s, self.speed, CONTROL_TOPIC)

        started = time.time()
        sim_t = 0.0
        try:
            while sim_t < duration_s and not self._stop.is_set():
                tick_start = time.time()
                for twin in self.twins:
                    for pkt in twin.tick(dt):
                        self._publish(pkt)
                sim_t += dt
                # Pace to wall clock so the demo looks like real telemetry.
                target = started + sim_t / self.speed
                sleep = target - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                elif sleep < -1.0:
                    log.debug("Publisher behind wall clock by %.1fs", -sleep)
        except KeyboardInterrupt:
            log.warning("Interrupted by user")
        finally:
            self.stop()
            if handle is not None:
                handle.stop()

        log.info("Published %d messages (%d malicious, %.1f%%) over %.0fs",
                 self.published, self.published_attack,
                 100.0 * self.published_attack / max(self.published, 1), sim_t)

    def _publish(self, pkt: Packet) -> None:
        """Publish one packet. Floods are summarised rather than sent 1:1.

        A 300 pkt/s flood would swamp the broker's client queue and starve the
        legitimate telemetry we want to keep visible in the demo. So volumetric
        attack traffic is published as periodic burst summaries carrying the
        true packet count -- the monitor reconstructs identical flow features
        from them, but the terminal stays readable.
        """
        if self.client is None:
            return
        topic = pkt.topic or f"{self.topic_root}/{pkt.dst}/raw"
        # MQTT forbids wildcards in publish topics; sanitise defensively so a
        # malformed attack topic can never take the live demo down.
        if "#" in topic or "+" in topic:
            topic = topic.replace("#", "_all").replace("+", "_any")
        body = {
            "t": round(pkt.t, 4), "src": pkt.src, "dst": pkt.dst,
            "type": pkt.msg_type, "proto": pkt.protocol,
            "bytes": pkt.size, "payload_len": pkt.payload_len,
            "flags": pkt.flags, "qos": pkt.qos, "retain": pkt.retain, "dup": pkt.dup,
            "label": pkt.label, "vital_z": round(pkt.vital_z, 3),
        }
        if pkt.payload:
            body["payload"] = pkt.payload
        try:
            self.client.publish(topic, json.dumps(body, separators=(",", ":")), qos=0)
        except (ValueError, OSError) as exc:
            log.debug("Dropped unpublishable packet on %r: %s", topic, exc)
            return
        self.published += 1
        if pkt.label != "BENIGN":
            self.published_attack += 1

    def stop(self) -> None:
        self._stop.set()
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None


def trigger_attack(kind: str, hospital: int = 0, duration: float = 8.0,
                   intensity: float = 0.9, target_type: str | None = None,
                   host: str = "127.0.0.1", port: int = 1883) -> None:
    """One-shot helper: publish an attack command to a running TwinRunner."""
    import paho.mqtt.client as mqtt

    payload = json.dumps({
        "cmd": "attack", "kind": kind, "hospital": hospital,
        "duration": duration, "intensity": intensity, "target_type": target_type,
    })
    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"swarmdef-ctl-{random.randrange(1<<16)}")
    cli.connect(host, port, keepalive=10)
    cli.loop_start()
    cli.publish(CONTROL_TOPIC, payload, qos=1).wait_for_publish(timeout=5)
    time.sleep(0.2)
    cli.loop_stop()
    cli.disconnect()
    log.info("Sent control command: %s", payload)
