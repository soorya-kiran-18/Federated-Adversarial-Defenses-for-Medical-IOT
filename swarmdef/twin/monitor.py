"""Live MQTT monitor -- the watch-the-attack-happen view for the demo.

Subscribes to the twin's telemetry, maintains per-device rolling statistics and
prints a refreshing table. If a trained detector is supplied it also scores each
flow window live, so the presenter can watch the model catch (or miss) an attack
the moment it is triggered.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from swarmdef.utils.logging import get_logger

log = get_logger("twin.monitor")

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[38;5;196m"
GREEN = "\033[38;5;42m"
YELLOW = "\033[38;5;214m"
CYAN = "\033[38;5;44m"
GREY = "\033[38;5;244m"

LABEL_COLOR = {
    "BENIGN": GREEN, "DDoS": RED, "MITM": YELLOW, "FirmwareTamper": RED,
    "Spoofing": YELLOW, "Recon": CYAN, "Mirai": RED,
}


@dataclass
class DeviceStats:
    """Rolling statistics for one device, over a sliding wall-clock window."""

    device_id: str
    device_type: str = "?"
    hospital_id: int = 0
    events: deque = field(default_factory=lambda: deque(maxlen=4000))
    last_readings: dict = field(default_factory=dict)
    last_label: str = "BENIGN"
    attack_events: int = 0
    total_events: int = 0

    def add(self, now: float, size: int, label: str, readings: dict | None) -> None:
        self.events.append((now, size, label))
        self.total_events += 1
        if label != "BENIGN":
            self.attack_events += 1
        self.last_label = label
        if readings:
            self.last_readings = readings

    def prune(self, now: float, window: float) -> None:
        while self.events and now - self.events[0][0] > window:
            self.events.popleft()

    def rates(self, window: float) -> tuple[float, float, str]:
        if not self.events:
            return 0.0, 0.0, "BENIGN"
        span = max(self.events[-1][0] - self.events[0][0], window * 0.25, 1e-3)
        pkts = len(self.events) / span
        byts = sum(e[1] for e in self.events) / span
        recent = [e[2] for e in self.events if e[2] != "BENIGN"]
        return pkts, byts, (recent[-1] if recent else "BENIGN")


class LiveMonitor:
    """Subscribing client that renders the twin's traffic as a live table."""

    def __init__(
        self,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
        topic_root: str = "hospital",
        window_s: float = 3.0,
        refresh_hz: float = 4.0,
        detector_fn=None,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = f"{topic_root}/#"
        self.window_s = window_s
        self.refresh_period = 1.0 / refresh_hz
        self.detector_fn = detector_fn
        self.stats: dict[str, DeviceStats] = {}
        # device_id -> (device_type, hospital_id), learned from telemetry topics
        self.known_devices: dict[str, tuple[str, int]] = {}
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.client = None
        self.started = time.time()
        self.alerts: deque = deque(maxlen=6)

    # ── mqtt ─────────────────────────────────────────────────────────────────
    def _on_message(self, cli, userdata, msg) -> None:
        """Attribute one observed message to a device row.

        A flood spoofs hundreds of source addresses; keying the table on the
        source would explode it into hundreds of one-packet rows and hide the
        device actually under attack. So we mirror the feature extractor's
        attribution: a *known* device endpoint wins, then a device named in the
        topic (this catches spoofed publishes and forged OTA pushes), and any
        remaining segment traffic rolls up to that hospital's gateway row.
        """
        try:
            body = json.loads(msg.payload.decode())
        except Exception:
            return

        parts = msg.topic.split("/")
        try:
            hid = int(parts[1])
        except (IndexError, ValueError):
            hid = 0

        # Telemetry topics are how the monitor learns the real device inventory:
        #   hospital/<hid>/<device_type>/<device_id>/telemetry
        if len(parts) >= 5 and parts[-1] == "telemetry":
            self.known_devices[parts[3]] = (parts[2], hid)

        src, dst = body.get("src", "?"), body.get("dst", "?")
        if src in self.known_devices:
            dev_id = src
        elif dst in self.known_devices:
            dev_id = dst
        else:
            dev_id = next((p for p in parts if p in self.known_devices), None)
        if dev_id is None:
            dev_id = f"H{hid}-gateway"
            self.known_devices.setdefault(dev_id, ("gateway", hid))

        dev_type, dev_hid = self.known_devices.get(dev_id, ("gateway", hid))
        label = body.get("label", "BENIGN")
        payload = body.get("payload") or {}
        now = time.time()

        with self.lock:
            st = self.stats.get(dev_id)
            if st is None:
                st = DeviceStats(dev_id, dev_type, dev_hid)
                self.stats[dev_id] = st
            st.add(now, int(body.get("bytes", 0)), label, payload.get("readings"))
            if label != "BENIGN" and (
                not self.alerts
                or self.alerts[-1][1] != label
                or self.alerts[-1][2] != dev_id
                or now - self.alerts[-1][0] > 2.0
            ):
                self.alerts.append((now, label, dev_id))

    def connect(self):
        import paho.mqtt.client as mqtt

        cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"swarmdef-mon-{int(time.time())}")
        cli.on_message = self._on_message
        cli.on_connect = lambda c, u, f, rc, p=None: c.subscribe(self.topic, qos=0)
        cli.connect(self.broker_host, self.broker_port, keepalive=30)
        cli.loop_start()
        self.client = cli
        log.info("Monitor subscribed to %s on %s:%d", self.topic, self.broker_host, self.broker_port)
        return cli

    # ── rendering ────────────────────────────────────────────────────────────
    def render(self) -> str:
        now = time.time()
        width = min(shutil.get_terminal_size((110, 40)).columns, 130)
        lines = [
            f"{BOLD}Swarm-Dynamic Federated Adversarial Defense -- Live MIoT Twin{RESET}"
            f"{GREY}   t+{now - self.started:6.1f}s{RESET}",
            f"{GREY}{'-' * width}{RESET}",
            f"{BOLD}{'HOSP':<5}{'DEVICE':<26}{'TYPE':<17}{'pkt/s':>8}{'KB/s':>9}"
            f"  {'STATUS':<16}{'VITALS'}{RESET}",
        ]
        with self.lock:
            devices = sorted(self.stats.values(), key=lambda s: (s.hospital_id, s.device_id))
            for st in devices:
                st.prune(now, self.window_s)
                pkts, byts, label = st.rates(self.window_s)
                colour = LABEL_COLOR.get(label, GREEN)
                status = "NORMAL" if label == "BENIGN" else f"** {label} **"
                vitals = "  ".join(
                    f"{GREY}{k[:11]}{RESET}={v}" for k, v in list(st.last_readings.items())[:3]
                ) or f"{DIM}(segment traffic){RESET}"
                lines.append(
                    f"{st.hospital_id:<5}{st.device_id[:25]:<26}{st.device_type[:16]:<17}"
                    f"{pkts:>8.1f}{byts / 1024:>9.1f}  {colour}{status:<16}{RESET}{vitals}"
                )
            alerts = list(self.alerts)

        lines.append(f"{GREY}{'-' * width}{RESET}")
        if alerts:
            lines.append(f"{BOLD}RECENT ALERTS{RESET}")
            for ts, label, dev in reversed(alerts):
                c = LABEL_COLOR.get(label, RED)
                lines.append(f"  {GREY}t+{ts - self.started:6.1f}s{RESET}  {c}{label:<16}{RESET}{dev}")
        else:
            lines.append(f"{GREEN}  no anomalies detected{RESET}")
        return "\n".join(lines)

    def run(self, duration_s: float = 60.0) -> None:
        self.connect()
        deadline = time.time() + duration_s
        try:
            while time.time() < deadline and not self._stop.is_set():
                # Clear screen + home cursor, then repaint.
                print("\033[2J\033[H" + self.render(), flush=True)
                time.sleep(self.refresh_period)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
            self.client = None
