#!/usr/bin/env python3
"""Step 2 demo -- run the hospital digital twin over a live MQTT broker.

    # terminal 1: stream telemetry (starts Mosquitto automatically)
    python scripts/run_twin.py stream --hospitals 2 --duration 120

    # terminal 2: watch it live
    python scripts/run_twin.py monitor --duration 120

    # terminal 3: trigger attacks on demand
    python scripts/run_twin.py attack --kind DDoS --hospital 0 --intensity 0.9
    python scripts/run_twin.py attack --kind MITM --hospital 0 --target infusion_pump
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarmdef.data.schema import ATTACK_CLASSES
from swarmdef.twin.monitor import LiveMonitor
from swarmdef.twin.runner import TwinRunner, capture_offline, trigger_attack
from swarmdef.utils.logging import banner, get_logger

log = get_logger("run_twin")


def cmd_stream(a: argparse.Namespace) -> None:
    print(banner("MIoT DIGITAL TWIN -- LIVE MQTT STREAM"))
    runner = TwinRunner(
        n_hospitals=a.hospitals, devices_per_hospital=a.devices, publish_hz=a.hz,
        broker_host=a.host, broker_port=a.port, speed=a.speed, seed=a.seed,
    )
    runner.run(duration_s=a.duration, schedule_attacks=a.auto_attacks)


def cmd_monitor(a: argparse.Namespace) -> None:
    LiveMonitor(broker_host=a.host, broker_port=a.port).run(duration_s=a.duration)


def cmd_attack(a: argparse.Namespace) -> None:
    trigger_attack(a.kind, hospital=a.hospital, duration=a.duration,
                   intensity=a.intensity, target_type=a.target,
                   host=a.host, port=a.port)


def cmd_capture(a: argparse.Namespace) -> None:
    """Headless capture -- no broker, straight to a labelled CSV."""
    print(banner("MIoT DIGITAL TWIN -- OFFLINE CAPTURE"))
    result = capture_offline(
        n_hospitals=a.hospitals, devices_per_hospital=a.devices,
        duration_s=a.duration, publish_hz=a.hz,
        attacks_per_device=a.attacks_per_device, seed=a.seed,
    )
    log.info(result.summary())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(out, index=False)
    log.info("Wrote %s (%d rows)", out, len(result.frame))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1883)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stream", help="publish twin telemetry to the broker")
    s.add_argument("--hospitals", type=int, default=2)
    s.add_argument("--devices", type=int, default=5)
    s.add_argument("--hz", type=float, default=5.0)
    s.add_argument("--duration", type=float, default=120.0)
    s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--auto-attacks", action="store_true", help="schedule attacks automatically")
    s.set_defaults(func=cmd_stream)

    m = sub.add_parser("monitor", help="live terminal view of the twin")
    m.add_argument("--duration", type=float, default=120.0)
    m.set_defaults(func=cmd_monitor)

    at = sub.add_parser("attack", help="trigger an attack on a running stream")
    at.add_argument("--kind", choices=ATTACK_CLASSES, default="DDoS")
    at.add_argument("--hospital", type=int, default=0)
    at.add_argument("--duration", type=float, default=10.0)
    at.add_argument("--intensity", type=float, default=0.9)
    at.add_argument("--target", default=None, help="device type, e.g. infusion_pump")
    at.set_defaults(func=cmd_attack)

    c = sub.add_parser("capture", help="headless capture to a labelled CSV")
    c.add_argument("--hospitals", type=int, default=4)
    c.add_argument("--devices", type=int, default=5)
    c.add_argument("--hz", type=float, default=5.0)
    c.add_argument("--duration", type=float, default=240.0)
    c.add_argument("--attacks-per-device", type=int, default=7)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--out", default="data/processed/twin_capture.csv")
    c.set_defaults(func=cmd_capture)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
