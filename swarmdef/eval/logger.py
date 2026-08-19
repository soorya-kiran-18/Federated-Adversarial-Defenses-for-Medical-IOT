"""Append-only experiment log: one CSV row per federated round, plus JSON runs.

Every step from 5 onward produces a per-round time series (accuracy, F1,
epsilon, evasion rate). Writing them through one logger means the figures in
Step 9 and the tables in Step 10 read from a single schema instead of from a
different ad-hoc dict per experiment.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoundRecord:
    """One federated round's outcome."""

    run: str
    round: int
    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    fpr: float = 0.0
    auc: float = 0.0
    evasion_rate: float = 0.0
    loss: float = 0.0
    epsilon: float = 0.0
    adv_accuracy: float = 0.0
    n_clients: int = 0
    n_byzantine: int = 0
    aggregator: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class ExperimentLogger:
    """Collects `RoundRecord`s and writes CSV + JSON side by side."""

    FIELDS = [f for f in RoundRecord.__dataclass_fields__ if f != "extra"]

    def __init__(self, run_name: str, log_dir: str | Path) -> None:
        self.run_name = run_name
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.records: list[RoundRecord] = []
        self.meta: dict[str, Any] = {"run": run_name, "started": time.time()}

    def log(self, record: RoundRecord) -> RoundRecord:
        self.records.append(record)
        return record

    def set_meta(self, **kw: Any) -> None:
        self.meta.update(kw)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame([
            {**{k: getattr(r, k) for k in self.FIELDS}, **r.extra} for r in self.records
        ])

    def save(self) -> tuple[Path, Path]:
        csv_path = self.dir / f"{self.run_name}.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS)
            writer.writeheader()
            for r in self.records:
                writer.writerow({k: getattr(r, k) for k in self.FIELDS})

        json_path = self.dir / f"{self.run_name}.json"
        json_path.write_text(json.dumps(
            {"meta": self.meta, "rounds": [asdict(r) for r in self.records]}, indent=2, default=str
        ))
        return csv_path, json_path
