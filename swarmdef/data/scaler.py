"""Feature normalisation for heavy-tailed network flow features.

Why this is not just StandardScaler
-----------------------------------
Flow features span many orders of magnitude: a benign device publishes at 5
packets/s while a flood reaches 400, and byte rates differ by 10^4. Fitting a
plain z-score on those raw values gives the flood a z of +30 and squashes every
benign distinction to ~0, which destroys the signal the detector needs for the
*stealthy* attacks (MITM, spoofing). So heavy-tailed features get a log1p
transform first, then a robust median/IQR scale.

Why the scaler is fit centrally, and what that costs
----------------------------------------------------
Strictly, fitting one scaler on pooled data is a small violation of federated
data isolation. It is retained deliberately, and it is the standard treatment
in the FL-IDS literature, because the alternative -- per-client scalers -- makes
each client's weights describe a different input space, so averaging them is
not meaningful. In a real deployment this would be replaced by a single
secure-aggregation round in which clients contribute DP-noised feature
quantiles. The statistics shared here are 30 medians and 30 IQRs, not records.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from swarmdef.data.schema import FEATURE_NAMES

# Features whose distribution is heavy-tailed and strictly non-negative.
LOG_FEATURES: set[str] = {
    "packet_rate", "byte_rate", "pkt_size_mean", "pkt_size_std", "pkt_size_min",
    "pkt_size_max", "payload_len_mean", "header_len_mean", "flow_duration",
    "iat_mean", "iat_std", "n_unique_src", "n_unique_dst", "n_unique_topic",
    "conn_attempt_rate", "vital_zscore_max",
}


class FlowScaler:
    """log1p on heavy-tailed columns, then a robust median/IQR z-score."""

    def __init__(self) -> None:
        self.log_mask: np.ndarray | None = None
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray | pd.DataFrame) -> "FlowScaler":
        arr = np.asarray(X, dtype=np.float64)
        self.log_mask = np.array([f in LOG_FEATURES for f in FEATURE_NAMES])
        arr = self._log(arr)
        self.center = np.median(arr, axis=0)
        q75, q25 = np.percentile(arr, [75, 25], axis=0)
        iqr = q75 - q25
        # Constant columns (IQR 0) would divide by zero; a scale of 1 leaves
        # them centred at 0, which is exactly right for an uninformative feature.
        self.scale = np.where(iqr > 1e-9, iqr / 1.349, 1.0)
        return self

    def _log(self, arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[:, self.log_mask] = np.log1p(np.clip(out[:, self.log_mask], 0, None))
        return out

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.center is None:
            raise RuntimeError("FlowScaler.transform called before fit")
        arr = self._log(np.asarray(X, dtype=np.float64))
        out = (arr - self.center) / self.scale
        # Clip to a sane range: a single extreme flow should not dominate the
        # first gradient step, and DP-SGD clipping is easier on bounded inputs.
        return np.clip(out, -10.0, 10.0).astype(np.float32)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "features": FEATURE_NAMES,
            "log_mask": self.log_mask.tolist(),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FlowScaler":
        blob = json.loads(Path(path).read_text())
        obj = cls()
        obj.log_mask = np.array(blob["log_mask"], dtype=bool)
        obj.center = np.array(blob["center"], dtype=np.float64)
        obj.scale = np.array(blob["scale"], dtype=np.float64)
        return obj
