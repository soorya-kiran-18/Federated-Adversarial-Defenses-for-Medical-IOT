"""Swarm-Dynamic Federated Adversarial Defense for Medical IoT.

A closed-loop intrusion-detection pipeline for Medical IoT (MIoT):

    digital twin -> GNN detector -> GAN adversary -> DP-SGD -> robust aggregation

Sub-packages
------------
twin       : MQTT digital twin of hospital medical devices + attack injection
data       : CIC-IoT2023 ingestion, feature engineering, non-IID sharding
detector   : GNN / Transformer anomaly detectors (defence layer)
gan        : adversarial perturbation engine (attack layer)
privacy    : Opacus DP-SGD wrapper + privacy accounting (security layer)
federated  : Flower orchestration + Byzantine-robust aggregators
eval       : metric logging and figure generation
"""

__version__ = "1.0.0"
__all__ = ["config", "twin", "data", "detector", "gan", "privacy", "federated", "eval"]
