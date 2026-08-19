#!/usr/bin/env python3
"""Step 6 -- the adversarial closed loop.

Runs the experiment the report's Phase 2 describes, in four phases:

  A. federate normally                      -> the undefended detector
  B. attack it with the GAN                 -> measured degradation
  C. federate again with adversarial training -> the defended detector
  D. attack the defended detector           -> measured recovery

Also sweeps the attacker's perturbation budget, because a single evasion number
is meaningless without knowing how much power the attacker was given.

    python scripts/train_adversarial.py --rounds 12
    python scripts/train_adversarial.py --rounds 12 --set gan.epsilon=1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarmdef.config import Config, ensure_dirs
from swarmdef.data.build import build_dataset, load_pooled
from swarmdef.federated.simulation import FederatedSimulation
from swarmdef.gan.adv_train import evaluate_under_attack
from swarmdef.utils.logging import banner, get_logger
from swarmdef.utils.seed import resolve_device, set_seed

log = get_logger("train_adversarial")


def budget_sweep(model, cfg: Config, device, df, budgets) -> pd.DataFrame:
    """How evasion scales with the attacker's L-inf budget."""
    rows = []
    original = cfg.gan.epsilon
    for eps in budgets:
        cfg.gan.epsilon = eps
        r = evaluate_under_attack(model, cfg, device, df, "test", fit_epochs=cfg.gan.epochs * 2)
        rows.append({
            "epsilon": eps,
            "accuracy": round(r["adv_accuracy"], 4),
            "attack_recall": round(r["adv_recall"], 4),
            "evasion_rate": round(r["evasion_rate"], 4),
            "linf_used": round(r["linf"], 3),
            "budget_ok": r["within_budget"] and r["frozen_ok"],
        })
    cfg.gan.epsilon = original
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--skip-sweep", action="store_true")
    a = p.parse_args()

    cfg = Config.from_yaml(a.config) if Path(a.config).exists() else Config()
    for o in a.overrides:
        cfg.override(o)
    cfg.federated.n_rounds = a.rounds
    cfg.propagate_seed()
    ensure_dirs(cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    print(banner("STEP 6 -- GAN ADVERSARIAL ENGINE"))
    build_dataset(cfg)
    df = load_pooled(cfg)
    log.info("Attacker budget: L-inf <= %.2f on mutable features only "
             "(vital_zscore_max, conn_attempt_rate, n_unique_dst, flag_rst_ratio are frozen)",
             cfg.gan.epsilon)

    # ── A + B: undefended ────────────────────────────────────────────────────
    print(banner("PHASE A/B -- UNDEFENDED DETECTOR, THEN ATTACKED"))
    set_seed(cfg.seed)
    plain = FederatedSimulation(cfg, device, run_name="step6_undefended",
                                adversarial=False, track_adversarial=True)
    plain.run()
    plain_rounds, plain_clean = plain.curve("accuracy")
    plain_adv = [h.extra.get("adv_adv_accuracy", 0.0) for h in plain.history]
    undefended = evaluate_under_attack(plain.global_model, cfg, device, df, "test",
                                       fit_epochs=cfg.gan.epochs * 2)

    # ── C + D: defended ──────────────────────────────────────────────────────
    print(banner("PHASE C/D -- ADVERSARIAL TRAINING, THEN ATTACKED AGAIN"))
    set_seed(cfg.seed)
    defended = FederatedSimulation(cfg, device, run_name="step6_defended",
                                   adversarial=True, track_adversarial=True)
    defended.run()
    def_rounds, def_clean = defended.curve("accuracy")
    def_adv = [h.extra.get("adv_adv_accuracy", 0.0) for h in defended.history]
    hardened = evaluate_under_attack(defended.global_model, cfg, device, df, "test",
                                     fit_epochs=cfg.gan.epochs * 2)

    # ── results ──────────────────────────────────────────────────────────────
    print(banner("STEP 6 RESULTS -- DEGRADATION AND RECOVERY"))
    table = pd.DataFrame([
        {"detector": "undefended", "traffic": "clean",
         "accuracy": round(undefended["clean_accuracy"], 4),
         "attack_recall": round(undefended["clean_recall"], 4), "evasion": 0.0},
        {"detector": "undefended", "traffic": "GAN-evasive",
         "accuracy": round(undefended["adv_accuracy"], 4),
         "attack_recall": round(undefended["adv_recall"], 4),
         "evasion": round(undefended["evasion_rate"], 4)},
        {"detector": "adversarially trained", "traffic": "clean",
         "accuracy": round(hardened["clean_accuracy"], 4),
         "attack_recall": round(hardened["clean_recall"], 4), "evasion": 0.0},
        {"detector": "adversarially trained", "traffic": "GAN-evasive",
         "accuracy": round(hardened["adv_accuracy"], 4),
         "attack_recall": round(hardened["adv_recall"], 4),
         "evasion": round(hardened["evasion_rate"], 4)},
    ])
    print(table.to_string(index=False))

    drop = undefended["clean_recall"] - undefended["adv_recall"]
    recovered = hardened["adv_recall"] - undefended["adv_recall"]
    tax = undefended["clean_accuracy"] - hardened["clean_accuracy"]
    print(f"\n  GAN degrades attack recall by      : -{drop:.4f}"
          f"  ({100*undefended['evasion_rate']:.1f}% of attacks evade)")
    print(f"  adversarial training recovers      : +{recovered:.4f}"
          f"  ({100*hardened['evasion_rate']:.1f}% still evade)")
    print(f"  cost on clean traffic (robustness tax): {tax:+.4f} accuracy")

    sweep = None
    if not a.skip_sweep:
        print(banner("ATTACKER BUDGET SWEEP (undefended vs defended)"))
        budgets = [0.25, 0.5, 0.75, 1.0, 1.5]
        s_un = budget_sweep(plain.global_model, cfg, device, df, budgets).add_suffix("_undef")
        s_df = budget_sweep(defended.global_model, cfg, device, df, budgets).add_suffix("_def")
        sweep = pd.concat([s_un, s_df.drop(columns=["epsilon_def"])], axis=1)
        print(sweep.to_string(index=False))

    if cfg.eval.plot:
        from swarmdef.eval.plots import adversarial_impact
        adversarial_impact(
            plain_rounds, plain_clean, plain_adv, def_adv,
            Path(cfg.eval.figure_dir) / "step6_adversarial_impact.png",
        )
        log.info("Figure -> %s/step6_adversarial_impact.png", cfg.eval.figure_dir)

    out = Path(cfg.eval.log_dir) / "step6_adversarial.json"
    out.write_text(json.dumps({
        "epsilon": cfg.gan.epsilon, "rounds": a.rounds,
        "undefended": undefended, "defended": hardened,
        "clean_curve_undefended": plain_clean, "adv_curve_undefended": plain_adv,
        "clean_curve_defended": def_clean, "adv_curve_defended": def_adv,
        "budget_sweep": json.loads(sweep.to_json(orient="records")) if sweep is not None else None,
    }, indent=2, default=float))
    log.info("Results -> %s", out)


if __name__ == "__main__":
    main()
