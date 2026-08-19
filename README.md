# Swarm-Dynamic Federated Adversarial Defense for Medical IoT

A closed-loop intrusion-detection pipeline for Medical IoT (MIoT):

```
digital twin → GNN detector → GAN adversary → DP-SGD → Byzantine-robust aggregation
```

Several simulated hospitals each run an anomaly detector on their own MQTT
telemetry (raw data never leaves the hospital), a GAN searches for stealthy
evasive attacks, the hospitals merge what they learned via Federated Learning,
and every merge is protected by Differential Privacy and Byzantine-robust
aggregation.

**Status: Steps 1–6 built and executed. Step 7 (DP) not implemented; Step 8
implemented but not yet run. See [Not yet built](#not-yet-built) for the full
picture.**

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Step 2 — live digital twin (3 terminals)
.venv/bin/python scripts/run_twin.py stream  --hospitals 2 --duration 120
.venv/bin/python scripts/run_twin.py monitor --duration 120
.venv/bin/python scripts/run_twin.py attack  --kind DDoS --hospital 0 --intensity 0.9

# Step 3 — build and verify the federated dataset
.venv/bin/python scripts/build_dataset.py --force
.venv/bin/python scripts/verify_dataset.py

# Step 4 — centralised baseline (all four architectures)
.venv/bin/python scripts/train_baseline.py
.venv/bin/python scripts/train_baseline.py --multiclass

# Step 5 — federated learning
.venv/bin/python scripts/train_federated.py --compare-local
.venv/bin/python scripts/train_federated_flower.py --rounds 10   # real Flower

# Step 6 — the adversarial closed loop
.venv/bin/python scripts/train_adversarial.py --rounds 10 --set gan.epsilon=1.0
```

`scripts/run_twin.py stream` starts Mosquitto automatically.

Any config field can be overridden from the CLI:

```bash
--set federated.aggregator=krum          # fedavg | krum | multikrum | trimmed_mean | median
--set federated.byzantine_clients=1      # compromise the last N hospitals
--set federated.byzantine_attack=scale   # sign_flip | gauss | scale | label_flip
--set data.partition=dirichlet --set data.dirichlet_alpha=0.2
--set detector.arch=transformer
--set gan.epsilon=1.5
```

---

## Repository layout

```
swarmdef/
  config.py            typed config for every experiment (YAML + CLI overrides)
  twin/                Step 2 — the MQTT digital twin
    devices.py         5 clinical device models + the ward Gateway
    packet.py          the wire-level Packet record and FlowWindow
    attacks.py         6 attack scenarios (DDoS, MITM, FirmwareTamper, …)
    events.py          6 BENIGN confounder events (code blue, imaging, …)
    hospital.py        one hospital: fleet + attack/event scheduler + clock
    broker.py          Mosquitto lifecycle management
    runner.py          online (real MQTT) and offline (headless) drivers
    monitor.py         live terminal view of the traffic
  data/                Step 3 — dataset construction
    schema.py          the canonical 30-feature schema + label vocabulary
    features.py        packets → per-device flow windows → 30-D vectors
    cic.py             CIC-IoT2023 loader, projected onto the same schema
    scaler.py          log1p + robust median/IQR normalisation
    partition.py       non-IID sharding (native + Dirichlet) and skew metrics
    build.py           the end-to-end builder
  detector/            Step 4 — the defence layer
    models.py          MLP / Transformer / GNN (SAGE, GCN, GAT) detectors
    graph.py           device-graph construction + the Transformer sequence view
    data.py            one loader interface for all three architectures
    train.py           the shared training / evaluation loop
  gan/                 Step 6 — the attack layer
    generator.py       bounded, mask-respecting perturbation generator
    engine.py          trains the generator against the live detector
    adv_train.py       per-hospital adversary + adversarial retraining
  federated/           Steps 5 & 8 — orchestration
    client.py          HospitalClient, incl. 4 Byzantine poisoning modes
    aggregators.py     fedavg | krum | multikrum | trimmed_mean | median
    simulation.py      the deterministic federated loop used for experiments
    flower_app.py      the same loop under the real Flower framework
  privacy/             Step 7 — NOT YET IMPLEMENTED (empty package)
  eval/                logging, shared figure styling, result plots
  utils/               seeding, logging, detection metrics
scripts/
  run_twin.py               stream / monitor / attack / capture
  build_dataset.py          Step 3 entry point
  verify_dataset.py         6-check verification suite
  train_baseline.py         Step 4 — centralised baseline
  train_federated.py        Step 5 — FedAvg + isolated-hospital comparison
  train_federated_flower.py Step 5 — the Flower deployment path
  train_adversarial.py      Step 6 — attack / defend / budget sweep
configs/default.yaml   default experiment configuration
data/hospitals/        hospital_0..3.csv  (the federated shards)
data/processed/        pooled.csv, scaler.json, dataset_meta.json
results/               logs/ (CSV + JSON per run), models/, figures/
```

48 Python files, ~6,700 lines.

---

## What is built so far

### Step 1 — Environment
`torch 2.13`, `torch-geometric 2.8`, `flwr 1.33`, `opacus 1.6`, `paho-mqtt 2.1`,
`pandas 3.0`, `numpy 2.5` on Python 3.14 (arm64). PyTorch Geometric — flagged in
the plan as the most likely source of environment pain — was installed and
smoke-tested in Step 1 rather than Week 5. Mosquitto is installed natively via
Homebrew instead of Docker (one 400 KB binary vs. a container runtime); the
Docker path still works if preferred.

`resolve_device()` deliberately resolves `auto` to CPU rather than Apple MPS:
Opacus' per-sample gradient hooks and several PyG kernels silently fall back or
error on MPS. Request `mps` explicitly to opt in.

### Step 2 — Digital twin
- **5 device types** (patient monitor, infusion pump, ventilator, wearable,
  environment sensor) driven by bounded Ornstein–Uhlenbeck vital-sign processes,
  plus a **Gateway** node carrying benign segment traffic (EHR, PACS, DNS/NTP).
- **6 attack scenarios**, each with a genuinely different feature signature:
  `DDoS` (volume), `MITM` (clinical implausibility only — volume unchanged),
  `FirmwareTamper` (large high-entropy OTA writes), `Spoofing` (identity),
  `Recon` (topic/destination fan-out), `Mirai` (brute force then C2 beacon).
- **6 benign confounder events** that deliberately mimic attacks — `CodeBlue`
  (looks like MITM), `AuthorisedFirmwareUpdate` (looks like FirmwareTamper),
  `ImagingTransfer`, `DeviceReboot`, `ShiftHandover`, `NetworkCongestion`.
- Live MQTT streaming with **on-demand attack triggering** over a control topic,
  and a refreshing terminal monitor.

### Step 3 — Dataset
- Packets are binned into **per-device 1-second flow windows** and summarised by
  the **30-feature schema** in `swarmdef/data/schema.py`.
- **12,600 windows**, 24.9% attack, all 7 classes present at every hospital.
- Split **train/val/test = 8824/1272/2504**, at the **graph level** (a whole
  `(hospital, time-window)` goes to one split, so the GNN's evaluation is
  inductive), stratified on the 7-way class, and performed *before* the scaler
  is fitted (no leakage).
- Sharded into **4 non-IID hospitals**: TV-distance from the pooled threat mix
  = **0.284** across attack classes (0.078 including benign). Each hospital has
  a different dominant threat — H0 DDoS, H1 MITM, H2 and H3 Spoofing.
- A `CIC-IoT2023` loader is implemented and maps that dataset's 46 native
  columns and 33 attack labels onto the same 30-feature / 7-class schema. Drop
  CSVs into `data/raw/cic_iot2023/` and the pipeline switches source
  automatically. **Not yet run on real data.**

**Verification** (`scripts/verify_dataset.py`, 6/6 passing): schema integrity,
stratified splits, no scaler leakage, measured non-IID skew, learnability, and
shard trainability.

### Step 4 — Detector architectures
Three models over identical features, identical splits and an identical training
loop, so the comparison is controlled. Binary detection on the test split
(n = 2504):

| architecture | params | accuracy |    F1 | precision | recall |   FPR |   AUC | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP          |  6,530 | 0.9788 | 0.9576 | 0.9214 | 0.9967 | 0.0268 | 0.9975 | 1.2 |
| Transformer  | 69,186 | 0.9764 | 0.9527 | 0.9181 | 0.9900 | 0.0278 | 0.9970 | 16.2 |
| **GNN/SAGE** | 18,690 | **0.9804** | **0.9605** | **0.9311** | 0.9917 | **0.0231** | **0.9986** | 1.3 |
| GNN/GAT      | 12,930 | 0.9780 | 0.9558 | 0.9238 | 0.9900 | 0.0257 | 0.9979 | 1.8 |

7-way attack typing gives GNN/SAGE 0.9736 multi-class accuracy, and it is the
only architecture at ≥0.967 recall on *every* one of the seven classes.

On clean traffic the margin over the MLP is small — about 0.2 accuracy points.
That is the honest reading, and Step 4 is a ceiling check rather than the
architecture argument. The GNN wins where it matters operationally (lowest FPR,
highest AUC, uniform per-class recall, 3.7× fewer parameters than the
Transformer), and the case for context is made by the `CodeBlue` vs `MITM`
construction (below) and by behaviour under attack in Step 6.

### Step 5 — Federated learning
Same detector, 4 hospitals, FedAvg, 15 rounds, raw data never leaves a hospital.

| setting | accuracy | F1 | recall | FPR |
|---|---:|---:|---:|---:|
| centralised baseline (GNN/SAGE, pooled data) | 0.9804 | 0.9605 | 0.9917 | 0.0231 |
| **federated, best (round 12)** | **0.9752** | **0.9502** | 0.9850 | 0.0278 |
| federated, final (round 15) | 0.9740 | 0.9476 | 0.9800 | 0.0278 |

**Privacy cost of federation: −0.0052 accuracy** against a pooled baseline that
would be illegal in practice. Convergence is clean and monotone: 0.861 → 0.937 →
0.962 → 0.975 over twelve rounds, with FPR falling from 9.98% to 2.78%.

Two harnesses share the *same* `HospitalClient` and the *same* `aggregate()`
function: `federated/simulation.py` (deterministic, instrumentable — used for all
experiments) and `federated/flower_app.py` (the real Flower client/server
protocol — the deployment path). Five aggregation rules are implemented: FedAvg,
Krum, Multi-Krum, coordinate-wise trimmed mean, coordinate-wise median.

### Step 6 — GAN adversarial engine
The adversary controls a host inside the ward and wants its traffic classified
BENIGN. It may reshape *how* the attack looks on the wire, but not what the
attack does:

```
x_adv = x + ε · tanh(G(x, z)) · mutable_mask
```

bounded in L∞, masked to features an attacker could actually control, and applied
in the same normalised space the detector consumes. This is a **white-box**
attack — the generator differentiates through the current detector, and is
retrained fresh against the global model every round.

Results at ε = 1.0, 4 hospitals, 10 federated rounds, 600 attack samples:

| detector | traffic | accuracy | F1 | attack recall | evasion |
|---|---|---:|---:|---:|---:|
| undefended | clean | 0.9665 | 0.9341 | 0.9917 | 0.8% |
| undefended | **GAN-evasive** | 0.8998 | 0.7733 | **0.7133** | **28.3%** |
| adversarially trained | clean | 0.9760 | 0.9520 | 0.9917 | 0.8% |
| adversarially trained | **GAN-evasive** | 0.9681 | 0.9350 | **0.9583** | **4.3%** |

The GAN costs the undefended detector 27.8 points of attack recall. Adversarial
training recovers 24.5 of them — and the **robustness tax is negative**: clean
accuracy also improved, 0.9665 → 0.9760. The likely mechanism is that bounded,
mask-respecting perturbations act as targeted augmentation along exactly the
directions the detector was weakest in; a larger budget would probably reverse
the sign, and the sweep already shows defended clean accuracy sliding from
0.9752 at ε = 0.25 to 0.9573 at ε = 1.5.

A single evasion number would be meaningless without knowing the attacker's
power, so evasion is swept across five budgets:

| ε | undefended evasion | defended evasion |
|---:|---:|---:|
| 0.25 |  3.8% | 1.3% |
| 0.50 |  9.0% | 1.5% |
| 0.75 | 13.7% | 2.2% |
| 1.00 | 27.8% | 2.8% |
| 1.50 | **54.0%** | **8.7%** |

Budget compliance (`within_budget`, `frozen_features_untouched`) is asserted at
every point.

**The security claim, made measurable.** Each hospital runs its own generator
against its own copy of the detector, so four adversaries search four different
regions of the feature space — H0's DDoS-shaped blind spots, H1's MITM-shaped
ones, and so on. Aggregation then propagates every hospital's hardening to all
the others. A blind spot found by *one* hospital is patched for *all* of them,
without any hospital sharing a packet.

---

## Calibrated difficulty

The first version of the dataset was solved by plain logistic regression at
**99.9%** accuracy. That is a defective benchmark for this project: with a
detector already at ceiling there is no headroom for the GAN to degrade, no
visible cost to DP noise, and no measurable damage from a poisoned client —
every downstream plot would be a flat line at 1.0.

Three changes fixed it, all of them corrections toward realism rather than
artificial noise:

1. **Benign confounder events** (`twin/events.py`) — legitimate activity that
   shares attack signatures.
2. **Bounded attacker source pools** — a real botnet controls a finite set of
   hosts. Drawing a fresh random source per packet had made `n_unique_src` a
   perfect give-away that nothing benign could ever produce.
3. **Stealth-skewed attack intensity** — intensities are drawn from a squared
   uniform, so most attacks are low-and-slow. Loud attacks are the easy case.

Linear-probe accuracy is now **95.8%** (vs. a 76.0% majority-class baseline) with
a 3.7% benign false-positive rate — a realistic IDS operating point. Per-class
probe recall on the test split:

| BENIGN | DDoS | MITM | FirmwareTamper | Spoofing | Recon | Mirai |
|---:|---:|---:|---:|---:|---:|---:|
| 0.963 | 1.000 | **0.806** | 0.932 | 0.991 | 0.988 | 1.000 |

MITM being the hardest class is the confounders working as designed: `CodeBlue`
is constructed to be indistinguishable from `MITM` in the victim device's own
feature vector, so a linear probe on a single window genuinely cannot separate
them.

### Why the detector is a GNN

`CodeBlue` and `MITM` produce the *same* `vital_zscore_max` distribution by
construction — the MITM tamper uses the same direction convention and the same
sigma-scaled magnitude range as a real resuscitation. What separates them is the
ward's reaction: a real code blue is a correlated, ward-wide event (rapid-response
paging, EHR charting at the gateway, neighbouring devices handled), while an MITM
produces one alarming device surrounded by a completely calm ward.

That evidence does not exist in the victim's own 30-D vector. No per-window model
can make the call; a graph model can, because it sees whether the neighbours
agree. `SAGEConv` specifically, because it keeps separate weights for a node's own
features and for its neighbours' mean — a convolution that averaged them together
would blur exactly the contrast being tested.

---

## Design decisions worth defending

| Decision | Rationale |
|---|---|
| One 30-D schema for twin *and* CIC-IoT2023 | A model trained on real captures can be evaluated on live twin traffic unchanged. |
| Window-level labels ("any attack packet ⇒ malicious") | Standard IDS convention; forces the detector to spot a hostile minority inside benign traffic. |
| Gateway pseudo-device | Recon and Spoofing address no specific device. Without a gateway node they were being silently dropped, losing two whole attack classes. |
| Attacks concurrent across devices, disjoint per device | Maximises attack density while keeping window labels unambiguous. |
| Group-level (inductive) splitting | Row-level splitting would let a test node's features reach the model through GNN message passing — transductive leakage. Whole time-windows go to one split, so the model is tested on graphs it has never seen any part of. That is also the deployment reality for an IDS. |
| Scaler fitted centrally on the train split | A mild relaxation of data isolation, standard in the FL-IDS literature. The alternative — per-client scalers — makes each client's weights describe a different input space, so averaging is not meaningful. Only 30 medians + 30 IQRs are shared, and a real deployment would obtain them via one DP-protected secure-aggregation round. |
| `IMMUTABLE_FEATURES` in the schema | The Step 6 GAN may only perturb features an attacker could actually control without breaking the attack's function. `vital_zscore_max`, `conn_attempt_rate`, `n_unique_dst` and `flag_rst_ratio` are frozen. |
| Native (not Dirichlet) partition by default | The twin's hospitals already differ in device fleet *and* threat mix — non-IID at the source, which is more realistic than relabelling a pooled set. Dirichlet is implemented and used for CIC-IoT2023, which has no hospital structure. |
| LayerNorm everywhere, never BatchNorm | Opacus cannot compute per-sample gradients through BatchNorm, and BatchNorm is a known cause of silent FedAvg divergence with non-IID client batches. Choosing it in Step 4 means Steps 5 and 7 need no architectural rewrite. |
| Hand-written self-attention in the Transformer | Opacus cannot attach per-sample hooks to `nn.MultiheadAttention`'s fused packed-projection weight. Every parameter here lives in an ordinary `Linear`. |
| No separate discriminator in the GAN | The detector *is* the discriminator. The quantity of interest is evasion of the deployed detector, not sample realism against an auxiliary critic. |
| The adversary is retrained fresh every round | A real attacker re-optimises against whatever is currently deployed; reusing a stale generator would understate the threat. |
| Two federated harnesses | The simulator is deterministic and instrumentable, which the ablations need; Flower proves the same clients and rules run under a real framework. Both call the *same* `aggregate()`, so there is no second implementation to diverge. |

---

## Not yet built

Named honestly, because the gaps matter as much as the results.

| Item | What exists | What is missing |
|---|---|---|
| **Step 7 — Differential privacy** | Opacus installed; `PrivacyConfig` complete (clipping norm, σ, target δ/ε, RDP/PRV accountant); the `privacy_engine` hook and `privatise()` call site are wired into `HospitalClient.fit()`; the training loop accepts a pre-built Opacus optimizer; the architecture is BatchNorm-free; `eval/plots.py::privacy_tradeoff()` is ready | `swarmdef/privacy/` is an empty package. No DP-SGD engine, no ε accounting, no privacy/utility experiment. |
| **Step 8 — Byzantine experiments** | All 5 aggregators and all 4 poisoning modes (`sign_flip`, `scale`, `gauss`, `label_flip`) implemented; `byzantine_comparison()` figure ready; config fields present | No experiment script and no run. `byzantine_clients` is 0 in every result currently on disk. |
| **CIC-IoT2023 validation** | Complete loader with the full 33-label mapping | No CSVs in `data/raw/`. Never executed on real capture data — every number in this README comes from the digital twin. |
| **Tests** | `verify_dataset.py`'s 6 checks; in-code invariant assertions (budget compliance, frozen-feature checks, degenerate-shard guards) | `tests/` is empty. No unit tests. |
| **Steps 9–10** | 9 figures generated; `eval/plots.py` complete | No consolidated final evaluation sweep or written report. |

### Known issues

- **The Flower run on disk did not converge.** `results/logs/step5_flower.csv`
  shows accuracy frozen at 0.7304 and AUC 0.403 across all 8 rounds — identical
  values every round plus an AUC below 0.5 is the signature of a global model
  stuck at initialisation, the Ray worker-serialisation failure documented in
  `federated/flower_app.py`. Every federated number quoted above comes from
  `federated/simulation.py`, which converged correctly
  (`results/logs/step5_fedavg_gnn.csv`). Re-verifying the Flower path is open.
- **Step 6 was run at ε = 1.0**, twice the configured default of 0.5. At the
  default the corresponding evasion figures are 9.0% → 1.5% (see the sweep).

---

## Reproducibility

Every run is seeded (`swarmdef/utils/seed.py`) and fully described by its
`Config`, which is serialised alongside its results.
`data/processed/dataset_meta.json` records the source, split sizes, class counts,
skew statistics and seed for the dataset currently on disk, so any number above
can be traced back to the configuration that produced it.

A full step-by-step engineering record — every module, every design decision,
and every measured result — is in `PROJECT_BUILD_LOG.pdf`.
