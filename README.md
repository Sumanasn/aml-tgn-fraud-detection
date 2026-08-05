# AML-TGN: Graph-Based Money-Siphoning & Shell-Ring Detection

Detects cyclic transfers and shell-account laundering rings in transaction
graphs, comparing a tabular baseline against a GraphSAGE GNN and a Temporal
Graph Network (TGN) — and honestly tests the claim that graph-aware models
catch these rings *earlier* than volume/degree-based tabular features can.

## Why this exists

Standard fraud tooling flags accounts by transaction volume, velocity, or
degree — signals that only accumulate *after* a laundering ring has been
moving money for a while. A cyclic transfer ring (money in, laundered
through N shell accounts, money out) is a **structural** pattern, visible in
the graph topology before it's visible in any single account's statistics.
The question this project actually tests: does a model that can see graph
structure detect the ring earlier than one that can't?

## Data

[IBM AMLSim](https://github.com/IBM/AMLSim)'s pre-generated `20K_cycle200`
sample — synthetic but with embedded ground-truth laundering cycles, so
detection quality and timing are exactly measurable (no real-world labels
needed, no entity-resolution mess).

| | |
|---|---|
| Nodes (accounts) | 20,000 |
| Transactions (edges) | 117,805 |
| Fraud nodes | 945 (4.7%) |
| Embedded cycle patterns | 200 |
| Time steps | 149 |

The graph is **homogeneous** (account → account transfers only). The
original design called for a heterogeneous schema (Account / Company /
Individual with `SHARES_DIRECTOR` / `OWNED_BY` edges), but AMLSim only
produces that structure by running its Java/MASON simulator with custom
account parameter files — this project uses the ready-made sample data
instead to get a working, honestly-evaluated pipeline first. See
[Future work](#future-work).

## Models

### 1. XGBoost baseline
Tabular features + hand-engineered graph aggregations (in/out-degree,
transaction volume, tx count, active steps) — no multi-hop message passing.
`src/training/train_baseline.py`

### 2. GraphSAGE GNN
10-layer GraphSAGE with residual connections and JumpingKnowledge
concatenation. `src/models/gnn.py`

The depth isn't arbitrary: enumerating cycles in the fraud-node subgraph
showed most embedded laundering cycles span **~10-12 hops**, not the 3-4
node rings a shallow GNN would be tuned for. A plain 3-layer GNN can't see
past 3 hops, so it structurally cannot represent "this account sits on a
long cycle" — it just re-derives a smoothed version of the same
volume/degree signal XGBoost already has. Residual connections + JK let the
model go deep enough to match the actual cycle length without the
oversmoothing that plain deep GCN/SAGE stacks suffer from.

### 3. Temporal Graph Network (TGN)
Memory module + temporal attention embedding (`torch_geometric`'s reference
TGN recipe), trained via a link-prediction pretext task with a fraud
classification head riding on the same causal embeddings.
`src/models/tgn.py`, `src/training/train_tgn.py`

Why add this on top of the GNN: the snapshot-based evaluation of the static
GNN has a real confound — its `BatchNorm` running statistics are calibrated
on the final, densest graph, then misapplied to much sparser early snapshots
during time-to-detection scoring. TGN's memory updates continuously per
transaction, so there's no snapshot to mismatch — the memory state at step
*t* **is** the as-of-*t* representation, by construction.

## Results

| Model | ROC-AUC | PR-AUC | Recall@100 | Recall@200 |
|---|---|---|---|---|
| XGBoost baseline | 0.884 | 0.635 | 0.471 | 0.593 |
| GraphSAGE GNN (10-layer) | 0.897 | **0.735** | **0.529** | **0.677** |
| TGN | *pending full GPU training* | | | |

The GNN clearly wins on final detection quality once it's given enough
receptive field to actually see the cycles — but that's the easier half of
the claim.

### Time-to-detection: the honest part

Both models are frozen and re-scored on cumulative transaction snapshots
(only past edges included), then compared against AMLSim's ground-truth
`fraudStep` — the step at which a node actually entered a laundering
pattern. This directly tests "does the model flag it earlier?"

```
gnn_median_lag_steps:          4.0
baseline_median_lag_steps:     3.0
gnn_faster_than_baseline_pct:  21.6%
```

**The GNN did not detect earlier than the tabular baseline** — despite
winning decisively on final accuracy. This survived one iteration (going
from 3 to 10 layers improved median lag from 5→4 steps, not enough to flip
the result) and pointed to a second, distinct confound: the BatchNorm
distribution-shift problem described above. TGN was built specifically to
remove that confound by evaluating online instead of via re-scored
snapshots; that comparison is pending the full GPU training run (see
`results/time_to_detection.json` for the latest numbers after re-running
`src/training/time_to_detection.py`).

This is reported as a finding, not hidden: a GNN with more model capacity
and a longer receptive field is not automatically faster to react, and
proving "earlier detection" requires ruling out training/evaluation
artifacts (receptive field, normalization) before it's a real result — not
just picking the model with the better headline PR-AUC.

## Explainability

`src/explain/explain.py` runs GNNExplainer on high-confidence true-positive
fraud predictions, extracting the exact subgraph and edge weights the
model's decision was actually driven by — the operational payoff of using a
GNN over XGBoost: a tabular model can say "this account is suspicious", a
GNN explanation can show *which transfer ring*. Output: `results/explain_node_<id>.png`
per explained node, plus `results/explanations.json` with edge importances.

Example: node 3209 (model score 1.000) — its top-25-edge explanatory
subgraph pulls in 49 other accounts, 6 of them independently fraud-labeled.
That's the signal a tabular model structurally can't produce: not just "this
node is suspicious" but "here are the specific transfers and the other
accounts implicated with it." Note the 10-layer receptive field means the
subgraph spans the node's full multi-hop neighborhood, not a single tight
visual ring — GNNExplainer is reporting the edges the *prediction* actually
depended on, which is a broader (and more honest) signal than a hand-picked
cycle would be.

## Repo layout

```
src/
  data_pipeline/
    download_amlsim.py    # pulls the pre-generated AMLSim sample (no Java sim needed)
    build_graph.py         # static/cumulative-snapshot PyG Data objects + stratified split
    temporal_data.py       # chronological TemporalData stream for TGN
  models/
    gnn.py                 # FraudGNN: residual + JumpingKnowledge GraphSAGE
    tgn.py                 # TGNFraudDetector: memory + attention embedding + two heads
  training/
    train_baseline.py      # XGBoost
    train_gnn.py            # GraphSAGE GNN
    train_tgn.py             # TGN (link-prediction pretext + classification)
    time_to_detection.py     # 3-way detection-lag comparison + plot
    metrics.py                # ROC-AUC / PR-AUC / Recall@K
  explain/
    explain.py               # GNNExplainer subgraph extraction
notebooks/
  kaggle_train.ipynb         # clones this repo, runs full pipeline on free Kaggle GPU
```

## Running it

**Local (CPU, for dev/smoke-testing):**
```bash
python -m venv venv && source venv/Scripts/activate   # or venv/bin/activate on Linux/Mac
pip install -r requirements.txt

python -m src.data_pipeline.download_amlsim --dataset 20K_cycle200
python -m src.training.train_baseline
python -m src.training.train_gnn
python -m src.training.train_tgn      # slow on CPU -- see notebooks/kaggle_train.ipynb for GPU
python -m src.training.time_to_detection
python -m src.explain.explain
```

**Full GPU training (free):** open `notebooks/kaggle_train.ipynb` on
[Kaggle](https://kaggle.com) (Settings → Accelerator: GPU, Internet: On) —
it clones this repo and runs the same pipeline with more epochs.

## Future work

- **Heterogeneous schema.** Run AMLSim's actual Java/MASON simulator with
  custom account parameter files to get real `Company` / `Individual` /
  `SHARES_DIRECTOR` / `OWNED_BY` structure, and swap `SAGEConv` for
  `RGCNConv` + edge types (the training loop doesn't need to change).
- **TGN vs. snapshot-GNN time-to-detection**, once full GPU training
  finishes — the actual test this project was built to answer.
- **Closed-walk / cycle-count features** (sparse adjacency matrix powers)
  as an explicit structural signal, to isolate whether remaining
  time-to-detection gaps are an information problem or a learning-capacity
  problem.
