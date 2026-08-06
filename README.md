# AML-TGN: Graph-Based Money-Siphoning & Shell-Ring Detection

Detects money-laundering rings hidden in transaction data by comparing three
approaches — a standard machine-learning baseline, a Graph Neural Network,
and a Temporal Graph Network — and rigorously tests a specific claim:
**do models that understand network structure catch laundering rings earlier
than models that only look at transaction volume?**

## The short version

- **Best accuracy**: a Graph Neural Network (GNN), enhanced with an explicit
  "closed-loop" structural feature, correctly identifies fraud accounts far
  better than a standard tabular model (XGBoost) — a ~22% improvement in
  detection quality.
- **Best speed**: that same structural feature, fed to the simpler tabular
  model instead, flags fraud rings *the moment they form* (0-step delay),
  faster than the GNN and faster than a more complex Temporal GNN built
  specifically to solve this.
- **The honest surprise**: bigger, more sophisticated models did not win on
  every dimension. The right answer wasn't "use the fanciest model" — it was
  "test everything, measure the actual tradeoff, and use two models for two
  different jobs." That production design (below) is the real deliverable.

## The problem, in plain terms

A money-laundering ring works by moving funds through a chain of shell
accounts back to where they started — account A pays B, B pays C, C pays D,
D pays back to A. Standard fraud tools flag accounts by looking at how much
money moves through them or how often — but that signal only builds up
*after* a ring has been operating for a while. The shape of the ring itself
(a closed loop of transfers) is visible in the data earlier, if a model
knows how to look for it. This project tests whether that's actually true.

## The data

[IBM AMLSim](https://github.com/IBM/AMLSim)'s pre-generated `20K_cycle200`
sample: synthetic transaction data with 200 laundering rings deliberately
built into it, so we know exactly which accounts are fraudulent and exactly
when each one joined a ring — which makes it possible to measure detection
*speed*, not just accuracy.

| | |
|---|---|
| Accounts | 20,000 |
| Transactions | 117,805 |
| Fraud accounts | 945 (4.7%) |
| Laundering rings | 200 |
| Time steps | 149 |

## What we tried

| Model | What it is |
|---|---|
| **XGBoost baseline** | Standard tabular ML on hand-built features: how much money moved through an account, how often, to/from how many other accounts. |
| **GraphSAGE GNN** | A 10-layer graph neural network that learns account risk by passing information along the transaction network itself. Depth was matched to the actual ring size found in the data (~10-12 hops), not picked arbitrarily. |
| **Temporal GNN (TGN)** | An online model that updates its understanding of each account continuously as transactions arrive, instead of only seeing periodic snapshots. |
| **XGBoost / GNN + closed-loop feature** | The baseline and GNN, each additionally given one explicit signal: a mathematical count of how many short closed loops of transfers pass through each account — a direct, computable stand-in for "is this account part of a ring." |

## Results

### Accuracy — how well each model tells fraud from normal accounts

| Model | PR-AUC (higher is better) |
|---|---|
| XGBoost baseline | 0.641 |
| XGBoost + closed-loop feature | 0.685 |
| Temporal GNN (TGN) | 0.689 |
| GraphSAGE GNN | 0.729 |
| **GraphSAGE GNN + closed-loop feature** | **0.781** |

The GNN wins decisively once given the right signal to work with — a ~22%
improvement in detection quality over the plain baseline.

### Speed — how fast each model catches a ring, relative to when it actually forms

Measured by replaying the transaction history and asking each model, "at
what point would you have flagged this account?" — then comparing that to
the moment AMLSim's ground truth says the ring actually started.

| Model | % of rings caught | Typical delay |
|---|---|---|
| GraphSAGE GNN | 83.2% | 2 steps late |
| Temporal GNN (TGN) | 86.0% | 2 steps late |
| GraphSAGE GNN + closed-loop feature | 83.6% | 4 steps late |
| XGBoost baseline | 86.6% | 1 step late |
| **XGBoost + closed-loop feature** | **87.0%** | **0 steps — flags it as it forms** |

This is the honest, non-obvious finding: **the same structural feature that
made the GNN more accurate did not make it faster.** The simpler model,
given the same information, reacted to it immediately; the GNN — which
aggregates information smoothed across many transactions and hops — got
better at long-run accuracy but slower to react to a still-forming pattern.

### The real tradeoff: precision vs. speed

Flagging more accounts catches more fraud but also flags more innocent ones
— every automated system has to pick an operating point. We measured this
directly instead of assuming it:

| Flag rate | Model | Of flagged accounts, % actually fraud | % of rings caught | Delay |
|---|---|---|---|---|
| 1% | XGBoost + feature | **100%** | 70% | 2 steps late |
| 1% | GNN + feature | **100%** | 64% | 6 steps late |
| **2%** | **XGBoost + feature** | **95%** | **87%** | **0 steps** |
| 2% | GNN + feature | 100% | 84% | 4 steps late |
| 5% | XGBoost + feature | 61% | 99% | 6 steps early* |
| 10%+ | either | 19-39% | ~100% | very early* |

*Past ~5%, so many accounts get flagged that "catches early" is misleading
— you're not detecting smarter, you're just casting a much wider net and
accepting a lot more false alarms (at 10%+, roughly 2 out of 3 flagged
accounts are innocent). That band is only usable for a human review queue,
never for automated action.

## Production recommendation: a two-tier system

This is where the results actually point, and it's a standard pattern in
real fraud systems for exactly this reason:

**Tier 1 — instant gatekeeper: XGBoost + closed-loop feature, at a 2% flag
rate.** Sits directly on the live transaction stream. 95% precision means
very few legitimate transactions get caught up in it; 0-step delay means it
can hold or step-up-authenticate a transaction *before the ring is even
fully aware it's been caught*. Use the tighter 1% setting (100% precision,
but only 70% of rings caught) if the business needs zero false positives on
automated blocking.

**Tier 2 — background deep scan: GraphSAGE GNN + closed-loop feature, run
asynchronously.** Doesn't need to block anything in real time, so it can run
looser (10-20% flag rate, 71-84% of rings found) and use its superior
accuracy to map out the *entire* ring — other shell accounts, other
connected transfers — once Tier 1 has already paused the live transaction.

Neither model alone is the right answer. The fast one misses ~13% of rings
entirely within the observation window; the accurate one is too slow to
stop money leaving an account in real time. Together, they cover both
failure modes.

## How we got here (why this is trustworthy, not cherry-picked)

The headline finding — "the fancy model didn't automatically win" — only
means something if you can see it wasn't the result of giving up early or
tuning until a convenient number appeared. Short version of the actual path:

1. Built a plain 3-layer GNN — it did *not* detect rings earlier than the
   tabular baseline, despite being far more complex.
2. Diagnosed why: laundering rings in this data span ~10-12 hops, and a
   3-layer GNN can only see 3 hops. Deepened it to 10 layers with residual
   connections — final accuracy improved, but detection speed barely moved.
3. Diagnosed a second issue: the GNN's normalization layers were calibrated
   on the full, dense final graph, then misapplied when scoring the sparse
   early snapshots used for speed testing — a real confound, not just an
   excuse.
4. Built a Temporal GNN specifically to eliminate that confound (it updates
   continuously instead of using snapshots). It still didn't beat the
   baseline on speed.
5. Ran the cheapest possible test to find out why: did the *timing signal
   even exist* in the data, independent of any model's ability to learn it?
   Handed one explicit structural feature directly to XGBoost — no learning
   required. It worked immediately (0-step delay), which proved the signal
   was there all along and the deep models simply weren't extracting it
   efficiently.
6. Fed the same feature to the GNN. Fixed a real bug along the way (the
   feature's numbers were too skewed for a neural network — a few accounts
   had raw values in the thousands while most were zero — which crashed
   training; a standard log-transform fixed it). Final result: the GNN's
   *accuracy* jumped to the best of any model tested, but its *speed* still
   didn't improve — confirming this is a genuine architectural property, not
   a bug.
7. Swept the flagging threshold to turn "GNN is slower" from an impression
   into an exact, deployable number.

## Explainability

For a flagged account, `src/explain/explain.py` (GNNExplainer) extracts the
*specific* subgraph the model's decision was based on — not just "this
account is suspicious" but "here are the other accounts and transfers
implicated with it." Example: node 3209 (flagged with 100% confidence) has
an explanatory subgraph of 49 other accounts, 6 of them independently
labeled as fraud — exactly the kind of output a tabular model cannot
produce, no matter how good its features are. See `results/explain_node_*.png`.

## Repo layout

```
src/
  data_pipeline/
    download_amlsim.py     # pulls the pre-generated AMLSim sample (no Java sim needed)
    build_graph.py          # static/cumulative-snapshot graphs + stratified split
    temporal_data.py        # chronological event stream for TGN
    cycle_features.py       # the closed-loop structural feature
  models/
    gnn.py                  # GraphSAGE GNN (residual + JumpingKnowledge)
    tgn.py                  # TGN: memory + temporal attention + two heads
  training/
    train_baseline.py               # XGBoost
    train_gnn.py                     # GraphSAGE GNN
    train_tgn.py                      # TGN
    cycle_feature_diagnostic.py        # XGBoost + closed-loop feature
    gnn_cycle_feature_diagnostic.py     # GNN + closed-loop feature
    threshold_sweep.py                   # precision vs. speed tradeoff curve
    time_to_detection.py                  # 3-way detection-speed comparison + plot
    metrics.py                             # ROC-AUC / PR-AUC / Recall@K
  explain/
    explain.py               # GNNExplainer subgraph extraction
notebooks/
  kaggle_train.ipynb         # clones this repo, runs the full pipeline on free Kaggle GPU
```

## Running it

**Local (CPU, for dev/smoke-testing):**
```bash
python -m venv venv && source venv/Scripts/activate   # or venv/bin/activate on Linux/Mac
pip install -r requirements.txt

python -m src.data_pipeline.download_amlsim --dataset 20K_cycle200
python -m src.training.train_baseline
python -m src.training.train_gnn
python -m src.training.train_tgn                      # slow on CPU -- see the Kaggle notebook for GPU
python -m src.training.cycle_feature_diagnostic
python -m src.training.gnn_cycle_feature_diagnostic
python -m src.training.threshold_sweep
python -m src.training.time_to_detection
python -m src.explain.explain
```

**Full GPU training (free):** open `notebooks/kaggle_train.ipynb` on
[Kaggle](https://kaggle.com) (Settings → Accelerator: **GPU T4 x2**,
Internet: On — avoid P100, see the notebook's first cell for why) — it
clones this repo and runs the same pipeline.

## Future work

- **Heterogeneous schema.** Run AMLSim's actual Java/MASON simulator with
  custom account parameter files to get real Company/Individual/
  shared-director/ownership structure, and swap `SAGEConv` for `RGCNConv` +
  edge types.
- **TGN + closed-loop features**, computed causally as the event stream
  unfolds rather than per static snapshot — skipped here due to the real
  risk of introducing a look-ahead bug, but the natural next test.
- **Adversarial time-spreading.** AMLSim's cycles here form in a short,
  concentrated burst. A launderer deliberately spreading transfers over time
  to evade volume-based detection is a real, documented evasion tactic
  ("structuring") — testing whether the GNN's advantage grows under that
  harder condition would require generating data with genuine temporal
  spread and real per-transaction ground truth (AMLSim's own simulator, not
  post-hoc editing of the existing sample).
