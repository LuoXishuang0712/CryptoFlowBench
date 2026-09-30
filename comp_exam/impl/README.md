# Comparison Experiment Framework

One experiment run binds one case, one edge classifier, and one node detector.
The runner loads supported QA tasks from MongoDB, dispatches by `task_type`,
persists each intermediate stage, evaluates deterministically, and writes a
run-level summary.

## Implemented algorithms

- Traditional edge policies: `poison`, `haircut`, `fifo`, `lifo`, and `tiho`.
  Haircut/FIFO/LIFO/TIHO are explicitly labelled Ethereum account-relation
  adaptations. They share one visible-outflow taint budget, isolate budgets by
  asset/token, and use timestamp/block ordering for FIFO/LIFO.
- Learned edge classifiers: `logistic_edge`, `random_forest_edge`, `mlp_edge`,
  `gcn_edge`, and `xgboost_edge`. MLP follows Weber et al. (2019) with one
  50-unit hidden layer, Adam at 0.001, and 200 epochs. GCN follows its two-layer,
  100-dimensional, Adam-at-0.001, 1000-epoch setup with weighted cross entropy;
  as an explicit edge-task adaptation, candidate relations are line-graph nodes
  connected when they share an endpoint. All inputs use the same public edge
  feature vector and case-disjoint training split.
- Paper-derived GNN adapters: `i2bgnn_edge`, `peae_gnn_edge`, and
  `tokenscout_edge`. All three use candidate relations as transaction nodes in
  the same line graph as `gcn_edge`. I虏BGNN adds propagated graph max-pool
  context; PEAE-GNN adds public structure/amount/interaction/interval features
  and RTM top-3 context; TokenScout adds temporal encoding, accumulated-flow
  windows, temporal messages, source/destination role fusion, and a
  training-class prototype refinement. These are common-interface adaptations,
  because the papers predict account ego-graph or token-graph labels rather
  than four-class edge actions. TokenScout's asymmetric supervised contrastive
  stage is approximated, not claimed as an exact reproduction.
- Paper-derived graph/path adapters: `denseflow` and `denseflow_plus`. They
  implement dynamic dense peeling and per-asset max-flow on the complete public
  case subgraph, but are explicitly benchmark-interface adaptations rather
  than official reproduction code.
- TRacer: `tracer` runs the bundled MIT-licensed BlockchainSpider
  `TTRRedirect` implementation with the paper's experimental settings
  (alpha 0.15, beta 0.7, epsilon 1e-3, conductance threshold 1e-3). Live API
  expansion is mapped to the benchmark's fixed public case graph and rollout
  node-expansion budget; TTR ranking and token/transaction-hash redirection are
  retained before candidate relations are mapped to follow/ignore.
- Node classifiers: `logistic_node`, `random_forest_node`, and `xgboost_node`
  over the same fixed public node feature vector.
- Primary whole-subgraph node detector: `xgboost_node_1pct`. It trains on one
  row per node from disjoint case subgraphs, labels actionable nodes by the
  presence of an evaluable follow edge, retains all positives, and
  deterministically samples negatives to a 1% training-positive rate.
- Plumbing heuristic: `max_outflow_source` ranks declared candidate seeds by
  largest visible outgoing relation. It remains a smoke baseline, not the
  recommended main comparison node model.

Install learned-baseline dependencies with `uv sync --extra baselines`.

Formal runs persist a loadable checkpoint for both the edge method and shared
node detector under
`detector_tool/<case_id>/<role>-<method>/{model.pkl,manifest.json}`. The manifest
contains the model digest, dataset version, disjoint training cases, and training
summary. `load_detector_checkpoint()` verifies SHA-256 before loading; only load
checkpoints produced by a trusted local experiment because the payload is a
Python pickle.

## Fair comparison protocol

Taint methods are seed-driven edge/value propagation rules; they do not include
an attacker-node classifier. Address profiling in the source paper identifies
high-activity service endpoints and provides a stopping rule, not a malicious
seed detector. The benchmark therefore keeps node localization separate:

1. Primary edge-method table: freeze one shared `xgboost_node_1pct` model per
   outer target fold and vary only the edge classifier.
2. Propagation upper bound: report every edge classifier in `oracle_seed` mode.
3. Node sensitivity: rerun the same edge methods with `random_forest_node` and
   `xgboost_node`, reporting seed localization separately.

The earlier `logistic_node` primary table remains a historical result. The
whole-subgraph prevalence pilot selected XGBoost at 1% training prevalence for
the v3.2 primary table. Its probabilities remain uncalibrated; use ranking
metrics as primary and do not treat a fixed 0.5 threshold as deployment tuned.

Learned models require repeated `--train-case` arguments. Target-case aliases
are rejected from training, only public graph fields become features, and the
run stores training case/version/sample/feature metadata in MongoDB. Thresholds
must be fixed from training/validation cases; do not tune them on the target.

## MongoDB collections

- `comparison_experiment_runs`: configuration, status, progress, and aggregate
  metrics.
- `comparison_experiment_task_results`: one final algorithm output and
  evaluation per `{run_id, task_id}`.
- `comparison_experiment_stage_results`: auditable `task_loaded`,
  `node_detection`, `edge_classification`, `path_generation`, `evaluation`, and
  `error` stage payloads.

The source QA and graph collections remain `chain_qa_tasks` and
`chain_subgraph_edges`.

## Usage

```bash
uv run python -m comp_exam.impl 01-2022-Ronin-2022 --limit 10
uv run python -m comp_exam.impl 01-2022-Ronin-2022 `
  --task-type edge_action_classification `
  --task-type path_completion `
  --max-depth 3 --beam-width 3 `
  --max-node-expansions 64 --max-inspected-edges 512

uv run --extra baselines python -m comp_exam.impl 01-2022-Ronin-2022 `
  --edge-classifier fifo --node-detector logistic_node `
  --taint-budget-ratio 0.5 --seed-probability-threshold 0.5 `
  --train-case 02-2022-Wormhole-2022 `
  --train-case 03-2022-Harmony-Horizon-Bridge-2022

uv run --extra baselines python -m comp_exam.impl 01-2022-Ronin-2022 `
  --edge-classifier random_forest_edge `
  --node-detector xgboost_node_1pct `
  --train-case 02-2022-Wormhole-2022 `
  --train-case 03-2022-Harmony-Horizon-Bridge-2022

uv run --extra baselines python -m comp_exam.impl 01-2022-Ronin-2022 `
  --edge-classifier peae_gnn_edge `
  --node-detector xgboost_node_1pct `
  --model-output-dir detector_tool `
  --train-case 02-2022-Wormhole-2022 `
  --train-case 03-2022-Harmony-Horizon-Bridge-2022
```

If `--dataset-version` is omitted, the runner selects only the latest version
available for the requested case and never mixes versions in one run.

Legacy `chain_qa.v3` rows remain valid one-step inputs. `chain_qa.v3_1` PC rows
run teacher-forced and free beam rollouts; seed-hidden SNF rows run oracle-seed
and predicted-seed modes. `path_generation` persists probability-ranked paths,
terminal reasons, and per-mode depth, inspected-edge, and node-expansion usage.
The evaluator reports precision/recall/F1, over-expansion, complete-path success,
seed localization, Brier score, ECE, and budget-normalized progress.
