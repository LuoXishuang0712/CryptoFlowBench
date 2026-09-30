# CryptoFlowBench

CryptoFlowBench builds an on-chain incident evidence corpus and Chain-QA
benchmark for evaluating transaction evidence and fund-tracing agents. This
repository publishes the dataset and code for evidence extraction, chain-agent
execution, and evaluation.

- [Project introduction](#project-introduction)
- [Deployment guide](#deployment-guide)
  - [Environment setup and data import](#1-environment-setup-and-data-import)
  - [Dataset extraction workflow](#2-typical-dataset-extraction-workflow)
  - [Chain agent workflow](#3-chain-agent-workflow)
  - [Configuration](#4-configuration)
  - [Development and maintenance](#5-development-and-maintenance-commands)
  - [Validation](#6-validation)

## Project Introduction

The repository contains four connected capabilities:

1. Collect public incident reports and extract structured incident evidence.
2. Generate report/KGQA questions and index-grounded Chain-QA tasks.
3. Run a chain agent and evaluate transaction-existence and edge-action tasks.
4. Run deterministic comparison experiments with graph policies and learned
   edge/node classifiers.

`selected_cases.txt` defines the canonical case set for batch processing.
The published dataset manifest identifies `chain_qa.v3_2` as the snapshot for
22 selected cases. The generator defaults to `chain_qa.v3_1`; use
`--dataset-version` to select a separate version for new extraction runs.

### Pipeline

1. `collect_report.py` collects report metadata into `cases/<case-id>/case.json`.
2. `extract_report_info.py` summarizes incident facts into `summarized/<case-id>/summary.json`.
3. `chain_evidence.py` converts summaries into `summarized/<case-id>/chain_evidence.json`.
4. `extract_qa.py` creates KGQA-style test questions in `qa/<case-id>/`.
5. `python -m extract_qa_chain` builds chain tracing QA tasks and writes them to MongoDB.
6. `agent_chain.py` evaluates chain QA tasks and writes every agent session plus result back to MongoDB.

`summary.json` captures incident-level facts such as overview, loss, root cause, attacker, victim contracts, and fund flow. `chain_evidence.json` captures entities, edges, transaction evidence, sources, scanner evidence, open questions, and negative constraints. Service entities use an `addresses` list for auditable `(chain, address)` attribution, while `entity_address_matching` reports mapped/unmapped entities, observed-edge matches, conflicts, invalid mappings, and missing evidence.

### QA Test Set

`extract_qa.py` writes test artifacts:

- `qa/<case-id>/qa.jsonl`: one QA record per line.
- `qa/<case-id>/qa.json`: the same records as a JSON array.
- `qa/<case-id>/manifest.json`: counts and file metadata.

QA records include:

```json
{
  "question_type": "<s,p,*>",
  "subject": "edge:example",
  "predicate": "verification_status",
  "object": "*",
  "question": "...",
  "answer": "...",
  "supporting_triples": [],
  "reasoning_type": "edge_metadata_lookup",
  "difficulty": "easy"
}
```

The QA rows are evaluation targets, not retrieval documents for the agent.

### Chain QA and Agent Sessions

The path-oriented chain QA pipeline lives in `extract_qa_chain/` and writes purpose-built Mongo collections:

- `chain_subgraph_bundles`
- `chain_seed_nodes`
- `chain_subgraph_edges`
- `chain_labels`
- `chain_candidate_sets`
- `chain_qa_tasks`

Local files under `chain_qa/<case-id>/` are backup/export artifacts. The chain agent reads tasks from Mongo first and falls back to local `chain_qa/` files only for smoke tests.

The generator/agent interface is defined in [`QA_AGENT_RULES.md`](QA_AGENT_RULES.md).
The generator supports four task types:

- `direct_transaction_existence` (DTE): query-only transaction verification
  with deterministic answer and tool-protocol scoring.
- `edge_action_classification` (EAC): one-step classification over public
  candidate edges with deterministic scoring.
- `path_completion` (PC): multi-step teacher-forced and free-rollout policies.
- `subgraph_noise_filtering` (SNF): one policy over public states, with
  oracle-seed and predicted-seed rollouts evaluated separately.

Tool-grounded tracing tasks require `eth_edge(with_raw=true)` evidence for one
raw edge in every public candidate group. Coverage, tool parameters, errors,
duplication, and budgets are scored deterministically. Public task context is
scrubbed of oracle fields before being sent to the agent or stored in sessions.

`--tracing-agent-profile context_only` generates a no-tool ablation and should
use a distinct dataset version. For evaluation, `--agent-profile context_only`
can project tool-grounded EAC/PC/SNF tasks in memory if the filtered snapshot
has no native context-only rows. The projection disables tools, supplies
sanitized public edge context, records provenance, and does not modify stored
tasks. DTE never uses this fallback.

The chain agent can answer all four task types. Its self-contained CLI
scoring paths cover DTE and EAC. PC/SNF CLI evaluation additionally requires a
supplementary LLM judge dependency that is not bundled, so the deployment
examples below evaluate only DTE/EAC. The comparison framework provides
separate deterministic PC/SNF evaluation.

`--skip-dte` omits transaction-existence generation and its oracle queries.
Use a new dataset version when preserving an existing snapshot: rerunning the
same `{case_id, dataset_version}` replaces that Mongo snapshot. The local
`chain_qa/<case-id>/qa_tasks.jsonl` file is a latest-generation backup and is
overwritten by a newer generation run.

Detector tools default to LR, RF, and XGB and load case-specific frozen
checkpoints from `detector_tool/`. SNF also uses the shared
`xgboost_node_1pct` seed detector. Model checkpoints must be generated before
using these tools; installing the `baselines` extra only installs dependencies.
Pass `--no-detector-tools` to run without checkpoints, or repeat
`--detector-tool-method` to choose trained methods.

`agent_chain.py eval` writes:

- `chain_agent_sessions`: one full agent session per task, including prompt messages, retrieved public graph context, answer, errors, and run metadata.
- `chain_agent_eval_results`: one judged result per `{run_id, task_id}`.
- `chain_agent_eval_summaries`: aggregate run-level summary.

Evaluation results retain model metadata and separate agent/judge token-usage
fields. Run summaries aggregate `input_tokens`, `output_tokens`, and
`total_tokens` across workers. DTE and EAC use deterministic scorers and make
no judge LLM calls.

For edge-action tasks, stored judgements include deterministic `hit@k`, action accuracy/macro-F1, over-expansion, none-of-above accuracy, MRR, and Brier score. Transaction-existence tasks use deterministic JSON parsing and report task correctness separately from tool-protocol correctness. Metrics include existence/count accuracy, edge-id and tx-hash precision/recall, required-tool success, query-parameter accuracy, tool errors, unnecessary calls, and tool-result groundedness. Overall pass requires both the task answer and tool protocol to pass; these tasks never use the LLM judge.

When `--dataset-version` is omitted, `agent_chain.py` selects one latest version
for the case from MongoDB (or from the local backup fallback) and never mixes
versions. Public task context is recursively scrubbed of oracle fields before it
is sent to the agent or persisted in the session. Edge-action thresholds and
top-k decisions are read from public `graph_context`; malformed structured
answers and malformed gold candidate sets fail deterministic evaluation without
an LLM judge call. Agent-generation failures are stored as run errors and also
skip the judge call.

Use `--strict-mongo` when Mongo write failures should fail the run instead of being recorded as warnings.

### Comparison Experiments

`comp_exam/impl/` includes the comparison runner, public graph features,
deterministic scoring, and checkpoint persistence:

- Graph policies: Poison, Haircut, FIFO, LIFO, and TIHO.
- Learned edge classifiers: Logistic Regression, Random Forest, MLP, GCN,
  and XGBoost.
- Paper-derived adapters: I2BGNN, PEAE-GNN, TokenScout, DenseFlow, and
  DenseFlow+. These adapt the methods to the benchmark interface.
- Node classifiers: Logistic Regression, Random Forest, XGBoost, and the
  shared `xgboost_node_1pct` detector. `max_outflow_source` is available as a
  heuristic smoke baseline.

The runner loads tasks from MongoDB and stores run summaries, per-task results,
and intermediate stages in `comparison_experiment_runs`,
`comparison_experiment_task_results`, and `comparison_experiment_stage_results`.
Learned methods require training cases disjoint from the target case. Runs
persist model checkpoints and manifests under
`detector_tool/<case-id>/<role>-<method>/` by default.

## Deployment Guide

This bundle contains the published Chain-QA dataset, extraction tools,
chain agent, and comparison framework. Published data is stored under
`published_data/`:

- `case_reports.jsonl`: `case.json` and the original `runN.json` reports for
  the 22 selected cases.
- `case_summaries.jsonl`: one `summary.json` payload per case.
- `case_graphs.jsonl`: one `chain_evidence.json` payload per case.
- `mongo/*.jsonl`: the six dataset collections for exactly one Chain-QA
  dataset version.
- `manifest.json`: the selected version and cases, row counts, file sizes, and
  SHA-256 hashes.

The data archive can be acquired from [GoogleDrive](https://drive.google.com/drive/folders/1qGfLKYWptQ0wL5R2wwuwBKLXrtcLtctL?usp=sharing) or [BaiduNetdisk](https://pan.baidu.com/s/1go4Q7iKdAB_nBHa2hr5fgg?pwd=euad).

### 1. Environment setup and data import

The commands below assume a Linux-based system with Python 3.12, `uv`, and a
running MongoDB instance. The Ethereum index and agent browser are external
services; configure their endpoints after deploying them separately.

```bash
cp .env.package.example .env
uv sync
```

Install the optional baseline dependencies if the chain agent should use the
LR, RF, and XGB detector implementations:

```bash
uv sync --extra baselines
```

Edit `.env` and verify the MongoDB settings, agent LLM settings,
`ETH_INDEX_URL`, and `BROWSER_BASE`.

Validate every published file against the manifest without writing to MongoDB:

```bash
uv run python scripts/import_publish_dataset.py published_data
```

An empty target database is recommended. Importing uses deterministic-key,
idempotent upserts and does not delete unrelated data from the database:

```bash
uv run python scripts/import_publish_dataset.py published_data --apply
```

To continue report summarization, evidence conversion, or dataset extraction,
materialize the three local JSONL files back to `cases/` and `summarized/`.
Existing files are not overwritten by default:

```bash
uv run python scripts/import_publish_dataset.py published_data --materialize-local
```

Use `--force-local` only when replacing existing local JSON artifacts is
intentional. Verify the imported Chain-QA tasks afterward:

```bash
uv run python agent_chain.py list 01-2022-Ronin-2022 --limit 2
```

### 2. Typical dataset extraction workflow

After materializing the local JSON files, the pipeline can continue from any
stage. The following example uses the Ronin case:

```bash
# Collect the original reports.
uv run python collect_report.py "Ronin Bridge Hack 2022"

# Extract a structured summary from cases/<case-id>/.
uv run python extract_report_info.py 01-2022-Ronin-2022

# Convert the summary into the case graph / chain evidence.
uv run python chain_evidence.py 01-2022-Ronin-2022

# Optionally generate the report/KGQA dataset.
uv run python extract_qa.py 01-2022-Ronin-2022

# Generate index-grounded Chain-QA data in MongoDB and local chain_qa/ output.
uv run python -m extract_qa_chain 01-2022-Ronin-2022 \
  --k-hop 2 \
  --neg-ratio 8 \
  --transaction-neg-ratio 1
```

Batch workflows use `selected_cases.txt` as the canonical ordered case list:

```bash
uv run python chain_evidence.py all
uv run python extract_qa.py all
```

The Windows batch runner is described in step 5.

Chain-QA extraction requires a complete Ethereum index. A small local-only
connectivity smoke test can be run without writing to MongoDB:

```bash
uv run python -m extract_qa_chain 01-2022-Ronin-2022 \
  --k-hop 1 \
  --max-nodes 50 \
  --max-edges 50 \
  --neighbor-limit 50 \
  --neg-ratio 5 \
  --transaction-neg-ratio 1 \
  --no-mongo
```

### 3. Chain agent workflow

List available tasks:

```bash
uv run python agent_chain.py list 01-2022-Ronin-2022 --limit 5
```

Run one task, replacing `<task-id>` with an ID from the previous command.
Disable detector tools until model checkpoints are available:

```bash
uv run python agent_chain.py answer 01-2022-Ronin-2022 \
  --task-id <task-id> \
  --no-detector-tools
```

Run a sampled DTE/EAC evaluation. This creates new sessions, per-task results,
and a run summary in MongoDB:

```bash
uv run python agent_chain.py eval 01-2022-Ronin-2022 \
  --dataset-version chain_qa.v3_2 \
  --task-type direct_transaction_existence \
  --task-type edge_action_classification \
  --no-detector-tools \
  -k 20 --seed 0 -p 4 --strict-mongo
```

Run the EAC context-only ablation:

```bash
uv run python agent_chain.py eval 01-2022-Ronin-2022 \
  --dataset-version chain_qa.v3_2 \
  --task-type edge_action_classification \
  --agent-profile context_only --no-detector-tools \
  -k 20 --seed 0 -p 4 --strict-mongo
```

Run a deterministic comparison with the default Poison edge policy and
`max_outflow_source` node heuristic:

```bash
uv run python -m comp_exam.impl 01-2022-Ronin-2022 \
  --dataset-version chain_qa.v3_2 --limit 10
```

To train and save an LR edge checkpoint plus the shared XGBoost node detector,
use disjoint training cases:

```bash
uv run --extra baselines python -m comp_exam.impl 01-2022-Ronin-2022 \
  --dataset-version chain_qa.v3_2 \
  --edge-classifier logistic_edge --node-detector xgboost_node_1pct \
  --train-case 02-2022-Wormhole-2022 \
  --train-case 03-2022-Harmony-Horizon-Bridge-2022 \
  --model-output-dir detector_tool --limit 10
```

After that run succeeds, enable the trained method explicitly:

```bash
uv run --extra baselines python agent_chain.py answer 01-2022-Ronin-2022 \
  --dataset-version chain_qa.v3_2 --task-id <task-id> \
  --detector-tool-method LR
```

The `eth_neighbors` and `eth_edge` tools require `ETH_INDEX_URL`. Web tools
require `BROWSER_BASE`. Deployment of these external services is outside this
bundle. See `QA_AGENT_RULES.md` for the task interface and evaluation boundary.

### 4. Configuration

Use `uv` with the checked-in lockfile:

```bash
uv sync
```

Common environment variables live in `.env`:

- `LLM_PROVIDER_URL`, `LLM_NAME`, `LLM_API_KEY`: agent LLM endpoint.
- `LLM_THINK`, `LLM_OUTPUT_LENGTH`: agent generation controls.
- `BROWSER_BASE`: optional browser service base URL for browser/Etherscan tools.
- `ETH_INDEX_URL`: optional local Ethereum index HTTP base URL.
- `MONGO_URL`, `MONGO_USER`, `MONGO_PASSWD`, `MONGO_DB`, `MONGO_AUTH_SOURCE`: MongoDB storage.
- `CHAIN_QA_TRANSACTION_NEG_RATIO`: negative-to-positive ratio for verified transaction-existence tasks, default `1`.
- `CHAIN_QA_TRANSACTION_QUERY_LIMIT`: completeness cap for transaction-existence neighbor queries, default `100000`; capped queries generate no oracle tasks.
- `CHAIN_QA_CONNECT_TIMEOUT`: Ethereum-index TCP connect timeout in seconds, default `5`.
- `CHAIN_QA_TIMEOUT`: Ethereum-index response/read timeout in seconds, default `60`.
- `CHAIN_QA_REQUEST_RETRIES`: retries for transient connect/read timeouts, default `1`.
- `CHAIN_QA_RETRY_BACKOFF`: initial retry backoff in seconds, default `1.0`.
- `CHAIN_QA_SHOW_PROGRESS`: enable the per-case transaction-oracle progress bar, default `true`.

Transaction-existence generation displays the current `(src, edge_type)`, completed
query count, most recent latency, and returned row count. Queries taking at least five
seconds are also written as explicit slow-query diagnostics. Each case manifest records
phase timings under `timings_seconds`; transaction validation additionally records total,
average, maximum, and per-query slow-query timings. Use `--no-progress` for redirected or
machine-readable batch logs. `all` mode prints a final per-case and total elapsed-time
summary, including the active case when the run is interrupted with Ctrl+C.

For upstream deadlock diagnosis, a shorter bounded run can be started with
`--request-timeout 30 --request-retries 0`; the progress bar and slow-query records then
identify the active address without changing transaction sampling or oracle semantics.
An explicit index response of `404 {"detail":"address not indexed"}` is treated as an
empty neighbor set during subgraph expansion and action-candidate enrichment, so empty or
zero-activity addresses do not fail a case. The same response is incomplete evidence for
transaction-existence QA and therefore skips affected oracle tasks instead of being used
as proof that no transaction exists. Other 404 responses remain fatal index errors.

`POST /expand` must return an integer `hop` on every edge, bounded by `1..k`. The chain-QA
client treats this field as authoritative and does not reconstruct missing hops. Missing or
out-of-range values raise an index contract error before hydration. Hydration preserves
the traversal hop even when `/edge/{id}` has no hop metadata; action-neighbor enrichment
uses the existing actionable outgoing relation's hop instead of assigning every new edge
to hop 1. Manifests and validation output include `hop_counts`, `missing_hop_count`, and
`invalid_hop_count`.

If a multi-seed expand returns `404 seed address not indexed`, the client removes only the
explicitly named seed, reports it, and retries the same bounded expand with the remaining
seeds. It does not fall back to `/neighbors`. The skipped seeds are marked
`usable_for_tracing=false` and counted as `counts.unindexed_seeds` in the manifest. Other
expand failures remain fatal.

Do not commit real API keys, browser session data, private generated artifacts unless an artifact snapshot is explicitly requested.

### 5. Development and Maintenance Commands

Evidence and QA generation:

```bash
uv run python collect_report.py "Ronin Bridge Hack 2022"
uv run python extract_report_info.py 01-2022-Ronin-2022
uv run python chain_evidence.py 01-2022-Ronin-2022
uv run python chain_evidence.py all
uv run python extract_qa.py 01-2022-Ronin-2022
uv run python extract_qa.py all
uv run python -m extract_qa_chain 01-2022-Ronin-2022
```

Report collection runs, final `case.json` manifests, summaries, and
`chain_evidence.json` are written to MongoDB through `JsonDocumentStore` first,
then to the same local paths as restorable backups. Import existing artifacts
idempotently with:

```bash
uv run python scripts/import_report_chain_documents.py all --include-runs
uv run python scripts/import_report_chain_documents.py all --include-runs --dry-run
```

Optional LLM-assisted QA expansion:

```bash
uv run python extract_qa.py 01-2022-Ronin-2022 --llm-extra --max-llm-extra 12
```

Batch case processing on Windows (the runner invokes `powershell`):

```bash
uv run python scripts/run_batch_cases.py
```

The batch runner reads `selected_cases.txt` directly and keeps no-overwrite
behavior unless regeneration is explicitly requested.

### 6. Validation

Validate the published archive without database writes, then check that imported
tasks can be listed:

```bash
uv run python scripts/import_publish_dataset.py published_data
uv run python agent_chain.py list 01-2022-Ronin-2022 --limit 2
```

For generated report/KGQA data, inspect `qa/<case-id>/qa.jsonl` and verify that
rows contain non-empty `question`, `answer`, and `supporting_triples` fields.
For chain-agent evaluation, start with the DTE/EAC command in step 3 using
`-k 1` before increasing the sample size and parallelism.
