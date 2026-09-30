# Chain QA and Agent Contract

This document is the interface contract between `extract_qa_chain` and any agent
or model evaluated on the generated chain-tracing tasks. Dataset generators,
agents, parsers, and evaluators must evolve together when this contract changes.

## Scope and Data Boundary

- `extract_qa_chain` generates deterministic benchmark tasks from summarized case
  evidence and the local Ethereum index.
- `chain_qa_tasks` in MongoDB is the primary task store. Files under
  `chain_qa/<case-id>/` are deterministic backup and inspection artifacts.
- An evaluated agent may receive only the task's public `question`,
  `graph_context`, `edge_summaries`, and public graph evidence.
- `answer`, `answer_value`, `labels`, and `evaluation` are oracle fields. They must
  never be included in the agent prompt, retrieval corpus, or tool results.
- Every task declares `agent_profile`, `required_tools`, `allowed_tools`,
  `tool_requirement`, and `public_context_profile`.
- `context_only` tasks receive public graph context and run without tools.
- `tool_grounded` tasks require their declared minimal tools. DTE receives
  query-only context; tool-grounded EAC/PC/SNF may retain the public decision
  state and candidate universe needed to define the task, but must retrieve or
  verify on-chain evidence through declared tools. The agent runner must not
  silently execute a required-tool task tool-free.
- One generated snapshot emits one tracing profile. The default is
  `tool_grounded`; `--tracing-agent-profile context_only` is reserved for a
  separately versioned ablation snapshot so baseline readers never
  double-count paired task rows. Tool dispatch, tool traces, and protocol
  scoring remain agent-side responsibilities.
- For backward-compatible ablation runs, the agent loader may derive an
  in-memory `context_only` view when a run requests only `context_only`, the
  filtered task set contains no native context-only rows, and tool-grounded
  EAC/PC/SNF rows exist. This is an explicit runtime projection, not a persisted
  relabel: remove the tool query contract, set `max_tool_calls=0`, disable all
  tools, expose the stored sanitized public edge context, and record the source
  profile/projection in sessions, results, summaries, and experiment task
  universes. Native context-only rows always take precedence; never fill a
  partially populated native snapshot from tool-grounded rows. DTE is never
  eligible for this fallback.

## Benchmark Hierarchy and Profile Policy

The benchmark has two distinct layers:

1. `direct_transaction_existence` (DTE) is the basic Agent tool-use diagnostic.
   It tests exact query construction, required-tool execution, completeness,
   parsing, and grounding. DTE is not an edge/path tracing comparison task, has
   no traditional baseline methods, and must be reported in a standalone
   capability/gating table rather than the method leaderboard.
2. `edge_action_classification` (EAC), `path_completion` (PC), and
   `subgraph_noise_filtering` (SNF) are the main benchmark tasks. Their primary
   Agent method is tool-grounded and must be compared with the applicable
   baselines on the same deterministic task metrics, candidate universe, case
   split, and inference budget.

The primary **Agent method** for EAC/PC/SNF is tool-grounded. It measures whether
an agent can decide what evidence it needs, invoke declared on-chain tools
correctly, and ground its structured edge/path decisions in tool results. A
required-tool task may not be replayed with pre-supplied evidence and reported
as an equivalent main result.

`context_only` EAC/PC/SNF runs are paired no-tool ablations. They measure how
much performance comes from the supplied context and isolate the contribution
of tool access. They must be labelled as ablations, reported as separate rows or
tables, and never used to fill missing tool-grounded Agent results. Baselines
are fixed-input comparison methods, not Agent profiles; their eligibility and
training rules are defined separately below.

Agent tool-grounded, Agent context-only, and baseline results may appear in the
same per-task comparison table only when method/profile identity is explicit.
The headline Agent row is tool-grounded, while the context-only row is marked as
an ablation. DTE must remain outside those EAC/PC/SNF method rankings. Scores
from DTE, EAC, PC, and SNF must not be collapsed into one undocumented overall
score.

Experiment plans and commands must name both `agent_profile` and `task_type`
explicitly. A main benchmark and a context-only ablation require separate runs,
run-id allowlists, completion checks, and result tables; preferably they also
use separate timestamped experiment records. When both are planned, complete
and validate the tool-grounded main run before interpreting the ablation.

One evaluation run must contain exactly one `agent_profile`. Run identifiers,
Mongo queries, summaries, token totals, and aggregate reports must retain the
profile filter. Historical or smoke-test context-only results must not be used
to fill gaps in a tool-grounded run. Adding a new task type to the primary Agent
benchmark requires an explicit tool-grounded public-context contract, minimal
allowed/required tools, completeness semantics, sanitized tool traces, and
deterministic tool-protocol scoring; merely exposing an existing context-only
task is insufficient.

The current generator defaults EAC/PC/SNF to `tool_grounded` and records
`tracing_agent_profile` in the manifest. Already-persisted snapshots retain the
profile stored in each row until explicitly regenerated; report code must never
relabel historical context-only rows. A context-only ablation must use
`--tracing-agent-profile context_only` with a distinct dataset version or
snapshot when a persistent standalone dataset is required. The runtime
tool-grounded-to-context-only projection above is allowed for paired evaluation
without materializing another snapshot, provided its provenance remains
explicit and result aggregation keys on the effective `context_only` profile.

### Traversal depth

Every indexed subgraph edge must carry `hop_from_seed`, copied from the authoritative
integer `hop` returned by `/expand`. Values must be in `1..k`; missing or out-of-range
values invalidate the dataset bundle. The generator must not infer missing expansion hops
with a client-side BFS. Hydration and candidate enrichment must preserve this traversal
depth so downstream path tasks can distinguish first-hop and later-hop evidence.

## Direct Transaction Existence

`direct_transaction_existence` replaces `direct_link_verification` in
`chain_qa.v3`. It asks only whether an exact directed Ethereum relation exists
under explicit query constraints. It does not ask whether the relation belongs
to the incident path. It is a standalone basic tool-call diagnostic, not a
method-comparison task; do not invent Poison/Haircut/FIFO/LIFO/TIHO or learned
edge baselines for DTE.

Public `graph_context` contains only:

- normalized lowercase `src` and `dst`;
- `chain=ethereum` and `direction=out`;
- explicit `block_min`, `block_max`, and `edge_types`;
- `require_tx_hashes`, response schema, and tool declarations.

The task must not expose matching edges or transaction hashes in
`edge_summaries`, `evidence.chain_edges`, or public graph context. Oracle-only
fields are stored under `answer_value` and `labels`:

```json
{
  "exists": true,
  "transaction_count": 2,
  "matching_edge_count": 2,
  "matching_edge_ids": ["123", "456"],
  "matching_tx_hashes": ["0x..."],
  "verified_absent": false,
  "negative_sample_type": null,
  "verification_query": {}
}
```

Positive and negative oracle values come from `eth_neighbors` with
`direction=out`, the exact block window and edge type, and `with_raw=true`.
Queries request one row beyond `transaction_query_limit`. If that extra row is
returned, the query is considered truncated and no task may be generated from
it. Absence from a sampled k-hop subgraph is never sufficient evidence for a
negative task.

Supported negative types currently include `reversed_direction`,
`wrong_edge_type`, `same_source_non_neighbor`, and `same_case_non_edge`. Their
count is controlled separately by `transaction_neg_ratio`; the existing
`neg_ratio` continues to control candidate composition inside action tasks.

The generated profile is fixed:

```json
{
  "agent_profile": "tool_grounded",
  "required_tools": ["eth_neighbors"],
  "allowed_tools": ["eth_neighbors", "eth_edge"],
  "tool_requirement": "required",
  "public_context_profile": "query_only"
}
```

The generator option `--skip-dte` skips DTE task generation and all DTE oracle
queries. The manifest must record `skip_dte=true`, an empty DTE task count, and
`validation.transaction_existence.skipped=true`. Because Mongo generation
replaces one `{case_id, dataset_version}` snapshot, use this option with a new
dataset version whenever an existing version must retain DTE.

Agent version selection is intentionally snapshot-level, not task-type-level:
without `--dataset-version`, the agent selects one numerically latest version
and does not fill its missing task types from older versions. To evaluate DTE
in the frozen pre-merge experiment source, explicitly pass
`--dataset-version chain_qa.v3_1`. The canonical Mongo snapshot
`chain_qa.v3_2` is a materialized, validated migration containing archived v3.1
DTE plus the frozen v3.1 EAC/PC/SNF rows for all cases in `selected_cases.txt`.
This is not runtime fallback or query-time task mixing: all four task types are
persisted under one version and retain migration provenance. Ad hoc mixing of
different versions inside one run remains prohibited.

The expected agent answer is JSON with `exists`, `transaction_count`,
`matching_edge_ids`, and `matching_tx_hashes`. Parsing, tool-protocol validation,
and answer scoring belong to the tool-grounded agent implementation.

## Tool-Grounded Execution and Evaluation

`agent_chain.py` dispatches tasks by `agent_profile`:

- `context_only` receives sanitized public graph context and `tools=None`.
- `tool_grounded` receives the minimum public context needed to define the
  query/candidate universe and only the task's declared `allowed_tools`.

For `direct_transaction_existence`, the first model turn must use a required
tool choice when supported by the provider. The runner must record every tool
call and result in `tool_trace`, including iteration, tool name, arguments,
status, sanitized result summary, and error. Tool output must recursively remove
case labels, action labels, black-address flags, path identifiers, gold fields,
and all other oracle metadata before it is returned to the model or persisted in
an agent session.

For tool-grounded EAC/PC/SNF, evidence collection and final synthesis are two
runtime phases. Collection accepts only unique `eth_edge(with_raw=true)` calls
whose edge ids belong to the public candidate groups, and rejects duplicate,
out-of-universe, or over-budget calls without dispatching them. Its turn bound
is derived from `tool_query_contract.max_tool_calls` and candidate coverage,
not a fixed four-turn assumption. Once coverage is complete, the budget is
exhausted, a turn makes no progress, or the bound is reached, the runner makes
one final synthesis call with tools disabled. Bounded tool-result summaries may
compact repeated raw rows for context safety, but must retain the query, status,
and enough returned evidence to audit grounding; compaction does not alter
candidate coverage or deterministic scoring.

Tool-grounded EAC/PC/SNF runs may additionally enable runtime-managed
`detector_predict` assistance. This runtime-only tool does not change the persisted QA
snapshot's `allowed_tools` or `required_tools`; it is never available to DTE or
`context_only` tasks. A run declares the enabled method list, defaulting to
`LR`, `RF`, and `XGB`. An empty run-level method list disables the tool.

After `eth_edge` evidence collection closes, the runtime must perform exactly one
detector consultation covering every public decision state and every enabled
edge method, then make the detector result available to the tools-disabled final
synthesis turn. For SNF the same result must include the frozen shared
`xgboost_node_1pct` seed ranking. Each checkpoint must belong to the target case,
match the task dataset version and role, and pass digest/metadata validation.
Predictions remain advisory and cannot satisfy or replace mandatory
`eth_edge(with_raw=true)` evidence. Calls and errors remain in the full session
`tool_trace`; `eth_edge` coverage, budget, and protocol metrics exclude this
auxiliary call. Sessions and run summaries retain the method list and checkpoint
root so runs with different assistance cannot be conflated.

After final synthesis, the runtime may make one tools-disabled structure-only
correction turn when the sole parse failure is inconsistent
`selected_follow_edges`. The correction may update only selected-edge and
derived path/list fields. It is accepted only when every candidate action,
probability, seed prediction, no-valid-seed decision, visited state, and edge id
is byte-for-byte equivalent after parsing; otherwise the original answer remains
the evaluated answer. The attempt and acceptance decision must be persisted.

Persisted raw Ethereum edge ids may use the namespaced form `eth:N`, while the
`eth_edge` tool schema exposes its `edge_id` argument as integer `N`. The runner,
tool dispatcher, duplicate detector, candidate-coverage tracker, and protocol
scorer must therefore canonicalize both representations to the same nonnegative
integer identity before comparison. A provider-emitted `"true"` for the
boolean-only `with_raw=true` contract may likewise be normalized to boolean
`true`. Sessions must retain the submitted arguments when normalization changes
them and use the normalized arguments for dispatch and deterministic scoring.
This compatibility rule changes neither the candidate universe nor the dataset
labels; it prevents serialization differences from being counted as model or
protocol failures.

The transaction answer parser is deterministic and strict:

- `exists` is boolean;
- `transaction_count` is a non-negative integer;
- `matching_edge_ids` and `matching_tx_hashes` are duplicate-free lists;
- `transaction_count` equals the number of unique matching transaction hashes;
- `exists=true` requires matching edge ids and a positive transaction count;
- `exists=false` requires zero count and empty matching-id lists.

The DTE tool protocol fails when a required tool is not called successfully, a tool
is not allowed, the source/direction/block window/edge type changes,
`with_raw=true` is omitted when hashes are required, the neighbor result is
truncated, or the final answer does not exactly match successful tool results.

Transaction-existence evaluation reports task correctness separately from tool
protocol correctness:

```text
overall_pass = task_pass AND tool_protocol_pass
```

Task metrics include existence/count accuracy and edge-id/tx-hash
precision/recall. Tool metrics include tool-use rate, required-tool success,
query-parameter accuracy, unnecessary-tool rate, tool-error rate,
tool-result groundedness, and average tool calls. DTE uses deterministic answer
and tool-protocol scorers. `edge_action_classification` uses deterministic
answer and tool-protocol scorers. Neither task may be routed through an LLM
judge for repair or reinterpretation.
`path_completion` and `subgraph_noise_filtering` also use deterministic task
scorers for their primary outcome, but additionally require the configured
no-tool LLM judge to assess the quality and relevance of the agent-visible
context. For a tool-grounded variant, the judge may also assess a sanitized
recorded tool-use process; a context-only run has no agent tool process to
assess. The judge is
supplementary: it must not repair malformed structured output, infer a missing
trajectory, or override deterministic task correctness.

## Candidate Semantics

`edge_action_classification` is the unified single-hop decision task. It replaces
the former single-edge action task and `single_hop_next_hop_prediction`.

- One task represents one `current_node` decision state.
- Every standard candidate must be a real outgoing relation from `current_node`.
  Cross-node random distractors belong in `subgraph_noise_filtering`.
- The generator may query additional `direction=out` neighbors for actionable
  decision nodes when the initial k-hop expansion does not contain enough local
  alternatives. These neighbors must use `with_raw=true` and the case block window.
- A candidate is aggregated at `(src, dst)` relation level and has a stable
  `pair:<hash>` candidate id.
- Raw Ethereum index edges and transactions remain attached through
  `candidate_edge_groups`, `edge_count`, `tx_count`, `sample_edge_ids`, and
  `sample_tx_hashes`.
- `chain_candidate_sets` may retain the complete candidate-to-raw-edge mapping,
  but public QA task context includes at most 12 raw edge ids and 12 tx hashes per
  candidate so one high-volume pair cannot crowd other candidates out of the prompt.
- If raw edges in one address pair have conflicting action labels, that pair is
  excluded from strict action QA until the conflict is resolved.
- Candidate order is not a relevance signal. Agents must rank by predicted
  follow probability.

## Allowed Actions

Each candidate receives exactly one action:

- `follow`: continue tracing this relation as part of the anomalous fund path.
- `inspect`: evidence is relevant but insufficient for an automatic continuation.
- `stop`: this is a tracing boundary or terminal relation.
- `ignore`: do not include this relation in the anomalous fund path.

Service boundaries must be resolved by exact `(chain, address)` joins against
`chain_evidence.entities[].addresses`; entity names and flow behavior are not
ownership evidence. A mapping records `address_role`, `attribution_status`,
`confidence`, and non-empty evidence. CEX deposit/terminal addresses and mixer
deposit contracts may produce `stop` only when the observed edge hits the mapped
address. A `reported` mapping must additionally match its recorded transaction
hash or index edge id. Bridge contracts, DEX routers, generic service addresses,
and conflicting address attributions produce `inspect`. Unmapped addresses keep
their ordinary graph action and must not become `stop` or `ignore` merely because
an unresolved service entity exists in the report.

`inspect` and `stop` are not positive next-hop predictions. Only `follow` is used
as relevance for top-k ranking.

## Required Agent Output

For `edge_action_classification`, the agent must return JSON only:

```json
{
  "predictions": [
    {
      "edge_id": "pair:<hash>",
      "action": "follow",
      "action_probabilities": {
        "follow": 0.82,
        "inspect": 0.10,
        "stop": 0.03,
        "ignore": 0.05
      }
    }
  ],
  "selected_follow_edges": ["pair:<hash>"]
}
```

The following rules are strict:

- Every candidate appears exactly once in `predictions`.
- No unknown candidate may appear.
- All four probabilities are required, each is within `[0, 1]`, and their sum
  must be within `0.02` of `1.0`.
- `action` must be an action with maximum probability.
- Ranking uses `action_probabilities.follow`, never confidence in the selected
  action.
- `selected_follow_edges` contains exactly the candidates whose action is
  `follow` and whose follow probability meets `graph_context.follow_threshold`.
- An empty `selected_follow_edges` is valid and required when no candidate meets
  the follow decision rule.

Malformed or incomplete structured output fails deterministic evaluation. The
evaluator must not ask an LLM judge to repair or reinterpret it.

## Deterministic Evaluation

The primary edge-action decision is binary:

- If gold follow edges exist, pass when at least one gold follow edge appears in
  the top `graph_context.top_k` candidates ranked by follow probability.
- If no gold follow edge exists, pass only when `selected_follow_edges` is empty.

The evaluator also records:

- `hit@1`, `hit@3`, and `hit@5` when supported by candidate-set size.
- `action_accuracy` and `action_macro_f1`.
- `over_expansion_rate` for incorrectly selected follow edges.
- `none_of_above_accuracy` for all-negative states.
- `mrr` for the first gold follow relation.
- multiclass `brier_score` for confidence calibration.

These detailed metrics must be retained even when downstream systems consume only
the binary pass result. Confidence values without calibration metrics are not a
meaningful benchmark output.

## Sessions, Results, and LLM Usage

Agent sessions and evaluation output are persisted in separate MongoDB
collections:

- `chain_agent_sessions`: public context, complete prompt/tool messages, tool
  trace, structured answer, protocol errors, agent model, and agent token usage;
- `chain_agent_eval_results`: one result per `{run_id, task_id}`;
- `chain_agent_eval_summaries`: one aggregate document per run.

Every result document must contain these top-level model and usage fields:

```json
{
  "agent_llm": "model name from LLM_NAME",
  "judge_llm": "model name from JUDGE_LLM_NAME",
  "agent_token_usage": {
    "input_tokens": 100,
    "output_tokens": 20,
    "total_tokens": 120
  },
  "judge_token_usage": {
    "input_tokens": 80,
    "output_tokens": 10,
    "total_tokens": 90
  },
  "token_usage_complete": true
}
```

Model columns identify the configured models even when a deterministic task does
not invoke the judge. In that case `judge_token_usage` is zero. Agent usage is
the sum of every LLM completion in the task's multi-turn tool loop; judge usage
is counted only when the judge endpoint is actually called. Usage must come from
the provider response and must support both OpenAI-style
`prompt_tokens`/`completion_tokens` and `input_tokens`/`output_tokens` names.

Each run summary must aggregate usage from all completed worker results after
the parallel pool joins. Do not use a shared mutable global token counter across
threads. The required summary shape is:

```json
{
  "agent_llm": "...",
  "judge_llm": "...",
  "token_usage": {
    "agent_llm": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    },
    "judge_llm": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    },
    "all_llms": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    }
  },
  "token_usage_complete": true,
  "token_usage_result_count": 20
}
```

Historical result rows may be backfilled with the current `.env` model names.
Historical token counts must not be estimated: when provider usage was not
stored, use zero-valued usage objects and set `token_usage_complete=false` on
both results and summaries.

Primary Agent reports must additionally verify that every included Agent main
result is `tool_grounded`, every required tool was called successfully, and
required queries were complete. Judge usage is zero for DTE. A tool error,
incomplete query, or required-tool failure is an infrastructure/protocol failure
that must be reported separately; it must not be silently counted as an ordinary
model mistake. Context-only ablation reports must state that tools were disabled
and must not include tool-use metrics as if a tool opportunity existed.

## Task Separation

- `direct_transaction_existence`: always `tool_grounded`; basic Agent tool-use
  diagnostic with no traditional comparison methods. It uses the local Ethereum
  index to verify an exact directed relation under explicit constraints.
- `edge_action_classification`: main task; local outgoing candidate ranking and
  four-action classification. Report the tool-grounded Agent beside baselines;
  report the context-only Agent as an ablation.
- `path_completion`: main task; complete a missing relation after a real positive
  path prefix. It must not be a restatement of single-hop classification.
  Report tool-grounded Agent, baselines, and context-only Agent ablation.
- `subgraph_noise_filtering`: main task; stress-test irrelevant local or random
  subgraphs, including pure-negative samples. Report tool-grounded Agent,
  baselines, and context-only Agent ablation, with node localization separated
  from propagation.

## Sequential Main Tasks and Context-Only Ablation (`chain_qa.v3_1+`)

The sequential task semantics are implemented under `chain_qa.v3_1`. New
generation defaults to tool-grounded EAC/PC/SNF; context-only ablations use the
same public task semantics but must be generated in a distinct snapshot/version.
Existing persisted `chain_qa.v3` and historical context-only v3.1 rows remain
replayable and must retain their recorded profile.

The three main scenarios share one probabilistic edge-policy interface across
tool-grounded Agent, context-only Agent ablation, and compatible baselines while
preserving their different sources of difficulty:

- `edge_action_classification` remains the direct one-step baseline: given a
  fixed `current_node` and its candidates, predict `follow`, `inspect`, `stop`,
  or `ignore` probabilities for every relation.
- `path_completion` becomes a multi-step rollout over the same edge policy.
  Baselines must expose conditional edge probabilities and construct
  probability-ranked paths under the same depth, beam-width, node-expansion,
  and evidence budgets as the agent. Report both teacher-forced next-step
  results and free-rollout results so accumulated decision error is measurable.
- `subgraph_noise_filtering` becomes a two-stage task when no trusted seed is
  supplied: first rank/select candidate seed nodes, then expand paths with the
  shared edge policy. Evaluate seed localization separately from expansion, and
  report both oracle-seed and predicted-seed end-to-end results. The Agent must
  emit one `state_policy` that evaluates every public state exactly once; it
  must never guess or receive hidden oracle seeds. The deterministic evaluator
  applies that same policy from hidden gold seeds for `oracle_seed` and from
  `selected_seed_nodes` for `predicted_seed`. Retain pure-negative/no-valid-seed
  cases.

For persisted v3.1/v3.2 SNF rows whose public `response_schema` still asks the
Agent to return `oracle_seed` and `predicted_seed` directly, the runner must
project only the public response contract in memory to `snf_state_policy_v2`.
The task, candidates, labels, Mongo snapshot, and dataset version remain
unchanged. Sessions must record the projected public context. The scorer rejects
mode-copy answers under the new runtime contract and records evaluator-derived
rollouts in `judgement.parsed_answer` with
`rollout_derivation=evaluator_seed_projection_v1`.

Do not use coverage alone: selecting the entire graph would score well. Pair
path/subgraph coverage with precision, over-expansion, complete-path or terminal
success, and coverage-at-budget. For iterative comparisons, normalize progress
by inspected edges and node expansions rather than raw conversation rounds
alone. The tool-grounded Agent row must additionally report and normalize tool
calls. Evaluate probability quality with proper calibration metrics such as
Brier score or ECE.

For `path_completion` and `subgraph_noise_filtering`, deterministic trajectory,
seed, calibration, stopping, and budget metrics are the primary comparison
outcomes. These same deterministic metrics place the tool-grounded Agent and
baselines in the main EAC/PC/SNF tables; tool-use metrics are additional Agent
diagnostics, not baseline requirements. The no-tool LLM judge remains a separate
evaluation layer for Agent runs. It judges public-context relevance and answer
quality without repairing deterministic output or changing the primary method
ranking. A context-only run supplies no Agent tool trace; a tool-grounded run
may expose only its sanitized trace to the judge. The judge receives no hidden
labels and has no direct tool access. Persist deterministic and judge dimensions
separately and report a clearly named composite only when its weights and
missing-judge behavior are fixed in the run configuration; do not use an
undocumented judge composite to replace the deterministic comparison metrics.

When adapting heterogeneous literature baselines, distinguish faithful edge/path
methods from node- or subgraph-classification methods adapted to the common edge
interface. All methods must receive the same public evidence, candidate universe,
data split, stopping rules, and inference budget; hidden labels must never be
used to construct baseline candidates or tune thresholds on the test set.

For baseline fairness, node localization and edge propagation are separate
components. The primary edge-method comparison freezes one shared, case-disjoint
node classifier; every edge method must also report an oracle-seed propagation
upper bound. Alternative node classifiers are a separately labelled sensitivity
study. Poison/Haircut/FIFO/LIFO/TIHO do not receive method-specific node models.
Learned node/edge classifiers may use hidden labels only from declared training
cases, never from the target case, and may derive features only from public task
fields. Ethereum adaptations of UTXO taint rules must persist their shared taint
budget and isolate amount allocation by asset/token.

For the materialized v3.2 primary baseline table, the shared node classifier is
`xgboost_node_1pct`: one model is trained per outer target fold on whole-subgraph
public node features from the other cases, retains every actionable positive,
and deterministically samples negatives to a 1% training-positive rate. Target
case labels must not be loaded for inference. DenseFlow and DenseFlow+ are
paper-derived common-interface adaptations, not official reproductions; their
maximum-flow capacities must be isolated by asset/token and their `follow`
mapping and frozen underspecified parameters must be disclosed with results.
LR/RF/MLP/GCN/XGBoost edge methods use the same public candidate-edge features
and case-disjoint folds. The Weber et al. MLP/GCN settings are retained where
applicable; GCN must be disclosed as an edge line-graph adaptation because the
paper classifies transaction nodes rather than this benchmark's candidate
relations. The shared node detector remains frozen across these edge methods.
TRacer uses the bundled BlockchainSpider `TTRRedirect` reference implementation
with paper-fixed alpha/beta/epsilon/conductance parameters. Its online
Pop/Expand loop must be adapted only to the already-public fixed case graph and
the common rollout node-expansion budget; its rank/community output may then be
mapped to candidate edge actions. It must share the same frozen node detector
and may not query target labels.

`single_hop_next_hop_prediction` is deprecated and must not be generated in
`chain_qa.v2` or later.

`direct_link_verification` is deprecated and must not be generated in
`chain_qa.v3` or later. Case-path relevance remains the responsibility of action,
path-completion, and noise-filtering tasks.

## Versioning and Migration

- Dataset versions use `chain_qa.v<major>_<minor>`. The legacy spelling
  `chain_qa.v3` is equivalent to `chain_qa.v3_0`.
- Versions with the same major number are backward compatible. A minor release
  may add optional public fields, answer fields, evaluation modes, scorer
  outputs, or metrics, but must preserve the meaning and validity of the
  existing major-version contract.
- `chain_qa.v3_1` extends v3 with sequential PC/SNF modes, budget accounting,
  and layered deterministic-plus-judge evaluation. Existing v3 task types,
  candidate ids, four-action probabilities, empty-selection behavior, and
  single-step records remain valid inputs.
- Newly generated v3.1 EAC/PC/SNF records default to tool-grounded and include a
  public candidate-to-raw-edge query contract. Historical rows retain their
  stored profile. A context-only ablation uses a separate version/snapshot;
  never mix both profiles in one baseline input snapshot.
- Removing or renaming required fields, changing the meaning of an existing
  field, invalidating previously valid answers, or changing hidden/public data
  boundaries requires a new major version.
- A scoring change within one major version must use a new explicit `scorer`
  version and preserve the previous scorer for historical replay. Newly added
  metrics must not silently redefine an existing metric name.
- Task ids are stable only within a generated dataset version and configuration.
- Rerunning a case replaces MongoDB rows for the same `{case_id, dataset_version}`.
- When no version is explicitly requested, the chain agent must select only the
  latest compatible dataset version and must not mix versions in one evaluation.
  Version ordering must parse numeric major/minor components, with `v3` treated
  as `v3_0`; it must not rely on plain lexical string ordering.
- Downstream adapters may transform the JSON representation, but they must
  preserve candidate ids, four action probabilities, empty-selection behavior,
  and deterministic metric semantics.
