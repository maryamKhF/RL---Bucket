# RL + Bucket: MAIN TEST

The round-one entry points are `main1_5.py` (training only) and
`main1_5_resume.py` (evaluation only). The legacy protocol uses `main.py` and
`main-resume.py`.
Round two uses `main2_5.py` (training only) and `main2_5_resume.py` (evaluation only).
Round three uses `main3_5.py` (training only) and `main3_5_resume.py` (evaluation only).
Round four uses `main4_5.py` (training only) and `main4_5_resume.py` (evaluation only).
Round five uses `main5_5.py` (training only) and `main5_5_resume.py` (evaluation only).
GitLab exposes only `run-main5-5` and `run-main5-5-resume` as manual runnable jobs.
Earlier CI definitions are hidden templates; local entry points remain available.
Legacy entry points remain in the project but are not offered as CI jobs.

## Run

### Final continual stage: main5/5

Fifth-stage training has completed in **Job 56791**. Run only
**`run-main5-5-resume`** to evaluate that saved model. It downloads the training
ZIP directly using **`MAIN5_5_RESUME_JOB_ID=56791`**, without depending on a new
training job. Job 56792 stopped during dependency installation, before the
evaluation script started; its log confirms the training artifact downloaded
successfully. Its uploaded ZIP is the input training artifact, not test results.
No training repeat is needed. For a different completed fifth-stage model,
override this variable with that successful training job's ID. CLI
`--resume-*` arguments take precedence; local artifacts remain auto-detected.

Both active CI jobs install dependencies with `Runtime/install_dependencies.py`:
a 120-second socket timeout, five connection retries and up to three complete
installation attempts, with 10/20-second waits only after failure. A persistent
installation failure stops the job before training/evaluation. The installer
preserves the configured package index, TLS verification and requirement
constraints, and does not request upgrades of satisfied packages. See the
[pip options](https://pip.pypa.io/en/stable/cli/pip/) and
[installation behavior](https://pip.pypa.io/en/stable/cli/pip_install/).

`run-main5-5` remains available for intentionally starting a different training
run from fourth-stage Job **56651** (`MAIN5_5_PREVIOUS_JOB_ID`). It retains the
actor, critic, optimizer, cumulative counter and compact lineage of all four
preceding stages. To evaluate that new training run, select its completed Job ID
in `MAIN5_5_RESUME_JOB_ID`. Testing does not train or change the policy.

```bash
python -u main5_5.py --previous-archive /path/to/round_four_artifacts.zip
python -u main5_5_resume.py
python -u main5_5.py --resume-archive /path/to/interrupted_round_five.zip
python -u main5_5_resume.py --resume-job YOUR_COMPLETED_FIFTH_STAGE_JOB_ID
```

Training seed is **46**, evaluation base seed **1046**, and transaction seed
remains **42**. The same 7,000 training / 3,000 held-out split is reused, with
an additional **20,000 requested training transitions** and **18,000 evaluation
payments** (3,000 at each failure rate from **1% through 6%**). Different seeds
change the sampled scenario, not a numerical difficulty level.

The optimized fourth-stage path is reused: **30 workers**, **66 steps per
worker**, **1,980 transitions per frozen-policy rollout**, minibatches of **60**,
`ready-balanced-v1` collection, isolated pickle graph clones and a durable
SQLite WAL/FULL journal. Per-worker contributions stay equal and GAE retains
trajectory order. Existing graph/state selection optimizations, routing search
budgets, recovery limits, sampled XAI and coherent checkpoint boundaries are
preserved. The progress log reports real payment/CPU/search times; `[ROLLOUT]`
reports collection, inference and audit durations. Earlier component speedups
are not a measurement or guarantee of fifth-stage Runner throughput.

The final artifact contains the trained model, fixed inputs, current training
and evaluation XAI traces, `report.txt`, `summary.json`, readable CSV reports
and `reports/main5_5.html`. Feature/critic sensitivity, candidate-cost, training
and recovery heatmaps are exported under `figures/` as **PNG, SVG and CSV**.
Sampled sensitivity is accompanied by full routing audits and sampling coverage;
it is not a causal or SHAP attribution. Simulator recovery snapshots are omitted
from completed policies; older model ZIPs are not recursively nested. After
evaluation and reports finish, the manifest sets `experiment_complete: true`
and `next_round: null`. Before evaluation, completion stays false. There is no
sixth stage.

### Continue the same agent: main4/5

Training uses **`run-main4-5`**. It automatically downloads completed third-stage Job **56650**; override
`MAIN4_5_PREVIOUS_JOB_ID` only to choose another completed third-stage agent.
For the already completed fourth-stage training, run only **`run-main4-5-resume`**.
It downloads artifacts directly from successful fourth-stage Job **56651**,
selected by `MAIN4_5_RESUME_JOB_ID`. It has no dependency on the newest training
attempt, so a canceled retry does not block testing or change its source.
For a different completed fourth-stage run, override this variable with that
job's ID. Explicit `--resume-job`, `--resume-run` and `--resume-archive` take
precedence. The selected archive must contain a valid training completion
marker and matching model/reference checksums; testing never starts training.
The actor, critic, optimizer and cumulative counter continue from stage three.
Training seed is **45**, evaluation base seed is **1045**, and transaction seed
remains **42**. Evaluation keeps six failure rates **1%–6%**, 3,000 held-out
payments at each rate, the CSV reports, sampled explanations and heatmaps.

```bash
python -u main4_5.py --previous-archive /path/to/round_three_artifacts.zip
python -u main4_5_resume.py
python -u main4_5.py --resume-archive /path/to/interrupted_round_four.zip
```

Stage four uses `ready-balanced-v1` collection. A free worker starts its next
payment immediately, without waiting for the other 29 after every step. Every
worker still supplies exactly **66 transitions per rollout**; the buffer keeps
the `(time, worker)` trajectory order for GAE. PPO updates only after all
**1,980 transitions** were collected under the same policy. Minibatches remain
**60** and the additional requested budget remains **20,000**. Gaussian noise
is indexed by trajectory in advance so completion order does not change
exploration. Audit sampling uses the canonical trajectory timestep and worker
identity. Ready-order collection is an execution change, not an exact replay of
the former stochastic rollout implementation. Wall-clock routing timeouts can
still make outcomes sensitive to machine load. No routing limits, recovery
attempts, XAI sampling probabilities or input data were shortened for speed.

Checkpoints remain at completed PPO update boundaries. Interrupting collection
discards the partial rollout and retains the last coherent policy/optimizer,
worker and RNG snapshot. The initial checkpoint remains available. ZIPs are
created at phase exit; completed artifacts omit simulator recovery snapshots
and contain only the immediate parent policy plus a compact ancestor hash index.

Evaluation already schedules independent payments as workers finish. It now
creates isolated graphs from a cached immutable pickle snapshot instead of
Python deep-copy traversal. Each payment still has its own graph, failures,
private liquidity and seed. Its journal uses SQLite **WAL + synchronous FULL**,
committing each payment and explanation together. CSV and gzip exports remain
rebuildable from the journal. Setup/payment timings are retained in JSONL;
training metadata includes collector timing and equal per-worker counts.
On the downloaded 15,100-node / 128,424-channel graph, median isolated clone
time over three repetitions was **1.730 → 0.248 seconds** (about **6.98×**).
This measures setup alone, not complete payment or evaluation speed.

The alternating-latency subprocess benchmark measured **0.925 → 0.468 seconds**
(about **1.97×**), with the same worker budgets. This validates removal of the
per-step waiting barrier; it does not establish a full GitLab speed of 1.2
transitions/second. `[ROLLOUT]` and `[PPO PROGRESS]` print measured speed.

### Continue the same agent: main3/5

In GitLab, start **`run-main3-5`**, wait for successful training, then start
**`run-main3-5-resume` in the same pipeline**. Its `needs` dependency downloads
that pipeline's training artifact. Testing does not call PPO training.
The previous trained agent comes from completed stage-two Job **56595**;
`MAIN3_5_PREVIOUS_JOB_ID` can select another completed stage-two training/test job.

Locally, `python -u main3_5.py` detects the unique completed stage-two run in
`theResult` or `main2-5-*/theResult`, including the downloaded evaluation artifact.
Explicit inputs and interrupted stage-three recovery are also supported:

```bash
python -u main3_5.py --previous-archive /path/to/round_two_artifacts.zip
python -u main3_5_resume.py
python -u main3_5.py --resume-archive /path/to/interrupted_round_three.zip
```

Stage three loads the stage-two actor, critic, optimizer and cumulative counter;
this agent already carries stage-one learning. It adds **20,000 requested
transitions** to the completed **43,560**, for a target of **63,560** (normally
**65,340 actual transitions** after completing PPO rollouts). It keeps 30
workers, 66 rollout steps per worker, minibatches of 60, the same PPO settings,
path-search timeouts, checkpoint interval and sampled XAI. The observation
selection optimization is preserved. There is no artificial delay or guarantee
of 1.2 transitions/second: throughput depends on routing work and server load.

The progress log reports the slowest worker's payment, reset, CPU and routing
search times. Thirty synchronous environments wait for the slowest payment in
each batch; repeated recovery searches can therefore lower batch throughput.
Compare payment CPU with payment wall time to help identify CPU contention or
other waits. The measurements add only scalar fields to worker
info; they do not add graph snapshots or expand the sampled XAI records.

Optimizing graph construction alone does not establish faster training.
Wall-clock search deadlines can end slow payments early; making one search
faster can permit additional recovery searches and increase total payment time.
A local compact-graph experiment reduced graph construction from about 6.75 to
2.63 seconds, but did not consistently improve complete payments. It was not
enabled in the runners. A full-job speed of 1.2 transitions/second remains
unverified; no search budgets or recovery attempts were reduced to claim it.

The paper specifies different stochastic seeds on the same test transactions,
without numerical per-repeat seeds. We extend the existing 42, 43 sequence to
**44** for training, retaining transaction generation seed **42** and the exact
7,000/3,000 split. Evaluation has a separate base seed **1044**, recorded in
configuration, metadata, trial CSV/JSONL and reports. Each payment seed also
includes its stage, failure-rate index and transaction index. A higher seed
generates different random scenarios; it does not guarantee a harder test.
Training retains the previous **3%** setting; evaluation runs **3,000
payments at each of 1%, 2%, 3%, 4%, 5%, 6%**, totaling **18,000**. Evaluation
does not update weights or optimizer. Continual stages remain correlated and
are not five independent statistical repeats of the paper.

Outputs use `theResult/main3_5_<timestamp>/`, `models/rep_03_seed_44/`,
`reports/main3_5.html`, readable training/evaluation CSVs usable in Excel,
sampling coverage, summaries and the same PNG/SVG/CSV heatmaps. Artifacts contain
one compact stage-three ZIP plus SHA256. The inputs retain only the immediate
parent policy, fixed data and a small ancestor index in metadata; previous
full archives and traces are not duplicated. Completed artifacts need no
stage-one/two downloads for evaluation or recovery.

### Continue the same agent: main2/5

The stage-two GitLab templates are hidden while stage three is active.
The stage-two training entry point uses completed round-one job **56583** as its fixed data source
(`MAIN2_5_PREVIOUS_JOB_ID` can select another completed round-one job).
Do not retry the training job to run evaluation: use the separate resume job.

Locally, specify the previous completed run or its artifact ZIP:

```bash
python -u main2_5.py --previous-archive /path/to/round_one_artifacts.zip
python -u main2_5_resume.py
```

If the locally downloaded round-one folder is the only completed run under
`main1-5-*/theResult` or `theResult`, `python -u main2_5.py` detects it automatically.
Testing automatically selects the single round-two run/ZIP in `theResult`;
use `--resume-run`, `--resume-archive` or `--resume-job` to select among several.
Interrupted round-two training uses those same `--resume-*` options on
`main2_5.py`; `--previous-*` starts stage two from the trained round-one agent,
whereas `--resume-*` continues an interrupted stage-two checkpoint.
Extraction never overwrites an existing run directory.

At the user's request, stage two **continues the same trained agent**: PPO loads
round one's actor/critic parameters, optimizer state and actual timestep count.
It does not call `build_ppo` or reset learned parameters. The next environment RNG
uses seed **43**; the transaction pool remains the one generated with 42.
All six failure probabilities **1%–6%** are retained; stage two does not mean 2%
or 20%. This continual-training protocol differs from the paper's five independent
stochastic repetitions. Correlated training stages cannot be used as five
independent samples for confidence intervals or statistical comparisons.

Stage two adds the same **20,000 requested transitions** as stage one; it does
not use 20,000 as a cumulative stopping point. The downloaded round-one policy
actually contains **21,780** transitions due to PPO rollout rounding. The next
cumulative target is therefore **41,780**, with the final count again rounded to
complete rollouts (normally **43,560** with the same 30-worker configuration).
Rollout/minibatch sizing uses the stage budget, so retaining the cumulative
counter does not double rollout size or the stage's compute budget.

The **18,000 previous evaluation payments did not update weights or optimizer**;
they are held-out measurements. Their reports remain in the stage-one artifact.
PPO retains past learning through its parameters and optimizer, not a replay
buffer of all past training/test payments. New XAI traces contain only the new
stage's decisions, with cumulative timestep labels; old traces are not duplicated.
The 3% training failure setting is inherited from round one; it is a project
training choice, not a new paper-specified value for the second repeat.

The source snapshot, carbon dataset, cached graph, exact 7,000/3,000 split,
PPO settings and transition budget, timeouts, 30 training/evaluation workers,
5%/25%/25% XAI sampling, reporting corrections, heatmaps and 1800-second local
checkpoint interval are reused. No periodic ZIP creation or remote uploads
are added. Equal compute settings do not guarantee equal wall-clock time:
new seeds can change routing/search/recovery work and server load can vary.
Each repeat still evaluates **18,000 payments: 3,000 at each of six rates**.

Outputs are separate under `theResult/main2_5_<timestamp>/`, with
`models/rep_02_seed_43/`, `reports/main2_5.html`, readable CSVs, full routing
audits and the same PNG/SVG/CSV heatmaps. Native artifacts include only the
compact round-two ZIP and checksum. Only a small parent-policy ZIP is retained
inside the stage-two inputs for restart and lineage; no previous result archive
or old routing traces are nested. Input hashes and round identity are checked before
reuse/resume. Round-one artifacts cannot be evaluated as a round-two model.

### First independent round: main1/5

`main1_5.py` and `Runtime.round_one.MainOneOfFive` implement **only round 1/5**.
Run locally or on the server (round-one GitLab jobs are currently hidden):

```bash
python -m pip install -r requirements.txt tensorboard
python -u main1_5.py --snapshot 20230716.gml.geo
```

Fresh runs default to the supplied **20230716.gml.geo** snapshot (16 July 2023).
The same snapshot is selected explicitly by the preserved round-one CI template. Its original
filename and SHA256 are recorded in `run_metadata.json`, and an exact copy is
included at `inputs/network_snapshot.gml.geo` in downloadable artifacts.

The runner uses 30 subprocesses for PPO training, closes them, then uses 30
evaluation workers. Numeric-library threads are limited to one per process.
There is one fresh PPO model, the exact 10,000 / 7,000 / 3,000 transaction pool,
and all six configured evaluation failure rates by default (18,000 held-out
trials). Training uses the configured transition budget and a 3% failure setting
unless `training_failure_rate` is supplied. Actual transitions, including PPO
rollout rounding, are recorded. `--failure-rates` can explicitly select fewer
conditions; `--snapshot`, `--config` and `--output-dir` select inputs/output.
The runner never starts rounds 2–5.

Every actual training decision and held-out trial has a compressed JSONL audit:
the observed normalized input, candidate costs and physical
channels, Bucket rank order, payment outcomes, local suffix searches and recovery
events. Costly policy sensitivity is sampled deterministically within phase,
failure-rate and outcome/recovery strata: **5% of successful decisions, 25% of
failed decisions, and 25% of decisions with partial-backtrack attempts** by default.
The partial-backtrack stratum takes precedence over success/failure. These are
selection probabilities, not fixed quotas; small strata can have no samples.
Selection does not advance experiment RNGs and stays identical after resumption.
All selected training workers share one inference batch per vector step. Readable
prose is generated at report time and audit writes are buffered until checkpoints,
avoiding a gzip flush at every vector step. A hard stop can
lose a buffered, uncommitted rollout; the last sealed safe update is replayable.

Feature occlusion on selected decisions replaces each observed feature group with its
training-only mean. Training uses an online reference; testing freezes the final
training reference. This measures policy sensitivity, not causal attribution or
SHAP values. Sampled training actions and deterministic policy means are distinct.
The audit does not expose simulator failure probabilities or hidden balances.
Unsampled decisions retain the full routing explanation, with policy sensitivity
explicitly marked as not sampled. Heatmaps label sample counts and use
self-normalized inverse-probability weighting to account for unequal selection
probabilities; they are sample estimates, not exhaustive measurements. Missing
values remain blank, never zero. `reports/policy_sampling_coverage.csv` and
`summary.json` list total decisions and selected counts per phase/rate/stratum.
Candidate-cost and recovery heatmaps still use all audited decisions.

Override sampling in the config for a **fresh run**, if needed:

```yaml
round_one_xai:
  version: routing-all-policy-sampled-v1
  success_probability: 0.05
  failure_probability: 0.25
  partial_backtrack_probability: 0.25
```

Use `1.0` for all three probabilities to obtain exhaustive sensitivity, or `0.0`
to keep only the full routing audit. Saved configurations are authoritative on
resume; older resumable runs without this section retain exhaustive sensitivity.
This reduces policy-explanation overhead; routing, checkpoint and ZIP costs remain.

Each `theResult/main1_5_<timestamp>/` contains:

- `reports/main1_5.html`: Persian interpretation guide and heatmaps.
- `reports/training_readable.csv` and `reports/evaluation_readable.csv`: human
  explanations for every recorded training decision and held-out payment.
- `figures/`: PNG, SVG and numeric CSV exports of policy eta/value sensitivity,
  training sensitivity, Bucket candidate costs and actual recovery events.
- `evaluation_trials.csv`, `summary.json`, `report.txt`, compressed XAI traces,
  final policy, training reference, model metadata and checkpoints.
- `inputs/`: exact train/test manifest, effective config, snapshot, carbon data,
  worker caches, requirements and a source archive.
- `HANDOFF.md`, `artifact_manifest.json`, and `run_status.json` for transfer.

Readable CSVs distinguish payment attempt numbers from Bucket candidate ranks.
Recovered ranks are inferred only from an unambiguous match of the complete
route and exact physical channel identities. Combined routes and ambiguous or
missing evidence are explicitly labelled; a missing rank is never replaced by
zero. Recovery summaries include unsuccessful continuation/full-reroute
searches. Zero backward hops means continuation at the reached node, not absence
of a recovery attempt. Raw simulator outcomes remain unchanged.

Rebuild readable CSVs, HTML and heatmaps from saved traces without training or
evaluation (close any spreadsheet application holding the output CSV open):

```bash
python -m Evaluations.round_one_report --run-dir /path/to/extracted/main1_5_run
```

A sibling ZIP and `.zip.sha256` are built **only when the phase exits**, including
handled failures. Compute loops never build ZIPs or upload/download backups.
The ZIP contains per-file SHA256 hashes. Completed-training and evaluation
packages omit simulator recovery generations, ordinary model checkpoints and
discarded rollout traces, while retaining the final model, frozen training
reference, exact inputs/caches, training audit, results and reports. Interrupted
training packages contain only the latest verified coherent recovery generation.
These exclusions affect the exported ZIP, not the saved files in the run folder.

GitLab uploads **only the ZIP and checksum**, not another copy of the source
folder, and retains them for one month. `after_script` packages local files if
cleanup did not complete; an unchanged package is reused using a cheap file
size/mtime index instead of recompressing it. Each new package logs its size,
file count and packaging time. The server's artifact size limit still applies;
the smaller layout cannot guarantee compliance with an unknown server limit.

`MAIN1_5_REMOTE_BACKUP` defaults to `0` in CI. Setting it to `1` explicitly enables
one additional Package Registry upload at phase exit, without download/read-back.
There are no periodic external backups. Local saves alone do not protect against
pod removal or failed native artifact upload. Cancellation must allow packaging
and Runner upload to finish; force cancellation can prevent both. The legacy
`--backup-seconds` argument remains accepted but no longer schedules ZIP creation.

Training and testing run separately, each with 30 workers:

1. Run `python -u main1_5.py --snapshot 20230716.gml.geo` locally/on the server.
   It trains, saves artifacts, marks `TRAINING_COMPLETED`
   and exits without scheduling evaluation.
2. Run `main1_5_resume.py` with those artifacts. This entry verifies the
   completed model and runs testing/reports
   only; incomplete training is rejected instead of restarting training.

Test **the same first round** from a downloaded ZIP, an extracted directory,
or a completed training/interrupted testing job's artifacts:

```bash
python -u main1_5_resume.py --resume-archive /path/to/main1_5_saved.zip
python -u main1_5_resume.py --resume-run theResult/main1_5_saved
python -u main1_5_resume.py --resume-job 12345
```

Round-one CI templates are currently hidden. To continue incomplete training,
use `main1_5.py --resume-archive /path/to/saved.zip`; it completes training only.
For local ZIP restoration the output
directory must not already contain that run; existing runs are never overwritten.
Direct run ZIPs, the new native artifact wrapper containing one run ZIP/checksum,
and legacy artifact folders are accepted. No external artifact download occurs
for `--resume-run`.
Resumption uses the snapshot saved inside that run, including for older runs;
changing the fresh-run default does not change an existing run's input data.

Coherent local training checkpoints save model/optimizer state, parent and worker
RNGs, each worker's routing/simulator state, XAI feature sums and a sealed training
audit. Initial/final checkpoints are saved; intermediate saves occur at the first
completed PPO update boundary after **1800 seconds** since the previous save.
Use `--checkpoint-seconds 1800` to adjust the interval, or `0` to save at every
update for small debugging runs. The shared trainer's additional model-only checkpoint
is scheduled just once at the training budget, rather than every 128 transitions;
it is excluded from the exported package. Two coherent generations are kept locally
for corruption fallback, but only one is exported for interrupted training, and none
are exported after training is complete. Resumption uses the latest intact
generation; work after that saved update is replayed, and incomplete traces are
retained separately. It requires the saved
Python/library versions and original training worker count. Simulator state in
these recovery files is private execution state; it is not exposed in XAI inputs.
Ordinary mid-rollout model checkpoints are not used as authoritative resume points.

Held-out results and their full explanations commit together in
`evaluation.sqlite3` after every payment. On resume only missing trial identities
are scheduled; CSV/gzip reports are regenerated from committed rows, including
when a previous export was truncated. A completed trained model is reused.
Input hashes, conditions, policy and feature-reference identities are checked
before combining results. Saves and packages are atomic; an interrupted save
cannot replace the last good checkpoint/ZIP. A hard kill can lose work since
the last saved/uploaded recovery point, never requiring an intentional restart
of already committed training updates or held-out payments in that artifact.

Inspect `run_status.json`: a partial archive is not a completed scientific
stage. Round 2 continues the trained round-one policy and optimizer: reuse the same
transaction split and inputs, preserve round-one results, and add one stage budget.
Continuing a failed stage does not increment its round, seed or cumulative target.
Artifacts produced by the earlier non-resumable implementation do not contain
these coherent recovery checkpoints and are rejected as incompatible.

Small synthetic validation only (eight-transition targets, four held-out cases,
two processes; includes intentional interruptions and recovery):

```bash
python -m unittest test_main1_5 test_main1_5_resume -v
```

Install `requirements.txt` and TensorBoard in the execution environment.
Start a fresh single-seed run with 6% held-out channel failures:

```bash
python -u main-test.py --failure-rates 0.06 --repetitions 1
```

Resume the saved run without retraining its completed model:

```bash
python -u main-test-resume.py --run-dir theResult/run_20261016_035115
python -u main-test.py --resume-run theResult/run_20261016_035115 --failure-rates 0.06 --repetitions 1 --evaluation-workers 30
```

`main-test-resume.py` uses the existing saved run when its required files are
present. If it is absent, the GitLab job downloads the configured source job's
artifacts. An explicit local ZIP can be supplied with `--archive` when restoring
into an absent destination. Existing directories are not overwritten.

The existing GitLab MAIN TEST allocation remains 30 CPU cores and 30 GiB RAM with
one available GPU, 25 PPO environments, and 30 held-out workers in the resume
job. Each process limits numeric-library threads to one.

## Preserved completed result

- Report: `result.txt`
- Run: `theResult/run_20261016_035115/`
- Trial records: `theResult/run_20261016_035115/evaluation_trials.csv`
- Summary: `theResult/run_20261016_035115/summary.json`
- Original transactions: `theResult/run_20261016_035115/transaction_split.csv`
- Final policy and metadata: `theResult/run_20261016_035115/models/rep_01_seed_42/`
- Existing figures: `output/charts/run_20261016_035115/`

This is the completed single-seed, 6% evaluation: 3,000 held-out trials,
2,755 successful payments, and 91.8333% success. The model used 7,000 training
transactions, seed 42, a 3% training failure setting and 20,000 training
transitions. Training and evaluation failure settings are different.
Only the final model is retained; intermediate model checkpoints and disposable
worker/training logs were removed. Historical reports can contain the label
MAIN4, the old name of the internal protocol used by MAIN TEST.

## Code and data

### Routing recovery

Recovery version `local-prefix-v2` checks the nearest reached node first. At each
node it tries cached Bucket suffixes, then local Top-K suffix search, before
moving one hop upstream. The exact successful prefix is preserved; only the
new suffix is forwarded. A failed continuation starts another recovery cycle.
Known failed nodes/channels, loops, the total hop limit and payment-attempt
budget constrain recovery. Source retries are used when no intermediate node
has a usable continuation, including first-hop failures. Search timeouts are
reported and can also prevent local recovery.

`evaluation_trials.csv` includes JSON `recovery_events` with the failed channel,
branch node/index, backtracked hops, preserved prefix, suffix origin and outcome.
`partial_backtrack_count` retains its legacy meaning of recovery checks after
failure. `partial_backtrack_retry_count` counts actual continuations with a
nonempty preserved prefix, and `partial_backtrack_success` counts payments
completed by one of those continuations. `source_retry_count` counts actual
retries from the source. `preserved_prefix_hops` and `backtracked_hops` sum these
quantities over recovery attempts. These totals also appear in the text report
and JSON summaries.

The preserved October 16 result uses the earlier implementation. Its outputs
are historical, not measurements of this correction. New runs record the
recovery version in metadata; resuming an older implementation is rejected to
prevent combining incompatible evaluation results. The historical resume
examples above apply to the original code version.

- `Runtime/experiment_core.py`: shared graph, reporting and PPO loading helpers.
- `Runtime/paper_protocol.py`: internal MAIN TEST training/evaluation protocol.
- `Evaluations/parallel_trials.py`: independent held-out trial scheduling.
- `Configs/config.yaml`: original experiment settings.
- `20190501.gml.geo`: the network snapshot used by the saved result.
- `global_energy_mix.json`: the country carbon-intensity dataset.
- `Network/`, `Simulation/`, `Pathfinding/`, `Bucket/`, `RL/`: routing libraries.

The former standalone main runners and unrelated experiment outputs were removed.
The shared modules retain the calculations used by MAIN TEST; scientific settings,
capacity assumptions and fee-unit handling were not revised by this cleanup.

## Preserved explainable AI utilities

- `Evaluations/explainable_routing.py`: observable-feature sensitivity and audits.
- `Evaluations/xai_report.py`: human-readable explanations and heatmaps.
- `Evaluations/explainable_analysis.py`: scientific summaries and figures.
- `Runtime/recovery_safety.py`: bounded-memory and recovery utilities they use.

These utilities remain available for subsequent integration. Preserving them
does not add diagnostic or XAI execution stages to the MAIN TEST protocol.

## Cleanup scope

Only the selected MAIN TEST result and its figures are retained. Legacy runners,
Colab bundles, unrelated artifacts, temporary presentation files, duplicate
reports and Python caches were removed. The supplied `20230716.gml.geo` snapshot
is retained for main1/5. Git history is retained; deleted historical blobs can
still contribute to repository size.
