# Native LightRAG, ER, and ER+RR experiment

This repository contains the research implementation used to study LLM-based graph construction.

```text
Native -> ER -> ER+RR
```

- **Native** is the graph produced by LightRAG.
- **ER** applies corpus-level entity resolution to Native entities.
- **ER+RR** applies evidence-based relation recovery to the ER graph.

For each builder model, extraction is performed once. ER and RR use the saved
extraction records and create separate workspaces; they do not call the builder
model or modify the Native workspace. RR may add verified relations, but it
does not add entities or external knowledge.

## Experimental design

The fixed corpus is [`multihoprag_120`](multihoprag_120):

| Item | Value |
| --- | --- |
| Questions | 120: 30 inference, 30 comparison, 30 temporal, 30 unanswerable |
| Documents | 155: 125 gold documents and 30 hard negatives |
| Subset ID | `multihoprag_120_a3b015d303a201a2` |
| Manifest SHA-256 | `97681f1d0693486a89a868cdaf2bb760a304d74eb476d8f04781a308a71eeec8` |

The experiment contains 12 builder models from the Qwen3.5, Qwen3, and Gemma 3
families. Exact model tags, digests, generation settings, and all scientific
parameters are stored in [`configs/experiment.yaml`](configs/experiment.yaml).

Each builder produces one Native graph and two derived graphs. Retrieval and
answering are then run for all three conditions on the same 120 questions:

```text
12 extractions
36 graph conditions
36 retrieval outputs
36 answer outputs
12 paired evaluations
```

The paired contrasts are:

1. primary: `ER+RR - Native`;
2. secondary: `ER - Native`;
3. incremental: `ER+RR - ER`.

These are cumulative conditions. The repository does not define a separate
Native+RR condition or estimate an independent RR effect.

## Repository contents

```text
experiment/
├── .gitmodules                  pinned external repositories
├── configs/experiment.yaml       fixed experiment configuration
├── LightRAG/                     LightRAG submodule
├── MultiHop-RAG/                 MultiHop-RAG submodule
├── multihoprag_120/              fixed questions, documents, and selection audit
├── requirements.txt              Python dependencies
├── scripts/experiment.py         command-line entry point
├── src/
│   ├── config/                   configuration and model identity checks
│   ├── extraction/               raw response capture and normalized records
│   ├── entity_resolution/        ER candidates, scoring, decisions, and clustering
│   ├── relation_recovery/        RR candidates, verification, and aggregation
│   ├── graph/                    graph rewriting, materialization, and validation
│   ├── retrieval/                context retrieval
│   ├── answering/                answer generation from saved contexts
│   ├── evaluation/               metrics and paired comparisons
│   └── orchestration/            stage execution, paths, status, and validation
├── tests/                        final experiment tests
├── analysis/                     thesis analysis notebooks and exported figures
├── runs/                         graph workspaces, records, metrics, and reports
```

The final pipeline is implemented in `src/` and exposed by
`scripts/experiment.py`.

## Clone the repository

LightRAG and MultiHop-RAG are Git submodules pinned to the versions used in the
experiment. Clone the repository together with both submodules:

```bash
git clone --recurse-submodules <repository-url>
cd experiment
```

If the repository was cloned without `--recurse-submodules`, initialize the
submodules before installing the dependencies:

```bash
git submodule update --init --recursive
```

## Installation

Python dependencies include the local LightRAG checkout. Install them from the
experiment directory:

```bash
cd experiment
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/experiment.py --help
```

The experiment uses the local `LightRAG/` checkout at commit
`ffd831f5de49d063da0cb69282781143cfb5b322` and the installed distribution
`lightrag-hku==1.5.2`. See [`LIGHTRAG_PARITY.md`](LIGHTRAG_PARITY.md) for the
version-specific adapter assumptions.

Model-backed stages require the configured Ollama models to be available at
`runtime.ollama_host` (default `http://localhost:11434`). The code checks exact
model names and digests and does not download or substitute models.

On the first real model-backed run, `tiktoken` may download the tokenizer data
for `gpt-4o-mini` once if it is not already cached locally. Later runs reuse the
cached tokenizer data. This small download is separate from the Ollama models.

## Configuration

[`configs/experiment.yaml`](configs/experiment.yaml) fixes:

- the corpus paths, hashes, and expected record counts;
- the 12 builder models and their identities;
- query, answer, embedding, ER-judge, and RR-verifier roles;
- extraction and chunking settings;
- ER and RR decision rules;
- graph storage and materialization settings;
- retrieval and answer-generation settings.

Important fixed settings include:

| Component | Setting |
| --- | --- |
| Chunking | 1,200 tokens with 100-token overlap |
| Retrieval | hybrid |
| `top_k` / `chunk_top_k` | 20 / 20 |
| Retrieval token limits | 2,048 entity; 3,072 relation; 7,000 total |
| Reranking | disabled |
| ER reject / automatic merge boundaries | 0.50 / 0.76 |
| ER ambiguous-pair review | reciprocal embedding neighbour, mutual top-1 |
| RR evidence | exact quotation from the source chunk required |
| Random seed | 42 |

The ER boundaries were fixed before the main evaluation. Main-experiment QA
results were not used to choose them. The Qwen3 8B judge-call limit override in
the configuration changes only the permitted number of already admitted judge
calls; it does not change ER candidates, scores, or decisions.

`preflight` checks the configured files, hashes, record counts, model identities,
LightRAG extraction prompt, and available disk space. With `--resolve-output`,
it writes a separate resolved configuration next to the source YAML.

Example for one builder:

```bash
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.yaml \
  preflight \
  --builder qwen35_0_8b \
  --resolve-output configs/experiment.resolved.yaml \
  --report runs/preflight/qwen35_0_8b.json
```

Use the resolved configuration for model-backed runs. A mismatch in an input,
hash, or model identity stops the command instead of changing the experiment.

## Running one builder

Run commands from `experiment/`. Replace `qwen35_0_8b` with any builder key in
the configuration.

The shortest command is:

```bash
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  resume --builder qwen35_0_8b
```

It runs or reuses compatible outputs in this order:

```text
Native build
-> ER plan and ER workspace
-> RR plan, verification, and ER+RR workspace
-> retrieval and answering for Native, ER, and ER+RR
-> paired evaluation
```

Completed stages are reused only when their configuration, input records,
lineage, and output hashes still match.

The same stages can be run individually:

```bash
# One Native extraction and graph build
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  native-build --builder qwen35_0_8b

# Entity resolution and ER graph
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  er-plan --builder qwen35_0_8b
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  er-quality-gate --builder qwen35_0_8b
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  er-materialize --builder qwen35_0_8b

# Relation recovery and ER+RR graph
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  rr-plan --builder qwen35_0_8b
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  rr-verify --builder qwen35_0_8b
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  rr-quality-gate --builder qwen35_0_8b
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  rr-materialize --builder qwen35_0_8b
```

Retrieval and answering are run separately for each condition:

```bash
for regime in native_lightrag advanced_lightrag_er advanced_lightrag_er_rr; do
  .venv/bin/python scripts/experiment.py \
    --config configs/experiment.resolved.yaml \
    retrieval --builder qwen35_0_8b --regime "$regime"
  .venv/bin/python scripts/experiment.py \
    --config configs/experiment.resolved.yaml \
    answering --builder qwen35_0_8b --regime "$regime"
done

.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml \
  evaluation --builder qwen35_0_8b
```

Use `--reclaim-running` only after confirming that an interrupted process is no
longer active. This option is available for commands that write stage outputs.

To inspect saved stage records:

```bash
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml status
```

The optional all-builder `resume` command requires both
`metadata.full_experiment_authorized: true` in the selected configuration and
the `--authorize-full-experiment` flag. This prevents an accidental expensive
run; it does not change the scientific method.

## Analysis

After all 12 builder evaluations are complete, collect the prespecified
within-builder contrasts:

```bash
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml primary-analysis
```

Then create the exploratory cross-builder comparisons:

```bash
.venv/bin/python scripts/experiment.py \
  --config configs/experiment.resolved.yaml exploratory-analysis
```

The evaluation uses the same question IDs for paired conditions. It reports
both intention-to-evaluate and complete-case estimates, keeps question types
separate, and also reports micro, macro, and source-weighted aggregates. Paired
95% confidence intervals use 10,000 question-ID-paired, question-type-stratified
bootstrap resamples with seed 42. Binary cross-builder comparisons use exact
McNemar tests with Holm correction within each defined family. Graph-level
metrics are descriptive; graph nodes and edges are not treated as independent
replicates.

## Data and artifacts

The publication-ready corpus and final experiment artifacts are available as
the [GraphRAG dataset on Hugging Face](https://huggingface.co/datasets/boblaros/GraphRAG).
The dataset includes extraction records, final Native/ER/ER+RR graphs,
retrieval outputs, generated answers, question-level metrics, aggregate
evaluation results, provenance metadata, and SHA-256 checksums. Large mutable
workspaces, vector stores, caches, logs, and model weights are intentionally
excluded.

The versioned [`multihoprag_120`](multihoprag_120) directory contains the fixed
120-question subset, its 155 documents, manifest, selection report, and
provenance records. The `MultiHop-RAG` submodule records the upstream source and
license. Its full locally downloaded `MultiHopRAG.json` and `corpus.json` files
are needed only to rebuild the subset and are not committed to this repository.

Generated files are stored under `runs/`:

```text
runs/
├── status.json
├── preflight/                    configuration and identity reports
├── base/<base_run_id>/           one extraction snapshot per builder
│   ├── config.frozen.json
│   ├── base_manifest.json
│   ├── artifacts/extraction/     normalized chunks, entities, and relations
│   └── <native_run>/             raw calls, Native exports, and workspace
├── variants/<variant_run_id>/
│   ├── variant_manifest.json
│   ├── artifacts/                ER/RR plans, graph exports, retrieval, answers
│   ├── gates/                    validation reports
│   ├── metrics/                  question metrics and paired summaries
│   └── workspace or materialization_attempts/
└── analysis/
    ├── prespecified_cascade_comparisons.<hash>.json
    ├── exploratory_model_pairs.<hash>.jsonl
    ├── exploratory_difference_in_differences.<hash>.jsonl
    └── global_experiment_report.<hash>.json
```

Scientific JSONL artifacts include schema and lineage metadata. Manifests store
the hashes and record counts used to decide whether an existing output can be
reused. Runtime reports record elapsed time and model-call counts separately
from the scientific ER and RR artifacts. The `runs/` directory is intentionally
excluded from Git because it contains large generated workspaces, model traces,
and reproducible outputs rather than source files.

## Tests

Run the complete local test suite:

```bash
.venv/bin/python -m pytest -q
```

Current result:

```text
196 passed
```

Run the source and test linter:

```bash
ruff check src scripts tests multihoprag_120/tests
```

The tests use deterministic local substitutes, fake Ollama clients, and
lightweight in-memory tokenizers. They do not require a running Ollama service,
installed Ollama models, or a tokenizer download.

## Licenses

The original experiment code is released under the [MIT License](LICENSE).
LightRAG remains under its original [MIT License](LightRAG/LICENSE), and the
MultiHop-RAG data is distributed under the [Open Data Commons Attribution
License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for versions, source links,
and attribution details.

## Citation

If you use this repository, please cite the accompanying thesis:

```bibtex
@mastersthesis{kutivadze2026graphrag,
  author = {Georgii Kutivadze},
  title = {Knowledge Graph Construction Quality in Local GraphRAG: An Empirical Study of Model Capacity, Graph Post-Processing, and Downstream Performance},
  school = {Università Cattolica del Sacro Cuore},
  year = {2026},
  type = {Master's thesis}
}
```

Use of the bundled third-party software or data should also cite the upstream
[LightRAG](LightRAG/README.md#-citation) and
[MultiHop-RAG](MultiHop-RAG/README.md#citation) publications.
