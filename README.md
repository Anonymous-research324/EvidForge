# EvidForge

### Example-Guided Tabular Repair with Frozen Skills

This anonymous review repository accompanies **EvidForge: Example-Guided Tabular Repair with Frozen Skills**. It contains the system implementation used in the paper, together with the configurations, execution scripts, and recorded results for the main and supplementary experiments.

[Overview](#overview) · [Getting started](#getting-started) · [Paper experiments](#paper-experiments) · [Results](#results-and-verification) · [Documentation](#documentation)

## Overview

EvidForge learns reusable tabular repair skills from authorized correction examples. It combines example-derived repair candidates, learned candidate comparisons, and bounded LLM proposals, then evaluates the resulting skill on a separate validation set before registering it for automatic use.

The key distinction is between **learning a repair skill** and **executing a frozen skill**. The LLM helps propose candidate-scoring and acceptance programs during learning. Once a skill is frozen, repair runs through the saved program without further LLM calls.

### Workflow

```text
Field and source declarations + correction examples
                         |
                         v
        Source-internal fitting and candidate generation
                         |
                         v
        Bounded LLM proposals and source-internal feedback
                         |
                         v
           Independent validation and version selection
                         |
                         v
                  Frozen repair skill
                         |
                         v
             Programmatic repair of new batches
                    (no new LLM calls)
```

- **Example-guided candidates.** Candidate generation uses the authorized examples and retains the original value as a KEEP option. Candidate ranking and the decision to edit are separate steps.
- **Bounded LLM participation.** Proposals are checked and evaluated through a restricted interface. The LLM does not directly fill the final table with arbitrary generated values.
- **Independent qualification.** Validation examples are kept separate from source-internal learning. Qualification is tied to the exact skill version, source context, and field roles.
- **Reusable execution.** A frozen skill can process compatible new batches without repeating learning. Unqualified research checkpoints remain distinguishable from skills approved for automatic repair.

The implementation-to-paper mapping is documented in [System protocol](docs/SYSTEM_PROTOCOL.md).

## Getting started

### Requirements

The experiment runners target Linux with NVIDIA GPUs. The core and inference environments use Python 3.12; several baseline environments use Python 3.9. Model-based runs also require the pinned Qwen3.5-9B weights, Codex CLI 0.153.1, and the dependencies of the selected experiment. HoloClean comparisons require PostgreSQL.

Package versions, baseline revisions, and model fingerprints are recorded in [`environments/`](environments/), [`provenance/baseline_sources.json`](provenance/baseline_sources.json), and [`provenance/model-files-9B.sha256`](provenance/model-files-9B.sha256).

### Installation

```bash
git clone https://github.com/Anonymous-research324/EvidForge.git
cd EvidForge

# Install the core system environment.
python3.12 scripts/setup.py core --python312 /path/to/python3.12

# Inspect the system command-line interface.
.venvs/core/bin/python engine/repair_platform.py --help
```

For model learning and baseline experiments, follow the environment setup in [Reproduction guide](docs/REPRODUCE.md). It covers inference dependencies, model download, baseline environments, and local machine configuration.

### Execute a frozen skill

Given a previously learned and independently qualified skill:

```bash
.venvs/core/bin/python engine/repair_platform.py run \
  --input /path/to/new_batch.csv \
  --skill /path/to/frozen_skill \
  --context /path/to/source_context.json \
  --out /path/to/new_output_directory
```

The source context must match the skill's recorded scope. The command checks qualification and writes the repaired table, decision records, and output fingerprints. It does not call the LLM. Learning, validation, and registration are separate stages; see [System protocol](docs/SYSTEM_PROTOCOL.md) for their interfaces.

## Paper experiments

The repository includes both the main study and the supplementary experiments used in the paper.

| Experiment group | Included evaluations |
| --- | --- |
| Main comparisons | Exact repair on eight tables; downstream prediction on six tables; four-method comparisons on eight real-error tables |
| Module ablations | Paired module ablations and the independent Car/Concrete supplementary study |
| LLM ablations | 70% single-digit replacement; three new error conditions: digit transposition, cross-field addition, and scale/suffix composition |
| Sensitivity and robustness | Confidence thresholds, joint learning/validation example budgets, pollution rates, and error types |
| Same-model comparison | Seven-table EvidForge/TabClean comparison using the same 9B model |
| Reuse, cost, and case studies | Frozen-skill execution on new batches, recorded learning/execution costs, Hospital examples, and controlled semantic-conflict checks |

**Start with the [paper experiment index](docs/PAPER_EXPERIMENTS.md)** for the mapping from paper tables and figures to configurations, scripts, and seed-level results.

### Main experiment entry points

After preparing data, installing the required environments, and setting local paths in `machine.json`:

```bash
.venvs/core/bin/python reproduce.py doctor
.venvs/core/bin/python reproduce.py run --study repair --seed 1013 --config machine.json
.venvs/core/bin/python reproduce.py run --study downstream --seed 1013 --config machine.json
.venvs/core/bin/python reproduce.py run --study ablation --seed 1013 --config machine.json
```

Run the remaining prescribed seeds and conditions according to each experiment's configuration. Supplementary studies have their own entry points; see [Reproduction guide](docs/REPRODUCE.md) and [TabClean and real-error baseline runs](docs/BASELINE_RUNS.md).

### Data and evaluation

[Data preparation](docs/DATA_SOURCES.md) lists the public sources, fixed subsets, entity partitions, and construction scripts. Data files and model weights are not bundled with the repository. The guide identifies which inputs can be rebuilt directly from public sources and which require the fixed input export procedure.

Learning, independent validation, and final evaluation have separate roles. Final references are read by the scorer after outputs are frozen. Controlled pollution and naturally occurring errors are reported as separate settings. Pollution denominators, sampling parameters, and seeds are experiment-specific; use each study's configuration rather than a single global default.

Repair metrics include exact-repair precision, recall, F1, and normal-cell damage. Results retain prescribed seeds, zero scores, and missing values; aggregate statistics use the study's recorded protocol. Deployment qualification is reported separately from repair performance.

## Results and verification

Recorded numerical results are available in [`results/studies/`](results/studies/). The text sources for the paper's 17 tables are in [`results/paper_tables/final/`](results/paper_tables/final/).

To check the packaged implementation and recorded numerical artifacts without running model experiments:

```bash
python3 scripts/verify_paper_release.py
python3 -B -m unittest discover -s tests -v
```

These checks verify file fingerprints, core implementation integrity, numerical consistency, and offline protocol behavior. They do not retrain skills or rerun the paper experiments.

## Repository structure

```text
engine/                  System implementation used in the paper
experiments/             Experiment orchestration and per-study configurations
results/                 Recorded results and paper table sources
scripts/                 Setup, data construction, execution, and verification tools
environments/            Dependency specifications and version locks
provenance/              Source revisions, fingerprints, and paper-to-file mappings
tests/                   Offline checks for system and experiment contracts
docs/                    System, data, and experiment documentation
```

## Documentation

- [System protocol and implementation mapping](docs/SYSTEM_PROTOCOL.md)
- [Paper experiments and result locations](docs/PAPER_EXPERIMENTS.md)
- [Installation and reproduction guide](docs/REPRODUCE.md)
- [Data sources and input preparation](docs/DATA_SOURCES.md)
- [TabClean and real-error baseline runs](docs/BASELINE_RUNS.md)
- [Baseline adaptations and interpretation limits](docs/KNOWN_LIMITATIONS.md)

## Third-party software and data

The experiments use third-party baselines, datasets, model weights, and tools under their respective terms. Source revisions and attribution are recorded in [Third-party notices](THIRD_PARTY_NOTICES.md). No additional license for those dependencies is granted by this repository.

For snapshot packaging details, see [Anonymous review artifact](docs/ANONYMOUS_REVIEW.md).
