# Reproduce the publication subset

## Environment

The pinned environments and model are unchanged. Use Linux, NVIDIA GPU inference, Python 3.12 for core/inference and the prescribed separate Python 3.9 baseline environments. Install PostgreSQL for HoloClean; install the pinned Codex CLI 0.153.1 and Qwen3.5-9B BF16 weights. Check `environments/`, `provenance/baseline_sources.json`, `model-files-9B.sha256` and `environment_sha256.json`.

```sh
python3.12 scripts/setup.py all --python312 /path/to/python3.12 --python39 /path/to/python3.9
npm install --prefix .tools @openai/codex@0.153.1
python3 scripts/download_model.py --output models/Qwen3.5-9B
cp machine.example.json machine.json
```

Fill machine-specific interpreter/model/Codex/PostgreSQL paths and free GPU/ports. Do not substitute model weights or change numerical settings to match a score.

## Main repair, downstream and selected module ablation

```sh
.venvs/core/bin/python reproduce.py doctor
.venvs/core/bin/python reproduce.py run --study repair --seed 1013 --config machine.json
.venvs/core/bin/python reproduce.py run --study downstream --seed 1013 --config machine.json
.venvs/core/bin/python reproduce.py run --study ablation --seed 1013 --config machine.json
```

Repeat for 1014 and 1015. Main repair retains all eight tables, downstream retains the six Table-IV datasets, and ablation dispatches Retinopathy/Airfoil with A/B/D. A_saved is an execution-parity control; it is not an extra reported comparison. B and D execute the same newly learned A skill. Core code, scoring functions and environment versions are unchanged.

## Supplementary studies

The [data guide](DATA_SOURCES.md) identifies the public-source reconstruction routes. Saved server input bundles and answers are not distributed. Reproducers download the pinned official data and generate their own method inputs and independent scoring references locally. Prepare new roots with `scripts/supplement_runtime.py`; use `doctor` before spending model calls. All frozen input members, including registered source ZIPs, are checked before materialization. Fresh case/cache provenance is separately recorded; scientific input bytes must match the original fingerprints.

```sh
python scripts/reproduction_data.py download-uci --dataset retinopathy --out raw/retinopathy.zip
python scripts/reproduction_data.py download-uci --dataset airfoil --out raw/airfoil.zip
python scripts/rebuild_controlled_inputs.py --study llm_digit70 --raw-dir raw --output prepared/llm_digit70
python scripts/supplement_runtime.py prepare --study llm_digit70 --inputs prepared/llm_digit70 --work runs/llm_digit70 --machine supplement.machine.json
python scripts/supplement_runtime.py doctor --work runs/llm_digit70
python scripts/supplement_runtime.py run-batch --work runs/llm_digit70 --gpu 0 --max-real-calls 100
```

For the remaining three supplementary studies, download official sources once. The raw archive fingerprints must match the paper versions.

```sh
python scripts/reproduction_data.py download-uci --dataset concrete --out raw/concrete.zip
python scripts/reproduction_data.py download-uci --dataset retinopathy --out raw/retinopathy.zip
python scripts/reproduction_data.py download-uci --dataset airfoil --out raw/airfoil.zip
```

For either 24-case robustness group, rebuild using the original pure curator. Example for Retinopathy; use `advisor_supplement` and separate directories for the Concrete study:

```sh
python scripts/rebuild_supplement_inputs.py --study robustness_retinopathy --raw-dir raw --output prepared/robustness_retinopathy
python scripts/supplement_runtime.py prepare --study robustness_retinopathy --inputs prepared/robustness_retinopathy --work runs/robustness_retinopathy --machine supplement.machine.json
python scripts/supplement_runtime.py run-batch --work runs/robustness_retinopathy --gpu 0 --max-real-calls 240
```

For the six-case reuse study, use the dedicated whole-batch entry. It learns all six skills, stops both owned GPU lanes, and only then executes and scores all 18 new batches. Do not repeat the single-case smoke entry to represent the full study.

```sh
python scripts/rebuild_supplement_inputs.py --study sealed_reuse --raw-dir raw --output prepared/sealed_reuse
python scripts/supplement_runtime.py prepare --study sealed_reuse --inputs prepared/sealed_reuse --work runs/sealed_reuse --machine supplement.machine.json
python scripts/supplement_runtime.py run-reuse-batch --work runs/sealed_reuse --max-real-calls 60
```

Robustness workers dispatch only their retained full condition; they preserve original source preparation, prior/context learning and validation. The numerical scorer functions are unchanged. Sealed reuse learns its skills, shuts down owned model services, then executes CPU-only batches with no new calls or fitting. An unknown failure must stop scoring and preserve evidence, not trigger automatic retries.

## Retinopathy threshold scan

```sh
python scripts/reproduce_threshold.py prepare --main-runs runs --work runs/threshold --machine machine.json
python scripts/reproduce_threshold.py run --work runs/threshold
```

Complete new main ablation A for all three seeds first. The paper scan has three dataset-seed cases and fifteen outputs. Original independent once-only metric functions, five offsets, zero-output parity and reference-access gates remain unchanged.

## Recorded results and scope checks

```sh
python3 scripts/verify_paper_release.py
python3 -B -m unittest discover -s tests -v
python3 scripts/export_paper_artifacts.py --out paper_outputs
```

These are offline checks and artifact export, not a fresh GPU experiment. This publication packaging has not retrained every included study from scratch. Archived request cost remains historical. Reproduction reports must distinguish verified recorded results, fixed-input reconstruction, interface smoke tests, and actual end-to-end reruns. Scope views are derived dispatch configurations; the full original research record remains in the author archive.
