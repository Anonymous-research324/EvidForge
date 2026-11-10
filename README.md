# EvidForge

Example-Guided Tabular Repair with Frozen Skills.

This public artifact covers the eight paper screenshot groups: Tables II--VI and Figures 3--7. The exact datasets, conditions, methods and seeds are in [paper_scope.json](provenance/paper_scope.json). The 124-file numerical core, dependency locks, baseline revisions and model fingerprints are unchanged from the archived experiment artifact.

## Paper results

The [paper experiment index](docs/PAPER_EXPERIMENTS.md) maps each table and figure to its per-seed recorded results and execution configuration. Table/figure data and editable table sources are in [results/paper_artifacts](results/paper_artifacts). All prescribed seeds, genuine zero scores, unavailable results, research-prior fallbacks and external-qualification statuses for included conditions are retained.

```sh
python3 scripts/verify_paper_release.py
python3 -B -m unittest discover -s tests -v
python3 scripts/export_paper_artifacts.py --out paper_outputs
```

These commands verify and aggregate recorded observations. They do not retrain the model or constitute a fresh end-to-end experiment run. The [publication validation record](provenance/publication_validation.json) lists the input reconstruction and runtime preparation checks actually completed.

## Reproduction

Follow [installation and execution](docs/REPRODUCE.md), [data preparation](docs/DATA_SOURCES.md), [baseline execution](docs/BASELINE_RUNS.md) and [system protocol](docs/SYSTEM_PROTOCOL.md). Linux/NVIDIA execution, the pinned Qwen3.5-9B BF16 weights and study-specific baseline environments are required. Original inputs or publicly rebuilt inputs must match their frozen fingerprints before execution.

```sh
python3.12 scripts/setup.py core --python312 /path/to/python3.12
.venvs/core/bin/python reproduce.py doctor
.venvs/core/bin/python reproduce.py run --study repair --seed 1013 --config machine.json
```

Use the prescribed remaining seeds and the study-specific instructions. Frozen-skill execution uses `engine/repair_platform.py run`; a qualified skill is required for deployment. Research-prior execution does not imply deployment qualification.

This is a publication subset of a larger research record. The six-table downstream scope was selected after the original seed1013 study; the representative real-error and module-ablation displays were selected from existing results. This packaging change does not change that selection history or establish an independent blind benchmark. Full source records remain in the author archive.

Third-party terms, source revisions and attributions are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). See [known interpretation limits](docs/KNOWN_LIMITATIONS.md).
