# Paper experiment index

This release contains only the declared publication subset. Results are under `results/studies/<study>/`, configuration and archived execution code under `experiments/paper_runs/<study>/`. `results/paper_artifacts/artifact_index.json` maps the eight screenshot groups to exported table and figure data.

| Paper location | Study | Datasets | Seeds |
|---|---|---|---|
| Table III | repair_main | monks1, cmc, bc_original, mammographic, ilpd, climate, voting, qsar | 1013, 1014, 1015 |
| Table IV | downstream_main | tic_tac_toe, car_evaluation, retinopathy, real_estate, concrete, airfoil | 1013, 1014, 1015 |
| Table V, Fig.5a, Fig.5b | module_ablation | retinopathy, airfoil | 1013, 1014, 1015 |
| Fig.6, Table VIa | advisor_supplement | concrete | 1013, 1014, 1015 |
| Fig.5c | llm_digit70 | retinopathy, airfoil | 2013, 2014, 2015, 2016, 2017 |
| Table VIb | threshold | retinopathy | 1013, 1014, 1015 |
| Fig.6, Table VIa | robustness_retinopathy | retinopathy | 1013, 1014, 1015 |
| Fig.7b | efficiency | tic_tac_toe, car_evaluation, retinopathy, real_estate, concrete, airfoil | 1013, 1014, 1015 |
| Fig.4 | tabclean_same9b | banknote, retinopathy, wdbc, airfoil, energy, wine_red, real_estate | 2021, 2022, 2023 |
| Fig.7a | sealed_reuse | retinopathy, airfoil | 1013, 1014, 1015 |
| Table II, Fig.3 | real_dirty | address_museum, movies, address_shoppingcenter, school_small, jobposting_small | 1013, 1014, 1015 |

The Concrete 70% mixed-pollution point in Fig.6 uses `advisor_supplement`'s original `main` full condition; the other Concrete robustness points use its `robustness` conditions. Do not replace this with the separate module-ablation Concrete experiment. Fig.7 cost uses a historical six-table ledger; its 11 first-run calls include 10 learning calls plus one scheduling call. Reuse skills in Fig.7a use 10 learning calls and zero new batch calls.

`original_config/` remains the compatibility directory name expected by the runner. JSON files marked `publication_scope_projection` are derived dispatch views; they are not mislabelled as byte-identical original historical protocols. Their original fingerprints and full archive are preserved separately. Case seeds, model settings, pollution settings, metric functions and qualification gates are unchanged.
