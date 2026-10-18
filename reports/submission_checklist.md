# Submission checklist

Checked on 17 October 2026 against the files in this repository. An open item is not done.

| Item | Status | Where to look |
| --- | --- | --- |
| Frozen champion | Histogram gradient boosting, untuned refit, 12 iterations. Not replaced by later comparisons. | `results/tables/06_frozen_decision.json`, `models/frozen_resolution_pipeline.joblib` |
| Work-centre comparison | Two models were fit on the training period and scored on the 2025 validation period. The setting without work centre matches the saved champion validation macro-F1. The setting with work centre scores higher and is not the champion, because the extract has no assignment time. The locked test was not used. | `results/tables/09_work_centre_comparison.csv`, `results/tables/09_work_centre_per_class.csv`, `results/tables/09_work_centre_config.json` |
| Earlier work-centre row | The Phase 6 `remove_work_centre` row did not fit a second model. That row alone is not the comparison above. | `results/tables/07_ablation_summary.csv` |
| Suburb and ward error audit | Computed from the saved locked-test predictions. Groups below 200 rows do not receive class metrics. This is not a claim of geographic bias. | `results/tables/09_suburb_error_audit.csv`, `results/tables/09_ward_error_audit.csv` |
| Per-class ROC and precision-recall | Computed from the saved locked-test probabilities. The earlier macro ROC-AUC 0.724 and macro average precision 0.519 were left in `06_locked_test_summary.csv` and matched. | `results/figures/09_roc_per_class_locked_test.png`, `results/figures/09_pr_per_class_locked_test.png`, `results/tables/09_per_class_ranking_locked_test.csv` |
| Form screenshots | Four screenshots were captured from the running Streamlit page. | `evidence/screenshots/` |
| Stakeholder interview | Pending. No interview has been conducted or recorded. | `reports/stakeholder_interview_guide.md` |
| Embedding network | Lecturer approval recorded. No approval date was supplied, and the architecture was not changed. Validation macro-F1 0.481 does not beat histogram gradient boosting at 0.482, so it is not the champion. The locked test was not used. | `results/tables/05_split_metrics.csv`, `results/tables/10_embedding_champion_decision.json` |
| Logistic convergence check | The reported fit is still saga with max_iter 200. It stops at the cap with a warning. A max_iter 1000 refit also warned and had validation macro-F1 0.453, so it was not adopted and the locked test was not scored. | `results/tables/11_logistic_convergence.json` |
| Hyperparameter search | Required and completed for the boosting champion only. No repository brief requires a search for logistic regression, the perceptron, or the embedding network, so those searches were not run. | `results/tables/06_search_summary.csv`, `reports/experiment_log.md` |
| Final report PDF | Regenerated after the documentation reconciliation. | `reports/final_report.pdf` |

Commands that reproduce the new tables and figures, from the repository root, with the labelled parquet and the saved test predictions in place:

```bash
python -m src.work_centre_comparison
python -m src.geographic_error_audit
python -m src.per_class_curves
python -m src.embedding_experiment
python -m src.logistic_convergence
python -m src.logistic_conditioning
```

The work-centre command refits two models. The geographic audit and the curve command only read saved predictions. The embedding command scores the saved checkpoint and retrains beside it when `models/embedding_network.pt` already exists, so that file is not replaced. The logistic conditioning command fits two separate variants on the training period and scores the 2025 validation period. None of these commands reads the locked test.

The report PDF is printed from the Markdown file. Pandoc is not installed in this environment. The export uses headless Chrome:

```bash
python scripts/export_report_pdf.py
```
