# Submission checklist

Checked on 17 October 2026 against the files in this repository. An open item is not done.

| Item | Status | Where to look |
| --- | --- | --- |
| Frozen champion | Histogram gradient boosting, untuned refit, 12 iterations. Not replaced by later comparisons. | `results/tables/06_frozen_decision.json`, `models/frozen_resolution_pipeline.joblib` |
| Work-centre comparison | Two models were fit on the training period and scored on the 2025 validation period. The setting without work centre matches the saved champion validation macro-F1. The setting with work centre scores higher and is not the champion, because the extract has no assignment time. The locked test was not used. | `results/tables/09_work_centre_comparison.csv`, `results/tables/09_work_centre_per_class.csv`, `results/tables/09_work_centre_config.json` |
| Earlier work-centre row | The Phase 6 `remove_work_centre` row did not fit a second model. That row alone is not the comparison above. | `results/tables/07_ablation_summary.csv` |
| Suburb and ward error audit | Computed from the saved locked-test predictions. Groups below 200 rows do not receive class metrics. This is not a claim of geographic bias. | `results/tables/09_suburb_error_audit.csv`, `results/tables/09_ward_error_audit.csv` |
| Per-class ROC and precision-recall | Computed from the saved locked-test probabilities. The earlier macro ROC-AUC 0.724 and macro average precision 0.519 were left in `06_locked_test_summary.csv` and matched. | `results/figures/09_roc_per_class_locked_test.png`, `results/figures/09_pr_per_class_locked_test.png`, `results/tables/09_per_class_ranking_locked_test.csv` |
| Common ROC figure | One locked-test figure with the dummy, logistic regression, histogram gradient boosting, the multilayer perceptron, and the embedding network. Class AUCs are in the accompanying table. No curve was skipped. | `results/figures/13_roc_all_models_locked_test.png`, `results/tables/13_roc_all_models_locked_test.csv` |
| Form screenshots | Four screenshots were captured from the running Streamlit page. | `evidence/screenshots/` |
| Stakeholder interview | Not conducted. The brief does not require one. | `reports/stakeholder_interview_guide.md` |
| Embedding network | Lecturer approval recorded. No approval date was supplied, and the architecture was not changed. Validation macro-F1 0.481 does not beat histogram gradient boosting at 0.482, so it is not the champion. The locked-test score is Question 6.2 only. | `results/tables/05_split_metrics.csv`, `results/tables/10_embedding_champion_decision.json`, `results/tables/13_question62_record.json` |
| Logistic convergence check | The reported fit is still saga with max_iter 200. It stops at the cap with a warning. A max_iter 1000 refit also warned and had validation macro-F1 0.453, so it was not adopted and the locked test was not scored. | `results/tables/11_logistic_convergence.json` |
| Hyperparameter search | The brief says to optimise only the strongest validation model. That search was completed for histogram gradient boosting. Logistic regression, the perceptron, and the embedding network were not searched. | `results/tables/06_search_summary.csv`, `reports/IDA117V_Project_Template.pdf` |
| Final report PDF | Regenerated on 18 October 2026 from `reports/final_report.md` after the references and disclosure were added. 35 pages. The file opens, starts with `%PDF`, and ends with `%%EOF`. It contains the cover, POPIA, the Question 6.2 table, the common ROC table, the reference list, the AI-assistance disclosure, Appendix A, and 20 embedded images. | `reports/final_report.pdf` |
| McNemar record | Question 6.2 compares the frozen booster with the embedding network on 422,033 aligned rows. Statistic 616.8719. p-value 3.582e-136. The perceptron table, statistic 160.6875, is the historical supplementary result. | `results/tables/13_question62_record.json`, `results/tables/07_mcnemar_test.csv` |
| Clean notebook run | Notebooks 01 to 06 have an isolated execution. Notebook 07's current source was executed on 18 October 2026 against a copy of the recorded champion files and matched the saved McNemar table. The current notebook 06 source was not executed again. | `evidence/reproducibility/notebooks/execution_log.txt`, `evidence/reproducibility/notebooks/pr14_notebook07/executed/07_ablation_and_statistical_testing.ipynb` |
| Contribution statement | Mpumelelo Ngozo (MSP, 218409202, 218409202@tut4life.ac.za), Selowa Tumiso (TS, 218340172, 218340172@tut4life.ac.za), Ualwela Netshidzati (U, 220069117, 220069117@tut4life.ac.za), and Thabang Mokoena (TG, 240668823, 240668823@tut4life.ac.za) are on the statement and on the report cover. Contribution categories are “To be confirmed by the team”. Signature and date cells are empty. The brief still requires signatures. | `reports/contribution_statement.md`, `reports/final_report.md` |
| References and AI disclosure | The report cites the City dataset, POPIA, the methods that were used, and the pinned libraries. The AI section follows the git history: four human commits on 15 October 2026, and later commits by Cursor Agent. It does not invent a prompt log. | `reports/final_report.md` |
| Presentation evidence | The brief asks for 10–12 minutes, every member, an interface demonstration, and a defence, from 5 to 17 October 2026. The outline matches that request. No attendance record or defence outcome is in the repository. | `reports/presentation_outline.md` |
| Logistic convergence | The warning is retained. `converged` is false. The saved validation macro-F1 0.460268 is the comparable 200-iteration score. The model is not described as converged. | `results/tables/11_logistic_convergence.json` |
| Notebook 07 section 4 | Executed in the notebook on 18 October 2026. Displayed Question 6.2 counts and the statistic match the saved record. | `notebooks/07_ablation_and_statistical_testing.ipynb` |

Commands that reproduce the new tables and figures, from the repository root, with the labelled parquet and the saved test predictions in place:

```bash
python -m src.work_centre_comparison
python -m src.geographic_error_audit
python -m src.per_class_curves
python -m src.embedding_experiment
python -m src.logistic_convergence
python -m src.logistic_conditioning
python -m src.locked_test_comparison
```

The work-centre command refits two models. The geographic audit and the curve command only read saved predictions. The embedding command scores the saved checkpoint and retrains beside it when `models/embedding_network.pt` already exists, so that file is not replaced. The logistic conditioning command fits two separate variants on the training period and scores the 2025 validation period. `python -m src.locked_test_comparison` scores the saved embedding checkpoint on the existing locked test and writes the common ROC figure. It does not replace the champion files.

The report PDF is printed from the Markdown file. Pandoc is not installed in this environment. The export uses headless Chrome:

```bash
python scripts/export_report_pdf.py
```
