# Submission readiness

Checked on 18 October 2026 against the files in this repository and pull request #14. This note does not say the assignment is complete. No IDA117V marking brief or rubric is stored in the repository, and a search did not recover one. The rows below are the requirements this project has treated as its own deliverables.

## Verified

| Item | Result | Path |
| --- | --- | --- |
| Frozen champion | Histogram gradient boosting, untuned refit, 12 iterations, locked-test macro-F1 0.493897. The locked test was not used to choose the setting. | `results/tables/06_frozen_decision.json` |
| Comparator | Multilayer perceptron. Phase 5 validation macro-F1 0.476466. The embedding network is excluded because it is not scored on the locked test. | `results/tables/06_frozen_decision.json` |
| Split identifier | `chronological\|train_end=2024-12-31\|validation_end=2025-12-31\|test_end=2026-04-30\|seed=42` | `results/tables/07_comparison_record.json` |
| Prediction alignment | 422,033 rows. Champion file and paired file share `source_index`, order, labels, and champion predictions. | `results/tables/06_test_predictions.parquet`, `results/tables/07_paired_predictions.parquet` |
| McNemar | Both correct 233,776. Champion only 19,282. Comparator only 21,854. Both wrong 147,121. Statistic 160.6875. p-value 8.006e-37. Continuity-corrected chi-square. The test compares paired error outcomes. | `results/tables/07_mcnemar_test.csv`, `results/tables/07_comparison_record.json` |
| Embedding curve | Report image is `results/figures/10_embedding_learning_curves.png`. The notebook file `results/figures/05_embedding_learning_curves.png` has the same bytes and the title "Approved embedding network". The presentation outline cites the corrected file. | `reports/final_report.md`, `reports/presentation_outline.md` |
| Per-class curves | ROC and precision-recall plots exist. Resolved Slow average precision is 0.216. Macro ROC-AUC 0.724 and macro average precision 0.519 match the locked-test summary. | `results/figures/09_roc_per_class_locked_test.png`, `results/figures/09_pr_per_class_locked_test.png`, `results/tables/09_per_class_ranking_locked_test.csv` |
| Suburb and ward audit | 7,806 suburb rows and 119 ward rows. | `results/tables/09_suburb_error_audit.csv`, `results/tables/09_ward_error_audit.csv` |
| Exploratory figures | Figures 1 to 6 are present. | `results/figures/03_01_target_distribution.png` through `results/figures/03_06_resolution_duration.png` |
| Final PDF | Regenerated from the Markdown source. 27 pages, 19 embedded images, 2,528,380 bytes. It opens and contains the champion, the non-convergence statement, the missing interview, and the McNemar statistic. | `reports/final_report.pdf` |
| Tests | `python -m pytest tests -q`: 80 passed, 1 existing date-parsing warning. | `tests/` |

## Notebook execution

| Notebook | Clean-environment result |
| --- | --- |
| 01 to 04 | Finished in `evidence/reproducibility/notebooks/isolated/`. Original notebooks were not replaced. |
| 05 | The first isolated run failed on `PENDING_EMBEDDING_NOTE`. A later run finished after that import was removed. |
| 06 | The isolated run finished. Locked-test macro-F1 matched the saved champion. That run still names `embedding_network` as the comparator because it used the earlier selector. The current rule, applied to that run's Phase 5 table, selects the multilayer perceptron. The current notebook 06 source was not executed again. The search and the locked-test score were already reproduced, so that experiment was not repeated. |
| 07 | The earlier isolated run stopped because the frozen comparator was the embedding network. On 18 October 2026 the current notebook was executed from a copy of the recorded champion files into `evidence/reproducibility/notebooks/pr14_notebook07/`. It finished with no error outputs. Paired predictions and the McNemar table matched the saved files exactly. Fit times differed. `results/tables/06_test_predictions.parquet` was not rewritten. Its timestamp remains 17 October 2026. |
| 08 | Executed earlier. It checks the saved pipeline and does not refit. |

The command that checks the saved comparison without fitting a model is `python -m src.model_comparison`.

## Limitations that remain

- Logistic regression did not converge. The reported fit is saga, max_iter 200, and it stops with a convergence warning. A 1000-iteration refit also warned and was not adopted. The no-intercept and variance-scaled checks also warned and were not adopted. The report does not claim a converged logistic solution. The record is `results/tables/11_logistic_convergence.json` and `results/tables/12_logistic_conditioning.json`.
- No stakeholder interview has been conducted. `reports/stakeholder_interview_guide.md` is still a blank checklist.
- The embedding network is not the champion. Validation macro-F1 is 0.480687, below histogram gradient boosting at 0.481660. Its metrics file has train and validation rows only. It has no locked-test score.
- The team contribution form has no names, student numbers, or signatures.
- There is no slide file. `reports/presentation_outline.md` is a speaking outline.
- The repository does not contain the assignment brief or a marking rubric, so this checklist cannot certify that every official mark has been met.

## Still needed before a claim of completion

1. A dated stakeholder interview note, if the assignment requires one. Do not invent it.
2. A completed `reports/team_contribution_template.md`, filled by the people who did the work.
3. The official brief or rubric, so any requirement that is not already in this repository can be checked.
4. A slide file, if the brief asks for one. The outline is not a slide deck.
5. Optional: one more isolated run of the current notebook 06, only if a marker requires the frozen-decision file itself to be rewritten by that notebook. The champion metrics are already reproduced, and the current comparator rule selects the multilayer perceptron from the saved Phase 5 table.
