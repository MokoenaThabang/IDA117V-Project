# Presentation outline

## What the brief requires

The only presentation text in this repository is `reports/IDA117V_Project_Template.pdf`. It says presentations run from 5 October to 17 October 2026. Question 8.3 asks for a concise team presentation and a live or recorded interface demonstration, a defence of the modelling choices, and a signed contribution statement. The suggested length is 10–12 minutes plus questions, and every member participates. The lecturer named for this submission is Dr Masethe. No message from Dr Masethe in this repository changes that text.

## Team

The group name is Point Break. The lecturer is Dr Masethe. The submission date is 20 October 2026. The members named for this submission are Mpumelelo Ngozo (218409202, MSP, 218409202@tut4life.ac.za), Selowa Tumiso (218340172, TS, 218340172@tut4life.ac.za), Ualwela Netshidzati (220069117, U, 220069117@tut4life.ac.za), and Thabang Mokoena (240668823, TG, 240668823@tut4life.ac.za). Speaking parts are not assigned. The repository does not say which member presents which section. The brief asks every member to participate. This outline does not record that they did.

## Evidence that is not in the repository

There is no slide file, no attendance record, no recording, and no defence mark. Today in this workspace is 18 October 2026, after the window printed in the brief. This outline is the speaking plan. It is not evidence that the presentation happened.

## Speaking plan

About 11 minutes, inside the brief’s 10–12 minute suggestion. Every figure below is a saved project output. Scores come from `results/tables/08_model_metrics.csv`, the phase reports, or `results/tables/13_roc_all_models_locked_test.csv` and `results/tables/13_question62_record.json`. The interface demonstration is `streamlit run app/app.py`, using the cases in `reports/demonstration_checklist.md`.

## 1. Problem (1 minute)

A City of Cape Town service request is logged in SAP C3. The project estimates, at logging time, whether that request falls in Resolved Fast, Resolved Slow, or Unresolved. The intended users are municipal service departments. The intended benefit is earlier human attention. The model must not be used to deny or reduce service.

## 2. Data and labels (1 minute 30 seconds)

The extract is the deprecated public feature layer, 5,372,466 rows, created from 1 January 2023 through 6 October 2026. The City terms are custom, not a Creative Commons licence, and the raw file is not in git. After exact-duplicate removal, 4,576,802 rows are labelled at a seven-day cut. Open requests younger than seven days stay unlabelled (15,981 rows). Show `results/figures/03_01_target_distribution.png`: 64.9% Resolved Fast.

## 3. What the EDA changes (1 minute 30 seconds)

Show the monthly chart, `results/figures/03_05_monthly_request_volume.png`. January 2025 is empty. From May 2026 the Unresolved share rises because the extract is frozen. That is why the locked test stops on 30 April 2026. Mention that suburb `#` and subcouncil `Not assigned` are stored placeholders, and that work centre has no assignment time.

## 4. Leakage and the split (1 minute)

Predictors are complaint type, suburb, ward, subcouncil, weekday, month, and weekend. Completed Date, Changed on, duration, the target, identifiers, coordinates, and work centre are excluded. Train is 2023–2024, validation is 2025, and the locked test is January–April 2026. The class mix shifts: Unresolved is 4.5% of training and 21.3% of validation.

## 5. Models on validation (2 minutes)

Primary metric is macro-F1. The most-frequent dummy scores validation macro-F1 0.273. Logistic regression scores 0.460 and did not converge at 200 iterations. A 1000-iteration check also did not converge and was not adopted. Histogram gradient boosting scores 0.482. The multilayer perceptron scores 0.476. The embedding network scores 0.481. Lecturer approval has been recorded, and it still does not choose the champion, because histogram gradient boosting scores 0.482 on the same validation rule. Show `results/figures/05_validation_comparison.png`, the MLP learning curve, and the approved embedding curve `results/figures/10_embedding_learning_curves.png`. Training loss fell while validation loss rose. Do not use a figure titled as pending approval.

## 6. Tuning and the one locked-test score (1 minute 30 seconds)

Eight random trials plus the untuned settings were scored inside 2023–2024 only. Trial 5 had the highest training-fold mean, 0.544, but the 2025 validation macro-F1 was 0.470 against 0.477 for the untuned refit. The frozen model is the untuned setting with 12 iterations. Its locked-test macro-F1 is 0.494. Accuracy is 0.600. Resolved Slow precision is 0.210. Show `results/figures/06_confusion_locked_test.png`.

## 7. Ablations and McNemar (1 minute 30 seconds)

Removing suburb, ward, and subcouncil changes validation macro-F1 by −0.008. Turning class weights off changes it by −0.101, and Unresolved recall falls to 0.037. The Phase 6 work-centre row did not fit a second model. A later sensitivity fit did, on validation only, and did not replace the champion. Question 6.2 compares the frozen champion with the embedding network on the same 422,033 test rows. The continuity-corrected chi-square is 616.8719, p = 3.582e-136. The champion is correct on 5,113 more discordant cases. The macro-F1 gap, champion minus embedding network, is −0.002, and the accuracy gap is +0.012. The test does not show that one model is better on every class, and the champion stays the frozen booster. The earlier perceptron comparison, statistic 160.6875, is a supplementary result.

## 8. App, ethics, and what is unfinished (1 minute)

The Streamlit form loads the saved pipeline and does not train. The demonstration sample is not a real case. Suburb and ward error audits have been run: `results/tables/09_suburb_error_audit.csv` and `results/tables/09_ward_error_audit.csv`. Groups below 200 rows are not evidence of geographic bias. The embedding network is approved, lost on validation macro-F1, and its locked-test score is only the Question 6.2 comparison. Show the common ROC figure `results/figures/13_roc_all_models_locked_test.png` with the class-AUC table `results/tables/13_roc_all_models_locked_test.csv`. The brief does not require a stakeholder interview, and none was recorded. The outline is not an attendance record. End on the recommendation: a person reviews the estimate, and the estimate is not a reason to reduce service in an area.
