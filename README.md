# Beyond Aggregate Calibration: Decomposing Income-Conditional Recall Disparities in Automated Credit Default Prediction

**Paper:** [arXiv:2608.08202](https://arxiv.org/abs/2608.08202)

This repository contains the 17-script experimental pipeline used to reproduce the empirical results, statistical tests, and mechanism decompositions reported in the study.

## A.1 Execution Pipeline and Script Inventory

* **Data Ingestion & Exploratory Baseline (`01_inspect_data.py` to `06_fairness_audit.py`):** 
  Ingests the LendingClub dataset, applies constant imputation, and establishes the 4-feature minimal baseline. `06_fairness_audit.py` conducts the initial exploratory confidence-threshold audit across income brackets.
* **Sequential Feature Blinding & CV Evaluation (`07_robust_baseline_fixed.py` to `09_blind_model_audit.py`):** 
  Trains the Full, Income-Blind, and Double-Blind XGBoost classifiers across stratified 5-fold cross-validation.
* **SHAP Mechanism & Composition Analysis (`10_shap_mechanism.py` to `13b_monotonicity_deep_dive.py`):** 
  Transitions mechanism analysis to strict out-of-sample TreeSHAP attributions. Executes the loan-amount decile decomposition to confirm composition artifacts.
* **Architectural Robustness (`14_model_robustness.py`):** 
  Replaces the decision-tree algorithm with a regularized logistic regression pipeline to confirm proxy effects are model-agnostic.
* **Strict Holdout Replication (`15_holdout_validation.py` & `16_oos_mechanism_shap.py`):** 
  Confirms the headline Equal Opportunity recall gaps and extracts strict out-of-sample SHAP attributions on the untouched 20% holdout test partition.
* **Statistical Formalization (`17_final_methods_checks.py`):** 
  Executes the formal chi-square test of independence, two-proportion Z-tests, Spearman rank correlation matrix, and the unlogged data loss coercion audit.

## A.2 Software Dependencies

All modeling, statistical testing, and SHAP attributions were executed in Python 3.11 using the following open-source packages:
* `pandas` (2.1.0+)
* `numpy` (1.26.0+)
* `scikit-learn` (1.3.0+)
* `xgboost` (2.0.0+)
* `shap` (0.43.0+)
* `statsmodels` (0.14.0+)
* `scipy` (1.11.0+)

## A.3 Determinism and Random Seeding

To ensure exact bit-for-bit reproducibility across runs, a global seed of `random_state=42` is applied across all scikit-learn dataset partitioning functions (`train_test_split`, `StratifiedKFold`), imputer transformers, and XGBoost estimator initializations.

## A.4 Data Availability

The raw LendingClub accepted loans dataset (2007–2018) used in this analysis is publicly available on Kaggle:
[All Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
