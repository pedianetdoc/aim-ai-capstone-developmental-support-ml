# Final Report

## AIM AI/ML Capstone Project

### Predicting Recognized Developmental Concern or Support in Preschool Children Using Machine Learning

---

## Step 1: Problem Understanding and Framing

### Project Domain
**Healthcare**

This project explores whether machine learning can help identify young children who have a recognized developmental concern or are already receiving developmental, behavioral, educational, or related support.

The project is intended as a proof-of-concept for future developmental decision-support research. It is not designed to diagnose developmental conditions or replace professional developmental assessment.

### Problem Statement

Developmental concerns in early childhood may be recognized through different pathways, including diagnosis, developmental or behavioral services, early intervention, special education, mental health treatment, or other forms of support. Recognition may occur only after concerns have become more apparent or after families have already entered the healthcare or educational system.

The primary modeling question was:

**Can supervised machine learning identify children ages 3 to 5 who meet a composite outcome of recognized developmental concern or support using variables available in the 2024 U.S. National Survey of Children's Health?**

A secondary question was whether a smaller and more interpretable feature set could retain comparable predictive performance. This was examined using both domain-informed and data-driven feature selection.

### Data Science Task

This is a **supervised binary classification** problem.

The target is **Recognized Developmental Concern or Support**.

The positive class includes children with at least one qualifying indicator of a developmental/behavioral/learning/communication diagnosis, developmental or behavioral treatment/service use, early-intervention or special-education support, or another qualifying developmental-support indicator.

The target does not represent confirmed latent developmental need or a definitive developmental diagnosis. It represents whether developmental concern or support has been recognized within the survey data.

### Evaluation Metrics

Because the positive class represented approximately 27.1% of the sample, accuracy alone was insufficient. Evaluation emphasized ROC-AUC, PR-AUC, F1, sensitivity, precision, specificity, and accuracy. Calibration and threshold-specific performance were also examined.

### Practical Success Criteria

A future decision-support system would need to identify a meaningful proportion of children who may warrant review, avoid unmanageable false-positive burden, maintain acceptable subgroup performance, use feasible features, remain interpretable, and avoid reproducing recognition disparities.

---

## Step 2: Data Collection and Understanding

### Data Source

The project used the **2024 U.S. National Survey of Children's Health (NSCH) Topical Public Use File**, a publicly available survey dataset.

The full dataset contained **51,375 observations and 457 variables**. The analytic cohort was restricted to children ages 3–5, resulting in **7,485 children**.

### Analytic Sample

- Age 3: 2,462
- Age 4: 2,489
- Age 5: 2,534
- Total: 7,485

Outcome:
- Negative: 5,455 (72.9%)
- Positive: 2,030 (27.1%)

### Candidate Predictors

Model A used **102 candidate predictors** covering child demographic/perinatal characteristics, communication/language, learning/attention, executive function/self-regulation, social-emotional functioning, motor skills, medical history, adverse childhood experiences, routines, nutrition, and family/caregiver context.

Of the 102 initial predictors:
- 4 were treated as numerical
- 98 were treated as categorical

### Missingness, Duplicates, and Outliers

- Maximum predictor missingness: **4.07%**
- No predictor exceeded 10% missingness
- No exact duplicate rows were identified

Traditional outlier removal was not emphasized because most predictors were categorical survey responses. Valid extreme count values were retained.

### Data Dictionary

The official NSCH variable list is the authoritative data dictionary for variable definitions, allowed values, and skip logic. A project-specific dictionary is represented by the exact feature lists in `src/config.py`.

---

## Step 3: Data Preprocessing, Applied EDA, and Feature Engineering

Detailed code, outputs, and figures are provided in the accompanying reproducible notebook.

### Data Cleaning and Preprocessing

- 80/20 train-test split, stratified by age + outcome
- Numeric variables: median imputation + standardization
- Categorical variables: most-frequent imputation + one-hot encoding
- Preprocessing learned within sklearn pipelines to reduce leakage

### EDA

EDA examined class distribution, age distribution, missingness, age-by-outcome strata, predictor structure, and subgroup patterns relevant to later evaluation.

### Feature Selection

Three feature sets were developed:
- **Model A:** 102 broad eligible predictors
- **Model B:** 38 domain-informed predictors
- **Model C:** 44 data-driven SHAP-ranked predictors

### Explainability

SHAP was used to estimate predictor contribution. Strong contributors included age, clear expression, distractibility, storytelling, focus, calming down, and other developmental variables.

### Dimensionality Reduction

PCA of the transformed Model B feature space showed substantial overlap between outcome groups, suggesting the composite outcome was not represented by a simple low-dimensional separation.

---

## Step 4: Model Implementation

Multiple supervised classifiers were evaluated:
- Logistic Regression L1
- Logistic Regression L2
- Linear SVM
- Random Forest
- Gradient Boosting
- XGBoost
- K-Nearest Neighbors

Models were evaluated using cross-validation, and leading candidates underwent hyperparameter tuning.

### Held-Out Model A Result

Tuned Gradient Boosting:
- ROC-AUC: 0.784
- PR-AUC: 0.650
- Accuracy: approximately 0.796
- Sensitivity at threshold 0.50: 0.369
- Specificity: 0.955
- Precision: 0.754

The default 0.50 threshold was conservative for a screening-oriented use case.

### Reduced Model Comparison

| Model | Features | ROC-AUC | PR-AUC | F1 | Precision | Sensitivity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model A: Full | 102 | 0.782 | 0.662 | 0.510 | 0.772 | 0.381 | 0.802 |
| Model B: Domain-informed | 38 | 0.766 | 0.643 | 0.500 | 0.759 | 0.373 | 0.798 |
| Model C: Data-driven | 44 | 0.785 | 0.664 | 0.518 | 0.770 | 0.390 | 0.803 |

Model C achieved the strongest overall exploratory cross-validated performance while using fewer than half of the original predictors.

---

## Step 5: Critical Thinking, Ethical AI, and Bias Auditing

### Model Explainability

SHAP was used to estimate feature contributions. SHAP importance was interpreted as predictive contribution rather than causation.

### Class Imbalance

The positive class represented 27.1% of the sample. Evaluation therefore emphasized PR-AUC, sensitivity, precision, and F1 in addition to ROC-AUC and accuracy.

### Leakage and Overfitting

Preprocessing was performed within modeling pipelines. Target-defining variables and obvious downstream derivatives were excluded from predictors.

The Model C SHAP-ranking experiment was exploratory because feature ranking and subset evaluation both used the training dataset. A future confirmatory analysis should use nested feature selection or independent external validation.

### Fairness Audit

Fairness analysis examined subgroup performance by age and sex.

At threshold 0.30:

**Male**
- Sensitivity: 0.633
- Specificity: 0.765
- Flagged rate: 36.3%

**Female**
- Sensitivity: 0.473
- Specificity: 0.887
- Flagged rate: 19.2%

The approximately **16 percentage-point sensitivity gap** is an important fairness signal.

Because the outcome reflects recognized concern and support rather than latent need, historical recognition and service-access disparities may already be embedded in the labels. A model trained on these labels could therefore reproduce existing disparities.

### Proposed Mitigations

Future work should consider:
- prospective and external validation;
- outcome labels closer to developmental need rather than recognition alone;
- subgroup monitoring of true-positive and false-positive rates;
- reweighting or resampling where appropriate;
- threshold and post-processing analysis;
- domain-expert review of retained variables; and
- explicit fairness criteria during model selection.

---

## Overall Conclusion

This proof-of-concept demonstrates that machine learning can identify meaningful patterns associated with recognized developmental concern or support using population survey data.

A 44-feature data-driven model retained and slightly improved exploratory predictive performance compared with the full 102-feature model, while the domain-informed model captured many of the same developmental signals.

However, threshold behavior, explainability, and subgroup analysis demonstrate why aggregate predictive performance alone is insufficient for clinical translation.

The model should be treated as a **research proof of concept**, not a deployable screening or diagnostic instrument. Future work should include prospective local validation, stronger outcome labels, nested or external feature-selection validation, and explicit fairness safeguards.
