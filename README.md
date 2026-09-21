# AIM AI Capstone: Predicting Recognized Developmental Concern or Support

This repository contains the reproducible code, technical materials, report, and presentations for an AIM AI/ML capstone project using the **2024 U.S. National Survey of Children's Health (NSCH)**.

## Project question

Can supervised machine learning identify patterns associated with **recognized developmental concern or support** among children ages 3–5, and can a reduced feature set retain comparable predictive performance?

> **Important:** This is a research proof of concept. The target reflects *recognized developmental concern or support* in survey data. It is **not a diagnostic model**, does not estimate latent "true need," and is not intended for clinical deployment.

## Headline results

Three Gradient Boosting feature sets were compared using the same 5-fold cross-validation framework:

| Model | Features | ROC-AUC | PR-AUC | F1 | Precision | Sensitivity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model A: Full | 102 | 0.782 | 0.662 | 0.510 | 0.772 | 0.381 | 0.802 |
| Model B: Domain-informed | 38 | 0.766 | 0.643 | 0.500 | 0.759 | 0.373 | 0.798 |
| Model C: Data-driven | 44 | 0.785 | 0.664 | 0.518 | 0.770 | 0.390 | 0.803 |

Model C used fewer than half of the original predictors while matching or slightly exceeding Model A in exploratory cross-validation.

A fairness audit also identified a notable sex-related sensitivity gap at a common threshold of 0.30:

- Male sensitivity: 0.633
- Female sensitivity: 0.473

Because the target itself reflects historical recognition and support patterns, this difference may partly reflect bias already present in the labels. Future work should explicitly investigate and mitigate such disparities.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── fairness.py
│   └── explainability.py
├── notebooks/
│   ├── 01_end_to_end_capstone.ipynb
│   └── 02_technical_presentation.ipynb
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── models/
│   └── README.md
├── reports/
│   └── final_report.md
├── presentations/
│   ├── AIM_AI_Capstone_Business_Deck.pptx
│   ├── AIM_AI_Capstone_Business_Deck.pdf
│   ├── AIM_AI_Capstone_Technical_Deck.pptx
│   └── AIM_AI_Capstone_Technical_Deck.pdf
└── docs/
    ├── methodology.md
    └── data_dictionary_scope.md
```

## Reproducing the analysis

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd aim-ai-capstone-developmental-support-ml
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain the NSCH dataset

Download the **2024 NSCH Topical Public Use File** from the official NSCH data source and place the `.dta` file at:

```text
data/raw/nsch_2024e_topical.dta
```

The raw dataset is intentionally not committed to this public repository.

### 5. Run the notebook

Open:

```text
notebooks/01_end_to_end_capstone.ipynb
```

The notebook reproduces the main analytic workflow using the reusable functions in `src/`.

## Reproducibility notes

- Random state: `42`
- Train/test split: 80/20, stratified by age + outcome
- Model comparison: 5-fold cross-validation
- Numeric preprocessing: median imputation + standardization
- Categorical preprocessing: most-frequent imputation + one-hot encoding
- Final feature-set comparison: Gradient Boosting with the tuned configuration documented in `src/modeling.py`

The later SHAP-ranked Model C analysis was exploratory because the ranking was derived on the training set before subset cross-validation. A publication-grade confirmatory analysis should use nested feature selection or independent external validation.

## Ethical and clinical limitations

This project:
- does not diagnose developmental conditions;
- does not establish latent developmental need;
- uses a U.S. population survey and has not been validated for Philippine children;
- may reproduce historical recognition and access biases encoded in the outcome;
- requires prospective validation, threshold selection tied to downstream actions, and explicit fairness criteria before any clinical use.

See `reports/final_report.md` for the full discussion.

## Capstone deliverables

- Reproducible code: `src/` and `notebooks/`
- Final report: `reports/final_report.md`
- Technical presentation: `notebooks/02_technical_presentation.ipynb`
- Business presentation: `presentations/AIM_AI_Capstone_Business_Deck.pptx`
