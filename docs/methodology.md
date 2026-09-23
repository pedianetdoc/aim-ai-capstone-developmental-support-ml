# Methodology Summary

## Population
Children ages 3–5 from the 2024 NSCH topical public-use dataset.

## Outcome
Binary composite: `recognized_dev_concern_support`.

Positive if at least one qualifying developmental/behavioral diagnosis, treatment/support indicator, early-intervention/special-education support indicator, or related qualifying survey item was present.

This target reflects **recognized concern/support**, not latent developmental need.

## Model A
102 eligible predictors spanning demographic/perinatal, developmental, medical, ACE, routine, nutrition, and family/context domains.

## Model B
38 domain-informed predictors selected for developmental relevance and practical observability.

## Model C
44 predictors retained from an exploratory SHAP-ranked feature-count experiment.

## Preprocessing
- 80/20 split stratified by age + outcome
- numeric: median imputation + standardization
- categorical: most-frequent imputation + one-hot encoding
- preprocessing performed inside sklearn pipelines

## Evaluation
Primary metrics:
- ROC-AUC
- PR-AUC
- F1
- sensitivity
- precision
- specificity
- accuracy

Threshold analyses were used to show the trade-off between sensitivity and the number of children flagged.

## Explainability
SHAP was used for feature contribution analysis.

## Fairness
Subgroup metrics were examined by sex and age. At threshold 0.30, male sensitivity was approximately 0.633 and female sensitivity approximately 0.473.

## Important caveats
- No clinical deployment claim
- U.S. survey dataset
- outcome may encode recognition/access bias
- Model C feature selection was exploratory and not fully nested
- external and prospective validation required
