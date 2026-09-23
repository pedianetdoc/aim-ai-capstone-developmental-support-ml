# Models

Serialized model artefacts are intentionally not committed by default.

To preserve reproducibility without bloating the repository, the final model configuration is specified in:

```text
src/modeling.py
```

If you choose to save a fitted model locally, use `joblib` and place it in this directory. Model binary formats are excluded by `.gitignore`.

Example:

```python
import joblib
joblib.dump(model, "models/model_c_gradient_boosting.joblib")
```
