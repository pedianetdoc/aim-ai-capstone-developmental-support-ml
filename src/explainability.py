"""SHAP utilities."""
import numpy as np
import pandas as pd
import shap

def transformed_feature_names(preprocessor):
    return list(preprocessor.get_feature_names_out())

def mean_absolute_shap_by_transformed_feature(pipe, X_sample):
    """Return mean absolute SHAP for transformed features of a fitted GB pipeline."""
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    Xt = pre.transform(X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xt)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    names = transformed_feature_names(pre)
    return pd.DataFrame({
        "transformed_feature": names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
