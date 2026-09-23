"""Preprocessing helpers."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def infer_numeric_categorical(features):
    """Use the capstone convention: only known count/age variables are numeric."""
    numeric_candidates = {"sc_age_years", "hhcount", "famcount", "totkids_r"}
    numeric = [f for f in features if f in numeric_candidates]
    categorical = [f for f in features if f not in numeric]
    return numeric, categorical

def make_preprocessor(features):
    numeric, categorical = infer_numeric_categorical(features)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
