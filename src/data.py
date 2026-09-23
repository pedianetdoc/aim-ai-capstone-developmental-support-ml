"""Data loading, cohort construction, and target creation."""
from pathlib import Path
import pandas as pd

from .config import DIAGNOSIS_VARS, SUPPORT_VARS, TARGET

def load_nsch(path):
    """Load the NSCH Stata public-use file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for setup instructions."
        )
    return pd.read_stata(path, convert_categoricals=False)

def restrict_age_3_to_5(df):
    """Restrict to children ages 3, 4, or 5."""
    return df.loc[df["sc_age_years"].isin([3, 4, 5])].copy()

def build_target(df):
    """Construct the binary recognized developmental concern/support composite."""
    out = df.copy()
    out[TARGET] = 0

    for var in DIAGNOSIS_VARS:
        out.loc[out[var] == 1, TARGET] = 1

    for var in SUPPORT_VARS:
        out.loc[out[var] == 1, TARGET] = 1

    out.loc[out["k4q22_r"].isin([1, 2]), TARGET] = 1
    out.loc[out["k4q23"] == 1, TARGET] = 1

    return out
