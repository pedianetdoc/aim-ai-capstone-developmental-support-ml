"""Model definitions and the tuned final Gradient Boosting pipeline."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .preprocessing import make_preprocessor

def candidate_models():
    """Representative supervised classifiers evaluated in the capstone."""
    return {
        "Logistic Regression L1": LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=5000, random_state=RANDOM_STATE
        ),
        "Logistic Regression L2": LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=5000, random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(),
    }

def tuned_gradient_boosting():
    """Best Gradient Boosting configuration from the capstone search."""
    return GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )

def make_gb_pipeline(features):
    return Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", tuned_gradient_boosting()),
    ])
