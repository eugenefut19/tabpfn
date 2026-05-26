
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from tabpfn import TabPFNClassifier

import xgboost as xgb


def train_tabpfn(X_train, y_train, preprocessor):

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            TabPFNClassifier(
                device="cpu",
                random_state=42,
                model_path=''
            )
        )
    ])

    model.fit(X_train, y_train)

    return model


def train_logistic(X_train, y_train, preprocessor):

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
        )
    ])

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train, preprocessor):

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=42,
            )
        )
    ])

    model.fit(X_train, y_train)

    return model
