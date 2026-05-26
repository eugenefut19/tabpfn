from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(
    X_train,
    max_categories,
):

    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_cols = X_train.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    # =========================
    # CATEGORICAL PIPELINE
    # =========================

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                max_categories=max_categories,
            )
        )
    ])

    # =========================
    # NUMERIC PIPELINE
    # =========================

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ])

    # =========================
    # FINAL PREPROCESSOR
    # =========================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                categorical_pipeline,
                categorical_cols,
            ),
            (
                "num",
                numeric_pipeline,
                numeric_cols,
            )
        ]
    )

    return preprocessor
