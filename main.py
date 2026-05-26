
import pandas as pd

from sklearn.model_selection import train_test_split

from config import Config
from pipelines.preprocessing import build_preprocessor
from pipelines.training import (
    train_tabpfn,
    train_logistic,
    train_xgboost,
)
from pipelines.calibration import (
    calibrate_probabilities,
    find_best_threshold,
)
from pipelines.evaluation import (
    evaluate_model,
    print_results,
)
from pipelines.explainability import (
    build_tabpfn_surrogate,
    logistic_feature_importance,
    xgb_feature_importance,
    shap_feature_importance,
)


def main():

    df = pd.read_csv(Config.DATA_PATH)

    X = df.drop(columns=[Config.TARGET_COLUMN])
    y = df[Config.TARGET_COLUMN]

    # =========================
    # SPLITS
    # =========================

    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_pool,
        y_train_pool,
        test_size=Config.VALIDATION_SIZE,
        stratify=y_train_pool,
        random_state=Config.RANDOM_STATE,
    )

    # Context Sampling
    if Config.CONTEXT_ROWS < len(X_train):

        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=Config.CONTEXT_ROWS,
            stratify=y_train,
            random_state=Config.RANDOM_STATE,
        )

    print(f"Using {len(X_train)} contextual rows")

    # =========================
    # PREPROCESSING
    # =========================

    preprocessor = build_preprocessor(
        X_train,
        Config.MAX_CATEGORIES,
    )

    # =========================
    # TRAIN MODELS
    # =========================

    tabpfn_model = train_tabpfn(
        X_train,
        y_train,
        preprocessor,
    )

    logistic_model = train_logistic(
        X_train,
        y_train,
        preprocessor,
    )

    xgb_model = train_xgboost(
        X_train,
        y_train,
        preprocessor,
    )

    # =========================
    # TABPFN CALIBRATION
    # =========================

    val_probs, calibrator = calibrate_probabilities(
        tabpfn_model,
        X_val,
        y_val,
        Config.BATCH_SIZE,
    )

    best_threshold = find_best_threshold(
        y_val,
        val_probs,
    )

    print(f"Best Threshold: {best_threshold}")

    # =========================
    # EVALUATE
    # =========================

    tabpfn_train = evaluate_model(
        tabpfn_model,
        X_train,
        y_train,
        threshold=best_threshold,
        batch_size=Config.BATCH_SIZE,
        calibrator=calibrator,
        model_name="TabPFN Train",
    )

    tabpfn_test = evaluate_model(
        tabpfn_model,
        X_test,
        y_test,
        threshold=best_threshold,
        batch_size=Config.BATCH_SIZE,
        calibrator=calibrator,
        model_name="TabPFN Test",
    )

    logistic_train = evaluate_model(
        logistic_model,
        X_train,
        y_train,
        threshold=0.5,
        model_name="Logistic Train",
    )

    logistic_test = evaluate_model(
        logistic_model,
        X_test,
        y_test,
        threshold=0.5,
        model_name="Logistic Test",
    )

    xgb_train = evaluate_model(
        xgb_model,
        X_train,
        y_train,
        threshold=0.5,
        model_name="XGBoost Train",
    )

    xgb_test = evaluate_model(
        xgb_model,
        X_test,
        y_test,
        threshold=0.5,
        model_name="XGBoost Test",
    )

    results_df = pd.DataFrame([
        tabpfn_train["metrics"],
        tabpfn_test["metrics"],
        logistic_train["metrics"],
        logistic_test["metrics"],
        xgb_train["metrics"],
        xgb_test["metrics"],
    ])

    print_results(results_df)

    # =========================
    # FEATURE IMPORTANCE
    # =========================

    print("\nLOGISTIC FEATURE IMPORTANCE")
    print(
        logistic_feature_importance(logistic_model).head(15)
    )

    print("\nXGBOOST FEATURE IMPORTANCE")
    print(
        xgb_feature_importance(xgb_model).head(15)
    )

    print("\nBUILDING TABPFN SURROGATE")



    surrogate_model, X_processed = build_tabpfn_surrogate(
    tabpfn_model,
    X_train,
    tabpfn_train["probs"],
    )

    feature_names = tabpfn_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    shap_df = shap_feature_importance(
        surrogate_model=surrogate_model,
        X_processed=X_processed,
        feature_names=feature_names,
    )

    print("\nTABPFN SHAP IMPORTANCE")
    print(shap_df.head(15))


if __name__ == "__main__":
    main()
