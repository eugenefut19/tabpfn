import pandas as pd
import numpy as np
import shap
import xgboost as xgb


def logistic_feature_importance(model):

    classifier = model.named_steps["classifier"]

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": classifier.coef_[0],
    })

    df["abs_coef"] = df["coefficient"].abs()

    return df.sort_values(
        by="abs_coef",
        ascending=False,
    )


def xgb_feature_importance(model):

    classifier = model.named_steps["classifier"]

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_,
    })

    return df.sort_values(
        by="importance",
        ascending=False,
    )



def shap_feature_importance(
    surrogate_model,
    X_processed,
    feature_names,
    sample_size=300,
):

    import pandas as pd
    import numpy as np
    import shap

    # Smaller sample for speed
    X_sample = X_processed[:sample_size]

    # Build explainer
    explainer = shap.TreeExplainer(
        surrogate_model
    )

    # Compute shap values
    shap_values = explainer.shap_values(
        X_sample
    )

    # Mean absolute SHAP importance
    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    # Build dataframe
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    })

    # Sort descending
    shap_df = shap_df.sort_values(
        by="mean_abs_shap",
        ascending=False,
    )

    return shap_df

def build_tabpfn_surrogate(
    tabpfn_model,
    X_train,
    train_probs,
):

    import xgboost as xgb

    X_processed = tabpfn_model.named_steps[
        "preprocessor"
    ].transform(X_train)

    surrogate_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )

    surrogate_model.fit(
        X_processed,
        train_probs,
    )

    return surrogate_model, X_processed
