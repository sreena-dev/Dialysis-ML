"""
train_predict.py — YDF Model Training, Evaluation, and Inference
====================================================================
Hemodialysis Target UF Prediction Pipeline

This module trains a Yggdrasil Decision Forests (YDF) RandomForest model to
predict UF_Deviation (= Target UF − Weight_Difference), then reconstructs the
full Target UF at inference time.

YDF is TF-DF's standalone sibling — same algorithms (Yggdrasil backend),
but with full Windows support and a simpler API.

Architecture:
    predicted_Target_UF = Weight_Difference + model.predict(UF_Deviation)

Clinical guardrails are enforced at inference:
    1. Short-circuit: if Weight_Difference ≤ 0.1 → return 0.0 L
    2. Post-prediction floor: max(predicted_UF, 0.0)
    3. Post-prediction ceiling: min(predicted_UF, 5.0)

Author: Clinical ML Pipeline
License: Internal Use Only — Medical Device Software
"""

from __future__ import annotations
from typing import Any, cast

import os
import logging
import numpy as np
import pandas as pd
import ydf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Local imports
from preprocessing import (
    get_training_data,
    build_inference_dataframe,
    export_xlsx_to_csv,
    MODEL_FEATURES,
    TARGET_COL,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = "uf_ydf_model"
CLINICAL_UF_FLOOR = 0.0    # Liters — UF cannot be negative
CLINICAL_UF_CEILING = 5.0  # Liters — hard safety cap
WEIGHT_DIFF_THRESHOLD = 0.1  # kg — below this, no UF needed


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    csv_path: str = "raw_data.csv",
    save_dir: str = DEFAULT_MODEL_DIR,
    test_size: float = 0.2,
    random_seed: int = 42,
    num_trees: int = 300,
) -> tuple[ydf.GenericModel, dict[str, float], pd.DataFrame | None]:
    """
    Train a YDF RandomForest model to predict UF_Deviation.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV data file.
    save_dir : str
        Directory to save the trained model.
    test_size : float
        Fraction of data reserved for the test set.
    random_seed : int
        Random seed for reproducibility.
    num_trees : int
        Number of trees in the random forest.

    Returns
    -------
    model : ydf.RandomForestLearner trained model
        The trained model.
    metrics : dict
        Dictionary with evaluation metrics:
        - rmse_deviation, mae_deviation: error on UF_Deviation prediction
        - rmse_target_uf, mae_target_uf: error on reconstructed Target UF
    importance_df : pd.DataFrame
        Feature importance table sorted descending.
    """
    # ---- Load and split data ----
    X, y = get_training_data(csv_path)

    X_train, X_test, y_train, y_test = cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        cast(object, train_test_split(X, y, test_size=test_size, random_state=random_seed)),
    )
    logger.info(
        "Train/Test split: %d train, %d test (%.0f%% test)",
        len(X_train), len(X_test), test_size * 100,
    )

    # ---- Prepare training DataFrame (YDF takes pandas directly) ----
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    # ---- Build and train the model ----
    logger.info("Initialising RandomForest with %d trees...", num_trees)
    learner = ydf.RandomForestLearner(
        label=TARGET_COL,
        task=ydf.Task.REGRESSION,
        num_trees=num_trees,
        random_seed=random_seed,
    )

    model = learner.train(train_df)
    logger.info("Training complete.")

    # ---- Evaluate: Deviation Predictions ----
    logger.info("Evaluating on test set...")
    y_pred_deviation = model.predict(X_test)

    rmse_dev = float(np.sqrt(mean_squared_error(y_test, y_pred_deviation)))
    mae_dev = float(mean_absolute_error(y_test, y_pred_deviation))
    logger.info("UF_Deviation — RMSE: %.4f, MAE: %.4f", rmse_dev, mae_dev)

    # ---- Evaluate: Reconstructed Target UF ----
    # Target UF = Weight_Difference + UF_Deviation
    weight_diff_test = cast(np.ndarray[Any, Any], X_test["Weight_Difference"].values)
    actual_target_uf = weight_diff_test + cast(np.ndarray[Any, Any], y_test.values)
    predicted_target_uf = weight_diff_test + y_pred_deviation

    rmse_uf = float(np.sqrt(mean_squared_error(actual_target_uf, predicted_target_uf)))
    mae_uf = float(mean_absolute_error(actual_target_uf, predicted_target_uf))
    logger.info("Target UF (full) — RMSE: %.4f, MAE: %.4f", rmse_uf, mae_uf)

    metrics = {
        "rmse_deviation": rmse_dev,
        "mae_deviation": mae_dev,
        "rmse_target_uf": rmse_uf,
        "mae_target_uf": mae_uf,
    }

    # ---- Feature importances ----
    importance_df = _extract_feature_importances(model)
    if importance_df is not None:
        print("\n" + "=" * 50)
        print("Feature Importances")
        print("=" * 50)
        print(importance_df.to_string(index=False))
        print("=" * 50)

    # ---- Save model ----
    model_path = os.path.abspath(save_dir)
    model.save(model_path)
    logger.info("Model saved to: %s", model_path)

    return model, metrics, importance_df

# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def _extract_feature_importances(model) -> pd.DataFrame | None:
    """
    Extract and format feature importance from a trained YDF model.

    Uses variable importances from the model inspector. Tries multiple
    importance types in order of preference.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['Feature', 'Importance'], sorted descending.
    """
    try:
        # YDF exposes variable importances directly
        var_importances = model.variable_importances()

        # Try importance types in order of preference
        for importance_type in [
            "MEAN_DECREASE_IN_ACCURACY",
            "NUM_AS_ROOT",
            "SUM_SCORE",
            "MEAN_MIN_DEPTH",
            "NUM_NODES",
        ]:
            if importance_type in var_importances:
                records = var_importances[importance_type]
                
                rows = []
                for vi in records:
                    # YDF returns (score, feature_name) or objects with .score / .name
                    feature_name = getattr(vi, "name", vi[1])
                    importance_score = getattr(vi, "score", vi[0])
                    
                    rows.append({
                        "Feature": str(feature_name),
                        "Importance": float(importance_score),
                    })
                    
                df = pd.DataFrame(rows).sort_values(
                    "Importance", ascending=False
                ).reset_index(drop=True)
                
                logger.info(
                    "Feature importances extracted using '%s'.", importance_type
                )
                return df

        logger.warning("No recognised importance type found.")
        return None

    except Exception as e:
        logger.warning("Could not extract feature importances: %s", e)
        return None

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_saved_model(model_dir: str = DEFAULT_MODEL_DIR):
    """
    Load a saved YDF model from disk.

    Parameters
    ----------
    model_dir : str
        Directory containing the saved model.

    Returns
    -------
    ydf.GenericModel
        The loaded YDF model.
    """
    model_path = os.path.abspath(model_dir)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory not found: {model_path}. "
            "Train the model first using train_model()."
        )

    logger.info("Loading model from: %s", model_path)
    model = ydf.load_model(model_path)
    logger.info("Model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Clinical inference with guardrails
# ---------------------------------------------------------------------------

def load_model_and_predict(
    model,
    input_df: pd.DataFrame,
) -> dict:
    """
    Run clinical-safe inference to predict Target UF.

    Implements the full inference pipeline:
        1. SHORT-CIRCUIT: If Weight_Difference ≤ 0.1 → return 0.0 L immediately.
        2. MODEL PREDICTION: Use YDF to predict UF_Deviation.
        3. RECONSTRUCTION: Target_UF = Weight_Difference + UF_Deviation.
        4. CLINICAL CAPS: Clamp result to [0.0, 5.0] L.

    Parameters
    ----------
    model : ydf.GenericModel
        A trained YDF model (predicts UF_Deviation).
    input_df : pd.DataFrame
        Single-row DataFrame from build_inference_dataframe().

    Returns
    -------
    dict
        {
            "predicted_target_uf": float,   # Final clamped prediction (L)
            "predicted_deviation": float,    # Raw deviation prediction (L)
            "weight_difference": float,      # Pre-dialysis − Dry weight (kg)
            "short_circuited": bool,         # True if guardrail skipped model
            "capped": bool,                  # True if clinical cap was applied
            "clinical_message": str,         # Human-readable clinical note
        }

    Clinical Safety Notes
    ---------------------
    - A UF > 5.0 L in a single session risks severe intradialytic hypotension,
      cramping, and cardiac events. This hard cap is a safety mechanism.
    - Negative UF is physically nonsensical (implies adding fluid) and is floored.
    - Weight_Difference ≤ 0.1 indicates the patient is at or below dry weight;
      ultrafiltration is contraindicated unless fluid administration is planned.
    """
    result = {
        "predicted_target_uf": 0.0,
        "predicted_deviation": 0.0,
        "weight_difference": 0.0,
        "short_circuited": False,
        "capped": False,
        "clinical_message": "",
    }

    # ---- Extract Weight_Difference ----
    if "Weight_Difference" not in input_df.columns:
        raise ValueError(
            "input_df must contain 'Weight_Difference'. "
            "Use build_inference_dataframe() to construct the input."
        )

    weight_diff = float(input_df["Weight_Difference"].iloc[0])
    result["weight_difference"] = weight_diff

    # ---- GUARDRAIL 1: Short-circuit if at or below dry weight ----
    if weight_diff <= WEIGHT_DIFF_THRESHOLD:
        result["short_circuited"] = True
        result["predicted_target_uf"] = 0.0
        result["clinical_message"] = (
            "Patient is at or below target dry weight "
            f"(Weight Difference = {weight_diff:.2f} kg). "
            "Ultrafiltration is not required unless fluid administration is planned."
        )
        logger.info(
            "Short-circuit: Weight_Difference=%.2f <= %.2f. Returning UF=0.0.",
            weight_diff,
            WEIGHT_DIFF_THRESHOLD,
        )
        return result

    # ---- MODEL PREDICTION ----
    # YDF takes pandas DataFrames directly — no conversion needed
    inference_features = input_df[MODEL_FEATURES].copy()
    raw_deviation = float(model.predict(inference_features)[0])
    result["predicted_deviation"] = raw_deviation

    # ---- RECONSTRUCTION ----
    raw_target_uf = weight_diff + raw_deviation

    # ---- GUARDRAIL 2: Clinical caps ----
    final_uf = raw_target_uf
    capped = False
    message_parts = []

    if final_uf < CLINICAL_UF_FLOOR:
        final_uf = CLINICAL_UF_FLOOR
        capped = True
        message_parts.append(
            f"Predicted UF ({raw_target_uf:.2f} L) was negative. "
            f"Floored to {CLINICAL_UF_FLOOR} L."
        )

    if final_uf > CLINICAL_UF_CEILING:
        final_uf = CLINICAL_UF_CEILING
        capped = True
        message_parts.append(
            f"WARNING: Predicted UF ({raw_target_uf:.2f} L) exceeded safety limit. "
            f"Capped at {CLINICAL_UF_CEILING} L to prevent severe volume depletion."
        )

    if not capped:
        message_parts.append(
            f"Predicted Target UF: {final_uf:.2f} L "
            f"(Weight Difference: {weight_diff:.2f} + "
            f"Clinician Adjustment: {raw_deviation:.2f})."
        )

    result["predicted_target_uf"] = float(final_uf)
    result["capped"] = capped
    result["clinical_message"] = " ".join(message_parts)

    logger.info(
        "Prediction: Weight_Diff=%.2f, Deviation=%.4f, Raw_UF=%.4f, "
        "Final_UF=%.4f (capped=%s)",
        weight_diff, raw_deviation, raw_target_uf, final_uf, capped,
    )

    return result


# ---------------------------------------------------------------------------
# Reporting utility
# ---------------------------------------------------------------------------

def print_evaluation_report(metrics: dict):
    """Print a formatted evaluation summary."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  UF_Deviation Prediction:")
    print(f"    RMSE : {metrics['rmse_deviation']:.4f} L")
    print(f"    MAE  : {metrics['mae_deviation']:.4f} L")
    print(f"\n  Reconstructed Target UF:")
    print(f"    RMSE : {metrics['rmse_target_uf']:.4f} L")
    print(f"    MAE  : {metrics['mae_target_uf']:.4f} L")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """
    Full pipeline: export data → train → evaluate → sample prediction.
    """
    print("=" * 60)
    print("Hemodialysis Target UF Prediction Pipeline (YDF)")
    print("=" * 60)

    # ---- Step 0: Ensure raw_data.csv exists ----
    csv_path = "raw_data.csv"
    if not os.path.exists(csv_path):
        logger.info("raw_data.csv not found. Exporting from Excel...")
        try:
            csv_path = export_xlsx_to_csv()
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.error(
                "Please place 'Hemodialysis Treatment Data.xlsx' in the "
                "working directory or provide 'raw_data.csv' directly."
            )
            return

    # ---- Step 1: Train model ----
    model, metrics, importance_df = train_model(csv_path=csv_path)

    # ---- Step 2: Print evaluation ----
    print_evaluation_report(metrics)

    # ---- Step 3: Sample inference ----
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS")
    print("-" * 60)

    # Normal case: patient 3 kg above dry weight
    sample_df = build_inference_dataframe(
        age=60, sex=0, pre_weight=75.0, dry_weight=72.0,
        sbp=130, dbp=80, hr=75
    )
    result = load_model_and_predict(model, sample_df)
    print(f"\n  Case 1 (Normal — 3 kg over dry weight):")
    print(f"    {result['clinical_message']}")
    print(f"    Final Target UF: {result['predicted_target_uf']:.2f} L")

    # Edge case: patient at dry weight (should short-circuit)
    sample_df_low = build_inference_dataframe(
        age=55, sex=1, pre_weight=70.0, dry_weight=70.0,
        sbp=120, dbp=70, hr=72
    )
    result_low = load_model_and_predict(model, sample_df_low)
    print(f"\n  Case 2 (At dry weight — should short-circuit):")
    print(f"    {result_low['clinical_message']}")
    print(f"    Final Target UF: {result_low['predicted_target_uf']:.2f} L")
    print(f"    Short-circuited: {result_low['short_circuited']}")

    # Edge case: high weight gain
    sample_df_high = build_inference_dataframe(
        age=70, sex=0, pre_weight=85.0, dry_weight=72.0,
        sbp=180, dbp=100, hr=95
    )
    result_high = load_model_and_predict(model, sample_df_high)
    print(f"\n  Case 3 (High weight gain — 13 kg, may be capped):")
    print(f"    {result_high['clinical_message']}")
    print(f"    Final Target UF: {result_high['predicted_target_uf']:.2f} L")
    print(f"    Capped: {result_high['capped']}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
