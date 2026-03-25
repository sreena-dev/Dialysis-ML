"""
preprocessing.py — Data Loading, Cleaning, and Feature Engineering
=====================================================================
Hemodialysis Target UF Prediction Pipeline (TF-DF)

This module handles all data preprocessing for the Target UF prediction model.
It transforms messy clinical spreadsheet data into clean, model-ready features.

Key design decision: The model predicts UF_Deviation (= Target UF − Weight_Difference),
NOT raw Target UF. This prevents the identity-function trap where the model would
simply learn that Target UF ≈ Weight Difference and ignore vital signs.

Author: Clinical ML Pipeline
License: Internal Use Only — Medical Device Software
"""

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path

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
# Columns expected in the raw spreadsheet (mapped from cols.txt)
RAW_COL_MAP = {
    "Age/Sex": "AgeSex",
    "Pre Dialysis Weight (kg)": "Pre_Dialysis_Weight",
    "Dry Weight (kg)": "Dry_Weight",
    "BP (mmHg)": "BP_Raw",
    "HR (min)": "Pre_Dialysis_Heart_Rate",
    "Target UF (kg)": "Target_UF",
}

# Features fed to the TF-DF model (after engineering)
MODEL_FEATURES = [
    "Age",
    "Sex",
    "Pre_Dialysis_Weight",
    "Weight_Difference",
    "Pre_Dialysis_SBP",
    "Pre_Dialysis_DBP",
    "Pre_Dialysis_Heart_Rate",
]

# Target column name produced by preprocessing
TARGET_COL = "UF_Deviation"

# Sex encoding map (integer-encoded for TF-DF)
SEX_ENCODE = {
    "M": 0,
    "MALE": 0,
    "F": 1,
    "FEMALE": 1,
}

# Regex patterns — compiled once for performance
_RE_AGE_SEX = re.compile(
    r"(\d+)\s*[/\-\s]\s*(M(?:ALE)?|F(?:EMALE)?)",
    re.IGNORECASE,
)
_RE_BP = re.compile(
    r"(\d{2,3})\s*[/\-]\s*(\d{2,3})",
)
_RE_BP_SBP_ONLY = re.compile(
    r"(\d{2,3})",
)

# ---------------------------------------------------------------------------
# Excel → CSV one-time export utility
# ---------------------------------------------------------------------------

def export_xlsx_to_csv(
    xlsx_path: str = "Hemodialysis Treatment Data.xlsx",
    csv_path: str = "raw_data.csv",
    force: bool = False,
) -> str:
    """
    Converts the clinical Excel workbook to a flat CSV file.

    Parameters
    ----------
    xlsx_path : str
        Path to the source .xlsx file.
    csv_path : str
        Destination path for the CSV output.
    force : bool
        If True, overwrites an existing CSV.

    Returns
    -------
    str
        The absolute path to the generated CSV.
    """
    csv_file = Path(csv_path)
    if csv_file.exists() and not force:
        logger.info("CSV already exists at %s — skipping export.", csv_file)
        return str(csv_file.resolve())

    xlsx_file = Path(xlsx_path)
    if not xlsx_file.exists():
        raise FileNotFoundError(
            f"Excel file not found: {xlsx_file.resolve()}. "
            + "Please place 'Hemodialysis Treatment Data.xlsx' in the working directory."
        )

    logger.info("Reading Excel file: %s", xlsx_file)
    df = pd.read_excel(xlsx_file, engine="openpyxl")
    df.to_csv(csv_file, index=False)
    logger.info(
        "Exported %d rows × %d cols → %s", len(df), len(df.columns), csv_file
    )
    return str(csv_file.resolve())


# ---------------------------------------------------------------------------
# Parsing helpers (robust regex-based)
# ---------------------------------------------------------------------------

def _parse_age_sex(value) -> tuple:
    """
    Extract (age: int, sex: int) from a combined 'Age/Sex' string.

    Examples
    --------
    >>> _parse_age_sex("65/M")
    (65, 0)
    >>> _parse_age_sex("72 / Female")
    (72, 1)
    >>> _parse_age_sex("bad data")
    (np.nan, np.nan)
    >>> _parse_age_sex(np.nan)
    (np.nan, np.nan)
    """
    if not isinstance(value, str):
        return (np.nan, np.nan)

    match = _RE_AGE_SEX.search(value)
    if match:
        age = int(match.group(1))
        sex_raw = match.group(2).upper()
        sex_int = SEX_ENCODE.get(sex_raw, np.nan)
        return (age, sex_int)

    return (np.nan, np.nan)


def _parse_bp(value) -> tuple:
    """
    Extract (SBP: float, DBP: float) from a combined BP string.

    Handles formats: "130/80", "130-80", "130 / 80".
    Falls back to SBP-only if no separator found (DBP → NaN).

    Examples
    --------
    >>> _parse_bp("130/80")
    (130.0, 80.0)
    >>> _parse_bp("130-80")
    (130.0, 80.0)
    >>> _parse_bp("130")
    (130.0, np.nan)
    >>> _parse_bp(np.nan)
    (np.nan, np.nan)
    """
    if not isinstance(value, str):
        # Handle numeric input (sometimes only SBP is recorded as a number)
        try:
            v = float(value)
            if not np.isnan(v):
                return (v, np.nan)
        except (TypeError, ValueError):
            pass
        return (np.nan, np.nan)

    # Try full SBP/DBP pattern first
    match = _RE_BP.search(value)
    if match:
        return (float(match.group(1)), float(match.group(2)))

    # Fallback: extract any 2–3 digit number as SBP
    match_sbp = _RE_BP_SBP_ONLY.search(value)
    if match_sbp:
        return (float(match_sbp.group(1)), np.nan)

    return (np.nan, np.nan)


# ---------------------------------------------------------------------------
# Core preprocessing pipeline
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: str = "raw_data.csv") -> pd.DataFrame:
    """
    Load raw CSV data and return a fully cleaned DataFrame.

    Steps
    -----
    1. Rename columns to standardised internal names.
    2. Parse composite fields (Age/Sex, BP) using regex.
    3. Convert all numeric fields, coercing errors to NaN.
    4. Handle missing weight data (drop rows where either weight is NaN).
    5. Compute Weight_Difference and UF_Deviation.
    6. Filter out clinical outliers and impossible values.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with all model features and `UF_Deviation`.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_file.resolve()}. "
            + "Run export_xlsx_to_csv() first or provide a valid path."
        )

    logger.info("Loading data from %s", csv_file)
    df = pd.read_csv(csv_file)
    original_rows = len(df)
    logger.info("Loaded %d rows × %d columns.", original_rows, len(df.columns))

    # ---- Step 1: Rename columns ----
    df = df.rename(columns=RAW_COL_MAP)

    # ---- Step 2: Parse Age/Sex ----
    if "AgeSex" in df.columns:
        parsed = df["AgeSex"].apply(_parse_age_sex)
        df["Age"] = parsed.apply(lambda x: x[0])
        df["Sex"] = parsed.apply(lambda x: x[1])
    else:
        logger.warning("'Age/Sex' column not found. Age and Sex will be NaN.")
        df["Age"] = np.nan
        df["Sex"] = np.nan

    # ---- Step 3: Parse Blood Pressure ----
    if "BP_Raw" in df.columns:
        parsed_bp = df["BP_Raw"].apply(_parse_bp)
        df["Pre_Dialysis_SBP"] = parsed_bp.apply(lambda x: x[0])
        df["Pre_Dialysis_DBP"] = parsed_bp.apply(lambda x: x[1])
    else:
        logger.warning("'BP (mmHg)' column not found. SBP and DBP will be NaN.")
        df["Pre_Dialysis_SBP"] = np.nan
        df["Pre_Dialysis_DBP"] = np.nan

    # ---- Step 4: Coerce numeric columns ----
    numeric_cols = [
        "Pre_Dialysis_Weight",
        "Dry_Weight",
        "Pre_Dialysis_Heart_Rate",
        "Target_UF",
        "Age",
        "Sex",
        "Pre_Dialysis_SBP",
        "Pre_Dialysis_DBP",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Step 5: Drop rows with missing weight data ----
    # CRITICAL: Weight_Difference = Pre_Dialysis_Weight − Dry_Weight
    # If either is NaN, pandas produces NaN and downstream math breaks.
    weight_cols = ["Pre_Dialysis_Weight", "Dry_Weight"]
    missing_weight_mask = df[weight_cols].isna().any(axis=1)
    n_dropped_weight = missing_weight_mask.sum()
    if n_dropped_weight > 0:
        logger.warning(
            "Dropping %d rows with missing Pre_Dialysis_Weight or Dry_Weight.",
            n_dropped_weight,
        )
    df = df[~missing_weight_mask].copy()

    # ---- Step 6: Drop rows with missing Target UF ----
    if "Target_UF" in df.columns:
        missing_target = df["Target_UF"].isna()
        n_dropped_target = missing_target.sum()
        if n_dropped_target > 0:
            logger.warning(
                "Dropping %d rows with missing Target_UF.", n_dropped_target
            )
        df = df[~missing_target].copy()
    else:
        raise ValueError(
            "Target_UF column not found after renaming. "
            + "Check that the raw CSV has a 'Target UF (kg)' column."
        )

    # ---- Step 7: Feature engineering ----
    df["Weight_Difference"] = df["Pre_Dialysis_Weight"] - df["Dry_Weight"]

    # CORE DESIGN: UF_Deviation = Target_UF − Weight_Difference
    # The model learns only the *clinician adjustment* beyond the simple
    # weight-based estimate, forcing it to value vital signs.
    df["UF_Deviation"] = df["Target_UF"] - df["Weight_Difference"]

    # ---- Step 8: Clinical Bounds Filtering (Outlier Removal) ----
    # This prevents typos in the raw data from destroying the RMSE.
    
    # 1. Filter impossible weights and targets
    valid_target = (df["Target_UF"] >= 0) & (df["Target_UF"] <= 7.0) # >7L is usually a typo
    valid_weight_diff = (df["Weight_Difference"] >= -5.0) & (df["Weight_Difference"] <= 15.0)
    
    # 2. Filter impossible vitals (if present)
    valid_sbp = df["Pre_Dialysis_SBP"].isna() | ((df["Pre_Dialysis_SBP"] >= 50) & (df["Pre_Dialysis_SBP"] <= 250))
    valid_hr = df["Pre_Dialysis_Heart_Rate"].isna() | ((df["Pre_Dialysis_Heart_Rate"] >= 30) & (df["Pre_Dialysis_Heart_Rate"] <= 200))

    # 3. Filter extreme calculated deviations
    # A deviation > 4L means the clinician ordered a UF 4 Liters away from the 
    # patient's actual weight gain. This is almost certainly a data entry error.
    valid_deviation = (df["UF_Deviation"] >= -4.0) & (df["UF_Deviation"] <= 4.0)

    # Apply masks
    clinical_mask = valid_target & valid_weight_diff & valid_sbp & valid_hr & valid_deviation
    n_dropped_outliers = (~clinical_mask).sum()
    
    if n_dropped_outliers > 0:
        logger.warning(
            "Dropped %d rows containing physically impossible clinical outliers.", 
            n_dropped_outliers
        )
    df = df[clinical_mask].copy()

    logger.info(
        "Preprocessing complete: %d / %d rows retained.",
        len(df),
        original_rows,
    )

    # ---- Sanity log ----
    logger.info(
        "UF_Deviation stats — mean: %.3f, std: %.3f, min: %.3f, max: %.3f",
        df["UF_Deviation"].mean(),
        df["UF_Deviation"].std(),
        df["UF_Deviation"].min(),
        df["UF_Deviation"].max(),
    )

    return df


def get_training_data(
    csv_path: str = "raw_data.csv",
) -> tuple:
    """
    Return model-ready (X, y) from the raw CSV.

    Parameters
    ----------
    csv_path : str
        Path to raw CSV file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns in MODEL_FEATURES.
    y : pd.Series
        Target vector (UF_Deviation).
    """
    df = load_and_clean(csv_path)

    # Ensure all expected feature columns exist (fill missing with NaN —
    # TF-DF handles missing values natively via surrogate splits)
    for col in MODEL_FEATURES:
        if col not in df.columns:
            logger.warning("Feature '%s' not found — filling with NaN.", col)
            df[col] = np.nan

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    logger.info("Training data shapes: X=%s, y=%s", X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def build_inference_dataframe(
    age: float,
    sex: int,
    pre_weight: float,
    dry_weight: float,
    sbp: float,
    dbp: float,
    hr: float,
) -> pd.DataFrame:
    """
    Construct a single-row DataFrame for model inference.

    Internally calculates Weight_Difference = pre_weight − dry_weight.

    Parameters
    ----------
    age : float
        Patient age in years.
    sex : int
        0 = Male, 1 = Female.
    pre_weight : float
        Pre-dialysis weight in kg.
    dry_weight : float
        Target dry weight in kg.
    sbp : float
        Pre-dialysis systolic blood pressure (mmHg).
    dbp : float
        Pre-dialysis diastolic blood pressure (mmHg).
    hr : float
        Pre-dialysis heart rate (bpm).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all MODEL_FEATURES columns.

    Raises
    ------
    ValueError
        If pre_weight or dry_weight is NaN/None, since Weight_Difference
        cannot be computed.

    Clinical Safety Note
    --------------------
    This function only builds the DataFrame. Clinical guardrails (short-circuit
    logic, caps) are enforced in train_predict.load_model_and_predict().
    """
    # --- SAFETY: validate weight inputs ---
    if pre_weight is None or dry_weight is None:
        raise ValueError(
            "Both pre_weight and dry_weight are required to compute "
            "Weight_Difference. Cannot proceed with NaN weight values."
        )
    if np.isnan(pre_weight) or np.isnan(dry_weight):
        raise ValueError(
            "Both pre_weight and dry_weight must be numeric (not NaN). "
            "Weight_Difference cannot be calculated."
        )

    weight_difference = pre_weight - dry_weight

    row = {
        "Age": [float(age)],
        "Sex": [int(sex)],
        "Pre_Dialysis_Weight": [float(pre_weight)],
        "Weight_Difference": [float(weight_difference)],
        "Pre_Dialysis_SBP": [float(sbp)],
        "Pre_Dialysis_DBP": [float(dbp)],
        "Pre_Dialysis_Heart_Rate": [float(hr)],
    }

    return pd.DataFrame(row)


# ---------------------------------------------------------------------------
# Quick self-test when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Preprocessing Module — Self-Test")
    print("=" * 60)

    # Test 1: Regex parsers
    print("\n--- Parser Tests ---")
    assert _parse_age_sex("65/M") == (65, 0), "Failed: 65/M"
    assert _parse_age_sex("72 / Female") == (72, 1), "Failed: 72 / Female"
    assert _parse_age_sex("30/male") == (30, 0), "Failed: 30/male"
    assert _parse_age_sex("bad") == (np.nan, np.nan), "Failed: bad data"
    assert _parse_age_sex(np.nan) == (np.nan, np.nan), "Failed: NaN"
    print("  _parse_age_sex: ALL PASSED")

    assert _parse_bp("130/80") == (130.0, 80.0), "Failed: 130/80"
    assert _parse_bp("130-80") == (130.0, 80.0), "Failed: 130-80"
    bp_sbp_only = _parse_bp("130")
    assert bp_sbp_only[0] == 130.0 and np.isnan(bp_sbp_only[1]), "Failed: 130"
    bp_nan = _parse_bp(np.nan)
    assert np.isnan(bp_nan[0]) and np.isnan(bp_nan[1]), "Failed: NaN"
    print("  _parse_bp: ALL PASSED")

    # Test 2: Inference DataFrame builder
    print("\n--- Inference DataFrame Test ---")
    inf_df = build_inference_dataframe(
        age=60, sex=0, pre_weight=75.0, dry_weight=72.0,
        sbp=130, dbp=80, hr=75
    )
    assert inf_df["Weight_Difference"].iloc[0] == 3.0, "Weight diff wrong"
    assert len(inf_df) == 1, "Should be single row"
    assert list(inf_df.columns) == MODEL_FEATURES, "Column mismatch"
    print("  build_inference_dataframe: PASSED")

    # Test 3: ValueError on NaN weights
    print("\n--- NaN Weight Validation Test ---")
    try:
        build_inference_dataframe(
            age=60, sex=0, pre_weight=np.nan, dry_weight=72.0,
            sbp=130, dbp=80, hr=75
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  NaN weight raises ValueError: PASSED")

    print("\n" + "=" * 60)
    print("All preprocessing self-tests PASSED.")
    print("=" * 60)

    # Optional: attempt data load if CSV exists
    csv = Path("raw_data.csv")
    if csv.exists():
        print("\n--- Loading real data ---")
        X, y = get_training_data(str(csv))
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Sample X:\n{X.head()}")
        print(f"  Sample y:\n{y.head()}")
    else:
        print(f"\n[SKIP] {csv} not found. Run export_xlsx_to_csv() first.")