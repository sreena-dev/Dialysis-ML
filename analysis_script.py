"""
Exploratory Data Analysis script for Hemodialysis Treatment Data.
This script loads the dataset, performs basic analysis and outputs
the results to a text file.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

output: list[str] = []


def p_out(text: str = "") -> None:
    """Appends text to the output list."""
    output.append(str(text))


def analyze_dataframe(df: pd.DataFrame) -> None:
    """Perform exploratory data analysis on the provided dataframe."""
    p_out("=" * 80)
    p_out("HEMODIALYSIS TREATMENT DATA - EXPLORATORY DATA ANALYSIS")
    p_out("=" * 80)
    p_out(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    p_out("=" * 80)
    p_out("COLUMN LISTING")
    p_out("=" * 80)
    for i, col in enumerate(df.columns):
        not_na = df[col].notna().sum()
        unique = df[col].nunique()
        p_out(
            f"  [{i:2d}] {col}  |  dtype: {df[col].dtype}  " +
            f"|  non-null: {not_na}/{len(df)}  |  unique: {unique}"
        )

    p_out("\n" + "=" * 80)
    p_out("FIRST 5 ROWS (first 20 columns)")
    p_out("=" * 80)
    p_out(df.iloc[:5, :20].to_string())

    p_out("\n" + "=" * 80)
    p_out("MISSING VALUES SUMMARY")
    p_out("=" * 80)
    missing = df.isna().sum()
    missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    for col in df.columns:
        if missing[col] > 0:
            p_out(f"  {col}: {missing[col]} missing ({missing_pct[col]}%)")

    total_missing = missing.sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_ratio = (total_missing / total_cells * 100) if total_cells else 0
    p_out(
        f"\nTotal missing cells: {total_missing} out of {total_cells} " +
        f"({missing_ratio:.2f}%)"
    )

    p_out("\n" + "=" * 80)
    p_out("NUMERIC COLUMNS - STATISTICS")
    p_out("=" * 80)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            p_out(f"\n--- {col} ---")
            p_out(f"  Count: {df[col].notna().sum()}")
            p_out(f"  Mean: {df[col].mean():.2f}")
            p_out(f"  Std: {df[col].std():.2f}")
            p_out(f"  Min: {df[col].min()}")
            p_out(f"  25%: {df[col].quantile(0.25)}")
            p_out(f"  50% (Median): {df[col].quantile(0.50)}")
            p_out(f"  75%: {df[col].quantile(0.75)}")
            p_out(f"  Max: {df[col].max()}")

    analyze_categorical_and_correlations(df, numeric_cols, missing_pct)


def analyze_categorical_and_correlations(
    df: pd.DataFrame, numeric_cols: pd.Index, missing_pct: pd.Series
) -> None:
    """Analyze categorical columns, correlations, and data quality issues."""
    p_out("\n" + "=" * 80)
    p_out("CATEGORICAL COLUMNS - VALUE COUNTS")
    p_out("=" * 80)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].notna().sum() > 0:
            p_out(f"\n--- {col} ---")
            vc = df[col].value_counts()
            if len(vc) > 10:
                for val, cnt in vc.head(10).items():
                    p_out(f"  {val}: {cnt} ({cnt/len(df)*100:.1f}%)")
                p_out(f"  ... and {len(vc) - 10} more unique values")
            else:
                for val, cnt in vc.items():
                    p_out(f"  {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

    p_out("\n" + "=" * 80)
    p_out("CORRELATION MATRIX (top numeric columns)")
    p_out("=" * 80)
    good_numeric = [
        col for col in numeric_cols if df[col].notna().sum() > len(df) * 0.3
    ]
    if len(good_numeric) > 1:
        corr = df.filter(items=good_numeric).corr()
        pairs = []
        for i, good_num_i in enumerate(good_numeric):
            for j in range(i + 1, len(good_numeric)):
                pairs.append((good_num_i, good_numeric[j], corr.iloc[i, j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        p_out("\nTop 20 correlations (by absolute value):")
        for c1, c2, r in pairs[:20]:
            p_out(f"  {c1} <-> {c2}: {r:.4f}")

    p_out("\n" + "=" * 80)
    p_out("DATA QUALITY NOTES")
    p_out("=" * 80)
    dups = df.duplicated().sum()
    p_out(f"  Duplicate rows: {dups}")

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        p_out(f"  Columns with constant/empty values: {constant_cols}")

    high_missing = [col for col in df.columns if missing_pct[col] > 80]
    if high_missing:
        p_out(f"  Columns with >80% missing: {high_missing}")


def main():
    """Main execution function for the analysis script."""
    # Load data
    df = pd.read_excel(r'd:\Dialysis ML\Hemodialysis Treatment Data.xlsx')

    # Run Analysis
    analyze_dataframe(df)

    # Write output
    report = "\n".join(output)
    with open(r'd:\Dialysis ML\analysis_output.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("Analysis complete. Output written to analysis_output.txt")


if __name__ == '__main__':
    main()
