"""
M1: Marketing Mix Modeling (MMM)
OLS regression per brand with adstock-transformed media variables.
Output: output/mmm/model_summary.json, media_decomposition.csv, elasticities.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.data_loader import load_config, fetch_all, get_supabase_client  # noqa: E402


# --- Adstock transformation ---

def adstock(series: pd.Series, decay: float) -> pd.Series:
    """Apply adstock transformation: adstock_t = x_t + decay * adstock_{t-1}."""
    result = np.zeros(len(series))
    result[0] = series.iloc[0]
    for i in range(1, len(series)):
        result[i] = series.iloc[i] + decay * result[i - 1]
    return pd.Series(result, index=series.index)


# --- Configuration ---

MEDIA_VARS = {
    "tv_grp": 0.7,
    "digital_spend_jpy": 0.4,
    "ooh_spend_jpy": 0.5,
    "trade_promo_spend_jpy": 0.5,
    "line_messages_delivered": 0.5,
    "campaign_entries": 0.5,
}

CONTROL_VARS = ["avg_temperature", "seasonal_index", "competitor_tv_grp"]

DEPENDENT_VAR = "sales_volume"


def run_mmm(df_brand: pd.DataFrame, brand_name: str) -> dict | None:
    """Fit OLS for a single brand and return results dict."""
    df = df_brand.sort_values("week_start_date").reset_index(drop=True)

    # Check minimum rows
    if len(df) < 10:
        print(f"  {brand_name}: skipped (only {len(df)} rows)")
        return None

    y = df[DEPENDENT_VAR].astype(float)

    # Build X with adstock-transformed media + control variables
    X_parts: dict[str, pd.Series] = {}
    for var, decay_rate in MEDIA_VARS.items():
        if var in df.columns:
            X_parts[f"{var}_adstock"] = adstock(df[var].astype(float).fillna(0), decay_rate)
    for var in CONTROL_VARS:
        if var in df.columns:
            X_parts[var] = df[var].astype(float).fillna(0)

    X = pd.DataFrame(X_parts)
    X = sm.add_constant(X)

    # Fit OLS
    model = sm.OLS(y, X).fit()

    # Elasticities: coeff * mean(x) / mean(y)
    mean_y = y.mean()
    elasticities = {}
    for col in X.columns:
        if col == "const":
            continue
        elasticities[col] = float(model.params[col] * X[col].mean() / mean_y) if mean_y != 0 else 0.0

    # Contribution (decomposition): coeff * actual values
    contributions = {}
    for col in X.columns:
        contributions[col] = (model.params[col] * X[col]).tolist()

    # Coefficients and p-values
    coefficients = {col: float(model.params[col]) for col in X.columns}
    p_values = {col: float(model.pvalues[col]) for col in X.columns}

    return {
        "brand_code": brand_name,
        "n_weeks": len(df),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "coefficients": coefficients,
        "p_values": p_values,
        "elasticities": elasticities,
        "contributions": contributions,
        "mean_sales": float(mean_y),
    }


def main() -> None:
    cfg = load_config()
    client = get_supabase_client(cfg)
    if client is None:
        print("ERROR: Supabase client not configured. Set data_source.type to 'supabase' in config.yaml.")
        sys.exit(1)

    print("Fetching mmm_weekly_market data...")
    rows = fetch_all(client, "mmm_weekly_market", "*")
    df = pd.DataFrame(rows)
    print(f"  mmm_weekly_market: {len(df):,} rows")

    if df.empty:
        print("ERROR: No data returned from mmm_weekly_market.")
        sys.exit(1)

    # Identify top 5 brands by total sales_volume
    brand_sales = df.groupby("brand_code")[DEPENDENT_VAR].sum().sort_values(ascending=False)
    top_brands = brand_sales.head(5).index.tolist()
    print(f"\nTop 5 brands by total sales: {top_brands}")

    # Run MMM per brand
    all_results: list[dict] = []
    all_decomposition_rows: list[dict] = []
    all_elasticity_rows: list[dict] = []

    for brand in top_brands:
        df_brand = df[df["brand_code"] == brand].copy()
        result = run_mmm(df_brand, brand)
        if result is None:
            continue

        all_results.append(result)

        # Decomposition rows
        n_weeks = result["n_weeks"]
        for col, values in result["contributions"].items():
            for i, val in enumerate(values):
                all_decomposition_rows.append({
                    "brand_code": brand,
                    "week_index": i,
                    "variable": col,
                    "contribution": val,
                })

        # Elasticity rows
        for var, elast in result["elasticities"].items():
            all_elasticity_rows.append({
                "brand_code": brand,
                "variable": var,
                "elasticity": elast,
            })

    # --- Output ---
    output_dir = Path(__file__).parent.parent / "output" / "mmm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # model_summary.json
    summary_for_json = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "contributions"}
        summary_for_json.append(entry)

    summary_path = output_dir / "model_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_for_json, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {summary_path}")

    # media_decomposition.csv
    decomp_path = output_dir / "media_decomposition.csv"
    pd.DataFrame(all_decomposition_rows).to_csv(decomp_path, index=False)
    print(f"Saved: {decomp_path}")

    # elasticities.csv
    elast_path = output_dir / "elasticities.csv"
    pd.DataFrame(all_elasticity_rows).to_csv(elast_path, index=False)
    print(f"Saved: {elast_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("MMM Results Summary")
    print("=" * 60)
    for r in all_results:
        print(f"\n  Brand: {r['brand_code']}")
        print(f"    R²: {r['r_squared']:.4f}  (Adj R²: {r['adj_r_squared']:.4f})")
        # Top 3 drivers by absolute elasticity
        sorted_elast = sorted(r["elasticities"].items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = sorted_elast[:3]
        print(f"    Top 3 drivers by elasticity:")
        for var, elast in top3:
            print(f"      {var}: {elast:.4f}")


if __name__ == "__main__":
    main()
