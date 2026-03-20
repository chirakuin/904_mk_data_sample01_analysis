"""
Customer LTV Prediction (C5)

purchase_transaction から BG/NBD + Gamma-Gamma モデルで
顧客別の予測購買回数・予測CLVを算出する。

Usage:
    uv run python customer/ltv.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

from lib.data_loader import fetch_all, get_supabase_client, load_config

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "ltv"

CID_COL = "unified_customer_id"
PREDICTION_MONTHS = 12


# ============================================================
# Data Fetch
# ============================================================


def fetch_purchase_data(cfg: dict) -> pd.DataFrame:
    """Fetch purchase_transaction."""
    client = get_supabase_client(cfg)
    if client is None:
        raise RuntimeError("Supabase client required")
    try:
        rows = fetch_all(
            client,
            "purchase_transaction",
            f"{CID_COL},purchase_datetime,total_amount",
        )
    finally:
        client.close()
    df = pd.DataFrame(rows)
    print(f"Fetched purchase_transaction: {len(df):,} rows")
    return df


# ============================================================
# RFM Summary
# ============================================================


def build_rfm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build RFM summary for lifetimes models."""
    df = df.copy()
    df["purchase_datetime"] = pd.to_datetime(df["purchase_datetime"])
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(0)

    # lifetimes requires: customer_id, datetime, monetary_value
    rfm = summary_data_from_transaction_data(
        df,
        customer_id_col=CID_COL,
        datetime_col="purchase_datetime",
        monetary_value_col="total_amount",
    )

    print(f"\nRFM summary: {len(rfm):,} customers")
    print(f"  frequency: mean={rfm['frequency'].mean():.1f}, max={rfm['frequency'].max():.0f}")
    print(f"  recency: mean={rfm['recency'].mean():.1f} days")
    print(f"  T: mean={rfm['T'].mean():.1f} days")
    if "monetary_value" in rfm.columns:
        print(f"  monetary_value: mean={rfm['monetary_value'].mean():.0f}")

    return rfm


# ============================================================
# Model Fitting
# ============================================================


def fit_models(rfm: pd.DataFrame) -> tuple[BetaGeoFitter, GammaGammaFitter, dict]:
    """Fit BG/NBD and Gamma-Gamma models."""
    # BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    print(f"\nBG/NBD model fitted:")
    print(f"  params: {dict(bgf.params_)}")

    # Gamma-Gamma model (requires frequency > 0)
    rfm_gg = rfm[rfm["frequency"] > 0].copy()
    if len(rfm_gg) == 0:
        raise ValueError("No customers with repeat purchases for Gamma-Gamma model")

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rfm_gg["frequency"], rfm_gg["monetary_value"])
    print(f"\nGamma-Gamma model fitted:")
    print(f"  params: {dict(ggf.params_)}")

    model_params = {
        "bg_nbd": {k: round(float(v), 6) for k, v in bgf.params_.items()},
        "gamma_gamma": {k: round(float(v), 6) for k, v in ggf.params_.items()},
        "n_customers_total": len(rfm),
        "n_customers_repeat": len(rfm_gg),
    }

    return bgf, ggf, model_params


# ============================================================
# Prediction
# ============================================================


def predict_ltv(
    rfm: pd.DataFrame, bgf: BetaGeoFitter, ggf: GammaGammaFitter
) -> pd.DataFrame:
    """Predict per-customer purchases and CLV."""
    t = PREDICTION_MONTHS * 30  # approximate days

    # Predicted purchases
    rfm = rfm.copy()
    rfm["predicted_purchases_12m"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t, rfm["frequency"], rfm["recency"], rfm["T"]
    )

    # Predicted CLV (only for repeat customers)
    rfm["predicted_clv_12m"] = 0.0
    repeat_mask = rfm["frequency"] > 0
    if repeat_mask.any():
        rfm.loc[repeat_mask, "predicted_clv_12m"] = ggf.customer_lifetime_value(
            bgf,
            rfm.loc[repeat_mask, "frequency"],
            rfm.loc[repeat_mask, "recency"],
            rfm.loc[repeat_mask, "T"],
            rfm.loc[repeat_mask, "monetary_value"],
            time=PREDICTION_MONTHS,
            freq="D",
        )

    # LTV quartiles
    rfm["ltv_quartile"] = pd.qcut(
        rfm["predicted_clv_12m"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    )

    return rfm


# ============================================================
# Output
# ============================================================


def save_outputs(rfm: pd.DataFrame, model_params: dict, ltv_summary: dict) -> None:
    """Save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Customer LTV
    out_cols = [
        "frequency", "recency", "T", "monetary_value",
        "predicted_purchases_12m", "predicted_clv_12m", "ltv_quartile",
    ]
    df_out = rfm[[c for c in out_cols if c in rfm.columns]].copy()
    df_out.index.name = CID_COL
    df_out = df_out.round(2)
    df_out.to_csv(OUTPUT_DIR / "customer_ltv.csv")
    print(f"\nSaved: {OUTPUT_DIR / 'customer_ltv.csv'}")

    # LTV summary
    with open(OUTPUT_DIR / "ltv_summary.json", "w", encoding="utf-8") as f:
        json.dump(ltv_summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {OUTPUT_DIR / 'ltv_summary.json'}")

    # Model params
    with open(OUTPUT_DIR / "model_params.json", "w", encoding="utf-8") as f:
        json.dump(model_params, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'model_params.json'}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  Customer LTV Prediction (C5)")
    print("=" * 60)

    cfg = load_config()
    df = fetch_purchase_data(cfg)

    if df.empty:
        print("ERROR: No purchase data. Exiting.")
        sys.exit(1)

    rfm = build_rfm_summary(df)

    print("\nFitting models...")
    bgf, ggf, model_params = fit_models(rfm)

    print("\nPredicting LTV...")
    rfm = predict_ltv(rfm, bgf, ggf)

    # Build summary
    ltv_summary = {
        "total_customers": len(rfm),
        "predicted_purchases_12m": {
            "mean": round(float(rfm["predicted_purchases_12m"].mean()), 2),
            "median": round(float(rfm["predicted_purchases_12m"].median()), 2),
            "std": round(float(rfm["predicted_purchases_12m"].std()), 2),
            "max": round(float(rfm["predicted_purchases_12m"].max()), 2),
        },
        "predicted_clv_12m": {
            "mean": round(float(rfm["predicted_clv_12m"].mean()), 2),
            "median": round(float(rfm["predicted_clv_12m"].median()), 2),
            "std": round(float(rfm["predicted_clv_12m"].std()), 2),
            "Q1": round(float(rfm["predicted_clv_12m"].quantile(0.25)), 2),
            "Q2": round(float(rfm["predicted_clv_12m"].quantile(0.50)), 2),
            "Q3": round(float(rfm["predicted_clv_12m"].quantile(0.75)), 2),
            "Q4": round(float(rfm["predicted_clv_12m"].quantile(1.0)), 2),
        },
        "ltv_quartile_sizes": rfm["ltv_quartile"].value_counts().to_dict(),
        "model_params": model_params,
    }

    save_outputs(rfm, model_params, ltv_summary)

    # Print summary
    print("\n" + "=" * 60)
    print("  LTV PREDICTION SUMMARY")
    print("=" * 60)
    print(f"\nCustomers: {len(rfm):,}")
    print(f"\nPredicted purchases (12m):")
    print(f"  Mean: {ltv_summary['predicted_purchases_12m']['mean']:.2f}")
    print(f"  Median: {ltv_summary['predicted_purchases_12m']['median']:.2f}")

    print(f"\nPredicted CLV (12m):")
    for k, v in ltv_summary["predicted_clv_12m"].items():
        print(f"  {k}: {v:,.0f}")

    print(f"\nLTV quartile distribution:")
    for q, cnt in sorted(rfm["ltv_quartile"].value_counts().items(), key=lambda x: str(x[0])):
        print(f"  {q}: {cnt:,} customers")

    print(f"\nModel fit:")
    print(f"  BG/NBD params: {model_params['bg_nbd']}")
    print(f"  Gamma-Gamma params: {model_params['gamma_gamma']}")


if __name__ == "__main__":
    main()
