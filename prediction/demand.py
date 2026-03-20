"""
Demand Forecasting (P1)

mmm_weekly_market データから上位5ブランドの需要予測を実行。
SARIMAX + 外部変数（季節指数、気温、祝日、TV GRP、デジタル広告費）でモデリング。
学習80% / テスト20%でMAPE・RMSEを評価し、12週先を予測。

Usage:
    uv run python prediction/demand.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from lib.data_loader import fetch_all, get_supabase_client, load_config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "demand"

EXOG_COLS = [
    "seasonal_index",
    "avg_temperature",
    "is_holiday_week",
    "tv_grp",
    "digital_spend_jpy",
]

SALES_COL = "sales_volume"
BRAND_COL = "brand_code"
WEEK_COL = "week_start_date"
TOP_N_BRANDS = 5
TRAIN_RATIO = 0.8
FORECAST_WEEKS = 12


# ============================================================
# Data Fetch
# ============================================================


def fetch_weekly_market(cfg: dict) -> pd.DataFrame:
    """Fetch mmm_weekly_market with all columns."""
    client = get_supabase_client(cfg)
    if client is None:
        raise RuntimeError("Supabase client required")
    try:
        rows = fetch_all(client, "mmm_weekly_market", "*")
    finally:
        client.close()
    df = pd.DataFrame(rows)
    print(f"Fetched mmm_weekly_market: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ============================================================
# Preprocessing
# ============================================================


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, sort, and fill missing exogenous values."""
    df = df.copy()
    df[WEEK_COL] = pd.to_datetime(df[WEEK_COL])
    df = df.sort_values([BRAND_COL, WEEK_COL]).reset_index(drop=True)

    # Ensure numeric columns
    for col in EXOG_COLS + [SALES_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing exog with median per brand
    for col in EXOG_COLS:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df.groupby(BRAND_COL)[col].transform(
                    lambda s: s.fillna(s.median())
                )
                df[col] = df[col].fillna(0.0)
                print(f"  {col}: {n_missing} missing values filled")

    return df


def get_top_brands(df: pd.DataFrame) -> list[str]:
    """Return top N brands by total sales volume."""
    brand_totals = df.groupby(BRAND_COL)[SALES_COL].sum().sort_values(ascending=False)
    top = brand_totals.head(TOP_N_BRANDS).index.tolist()
    print(f"\nTop {TOP_N_BRANDS} brands by total sales:")
    for b in top:
        print(f"  {b}: {brand_totals[b]:,.0f}")
    return top


# ============================================================
# Modeling
# ============================================================


def fit_and_forecast(
    df_brand: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Fit SARIMAX model, evaluate on test set, and forecast future weeks."""
    df_brand = df_brand.sort_values(WEEK_COL).reset_index(drop=True)
    n = len(df_brand)
    n_train = int(n * TRAIN_RATIO)

    if n_train < 10:
        return (
            {"error": "insufficient data", "n_rows": n},
            pd.DataFrame(),
            pd.DataFrame(),
        )

    y = df_brand[SALES_COL].values.astype(float)

    # Build exog matrix with available columns
    available_exog = [c for c in EXOG_COLS if c in df_brand.columns]
    if available_exog:
        exog = df_brand[available_exog].values.astype(float)
    else:
        exog = None

    y_train, y_test = y[:n_train], y[n_train:]
    exog_train = exog[:n_train] if exog is not None else None
    exog_test = exog[n_train:] if exog is not None else None

    # Fit SARIMAX with seasonal order (period=52 if enough data, else 13 for quarterly)
    seasonal_period = 13 if n_train < 104 else 52
    # Use simple orders to avoid convergence issues
    try:
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
    except Exception as e:
        # Fallback: simpler model without seasonal component
        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)
        except Exception as e2:
            return (
                {"error": f"model fit failed: {e2}"},
                pd.DataFrame(),
                pd.DataFrame(),
            )

    # Test set prediction
    n_test = len(y_test)
    pred_test = result.get_forecast(steps=n_test, exog=exog_test).predicted_mean
    pred_test = np.maximum(pred_test, 0)  # No negative sales

    # Metrics
    mape = float(np.mean(np.abs((y_test - pred_test) / np.maximum(y_test, 1))) * 100)
    rmse = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))

    metrics = {
        "n_train": n_train,
        "n_test": n_test,
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "aic": round(float(result.aic), 2),
    }

    # Actual vs predicted dataframe
    test_dates = df_brand[WEEK_COL].iloc[n_train:].values
    df_avp = pd.DataFrame(
        {
            WEEK_COL: test_dates,
            "actual": y_test,
            "predicted": np.round(pred_test, 2),
        }
    )

    # Forecast next N weeks
    last_date = df_brand[WEEK_COL].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1), periods=FORECAST_WEEKS, freq="W-MON"
    )

    # For future exog, use mean of last 12 weeks as proxy
    if exog is not None:
        last_12 = exog[-min(12, len(exog)) :]
        future_exog = np.tile(last_12.mean(axis=0), (FORECAST_WEEKS, 1))
    else:
        future_exog = None

    forecast_vals = result.get_forecast(
        steps=FORECAST_WEEKS, exog=future_exog
    ).predicted_mean
    forecast_vals = np.maximum(forecast_vals, 0)

    df_forecast = pd.DataFrame(
        {
            WEEK_COL: future_dates,
            "predicted_volume": np.round(forecast_vals, 2),
        }
    )

    return metrics, df_avp, df_forecast


# ============================================================
# Output
# ============================================================


def save_outputs(
    all_metrics: dict,
    all_avp: list[pd.DataFrame],
    all_forecasts: list[pd.DataFrame],
) -> None:
    """Save all outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Model metrics
    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_DIR / 'model_metrics.json'}")

    # Actual vs predicted
    if all_avp:
        df_avp = pd.concat(all_avp, ignore_index=True)
        df_avp.to_csv(OUTPUT_DIR / "actual_vs_predicted.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'actual_vs_predicted.csv'}")

    # Forecast
    if all_forecasts:
        df_fc = pd.concat(all_forecasts, ignore_index=True)
        df_fc.to_csv(OUTPUT_DIR / "forecast.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'forecast.csv'}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  Demand Forecasting (P1)")
    print("=" * 60)

    cfg = load_config()
    df = fetch_weekly_market(cfg)

    if df.empty:
        print("ERROR: No data fetched. Exiting.")
        sys.exit(1)

    print("\nPreparing data...")
    df = prepare_data(df)

    top_brands = get_top_brands(df)

    all_metrics: dict[str, dict] = {}
    all_avp: list[pd.DataFrame] = []
    all_forecasts: list[pd.DataFrame] = []

    for brand in top_brands:
        print(f"\n{'─' * 40}")
        print(f"  Modeling: {brand}")
        print(f"{'─' * 40}")

        df_brand = df[df[BRAND_COL] == brand].copy()
        print(f"  Data points: {len(df_brand)}")

        metrics, df_avp, df_forecast = fit_and_forecast(df_brand)
        all_metrics[brand] = metrics

        if not df_avp.empty:
            df_avp[BRAND_COL] = brand
            all_avp.append(df_avp)

        if not df_forecast.empty:
            df_forecast[BRAND_COL] = brand
            all_forecasts.append(df_forecast)

        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
        else:
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  RMSE: {metrics['rmse']:,.2f}")

    save_outputs(all_metrics, all_avp, all_forecasts)

    # Print summary
    print("\n" + "=" * 60)
    print("  DEMAND FORECAST SUMMARY")
    print("=" * 60)

    print("\nPer-brand MAPE:")
    for brand, m in all_metrics.items():
        if "error" not in m:
            print(f"  {brand}: MAPE={m['mape']:.2f}%, RMSE={m['rmse']:,.2f}")
        else:
            print(f"  {brand}: {m['error']}")

    print("\nNext 4 weeks forecast:")
    if all_forecasts:
        df_fc = pd.concat(all_forecasts, ignore_index=True)
        for brand in top_brands:
            fc_brand = df_fc[df_fc[BRAND_COL] == brand].head(4)
            if not fc_brand.empty:
                vals = fc_brand["predicted_volume"].tolist()
                print(f"  {brand}: {[f'{v:,.0f}' for v in vals]}")


if __name__ == "__main__":
    main()
