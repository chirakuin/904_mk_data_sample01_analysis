"""
Budget Optimization (P3)

MMM分析 (M1) の結果を読み込み、チャネル別広告予算の最適配分を算出。
scipy.optimize (SLSQP) で対数変換による逓減リターンモデルを最適化。

前提: effectiveness/mmm.py の出力が output/mmm/ に存在すること。
存在しない場合はエラーメッセージを表示して終了。

Usage:
    uv run python prediction/budget.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from lib.data_loader import fetch_all, get_supabase_client, load_config

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "budget"
MMM_DIR = PROJECT_ROOT / "output" / "mmm"

MIN_RATIO = 0.10  # Each channel >= 10% of current
MAX_RATIO = 3.00  # Each channel <= 300% of current


# ============================================================
# Load MMM Outputs
# ============================================================


def load_mmm_outputs() -> tuple[dict, pd.DataFrame]:
    """Load model_summary.json and elasticities.csv from MMM output."""
    summary_path = MMM_DIR / "model_summary.json"
    elasticities_path = MMM_DIR / "elasticities.csv"

    if not summary_path.exists():
        print(f"ERROR: MMM model summary not found at {summary_path}")
        print("Please run effectiveness/mmm.py first to generate M1 outputs.")
        sys.exit(1)

    if not elasticities_path.exists():
        print(f"ERROR: MMM elasticities not found at {elasticities_path}")
        print("Please run effectiveness/mmm.py first to generate M1 outputs.")
        sys.exit(1)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    elasticities = pd.read_csv(elasticities_path)
    print(f"Loaded MMM model summary from {summary_path}")
    print(f"Loaded elasticities: {len(elasticities)} channels")
    print(f"  Variables: {elasticities['variable'].tolist()}")

    return summary, elasticities


# ============================================================
# Optimization
# ============================================================


def response_function(spend: np.ndarray, coefficients: np.ndarray) -> float:
    """Compute total predicted sales from spend allocation.

    Uses diminishing returns: sales_contribution = coeff * log(1 + spend)
    """
    contributions = coefficients * np.log1p(spend)
    return float(np.sum(contributions))


def negative_sales(
    spend: np.ndarray, coefficients: np.ndarray
) -> float:
    """Objective function to minimize (negative sales for maximization)."""
    return -response_function(spend, coefficients)


def optimize_budget(
    channels: list[str],
    coefficients: np.ndarray,
    current_spend: np.ndarray,
) -> dict:
    """Run budget optimization with SLSQP.

    Constraints:
    - Total budget = sum of current spend
    - Each channel >= 10% of its current allocation
    - Each channel <= 300% of its current allocation
    """
    n = len(channels)
    total_budget = float(np.sum(current_spend))

    print(f"\nOptimization setup:")
    print(f"  Total budget: {total_budget:,.0f}")
    print(f"  Channels: {n}")
    print(f"  Constraints: each channel in [{MIN_RATIO*100:.0f}%, {MAX_RATIO*100:.0f}%] of current")

    # Bounds per channel
    bounds = [
        (max(1.0, current_spend[i] * MIN_RATIO), current_spend[i] * MAX_RATIO)
        for i in range(n)
    ]

    # Budget equality constraint
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}
    ]

    # Initial guess = current allocation
    x0 = current_spend.copy()

    result = minimize(
        negative_sales,
        x0,
        args=(coefficients,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    current_sales = response_function(current_spend, coefficients)
    optimal_sales = response_function(result.x, coefficients)
    lift = (optimal_sales - current_sales) / current_sales * 100 if current_sales > 0 else 0

    opt_result = {
        "success": bool(result.success),
        "message": result.message,
        "total_budget": round(total_budget, 2),
        "current_predicted_sales": round(current_sales, 2),
        "optimal_predicted_sales": round(optimal_sales, 2),
        "expected_sales_lift_pct": round(lift, 2),
        "iterations": int(result.nit),
    }

    return opt_result


# ============================================================
# Output
# ============================================================


def save_outputs(
    channels: list[str],
    coefficients: np.ndarray,
    current_spend: np.ndarray,
    optimal_spend: np.ndarray,
    opt_result: dict,
) -> None:
    """Save all outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optimal allocation CSV
    df_alloc = pd.DataFrame(
        {
            "channel": channels,
            "current_spend": np.round(current_spend, 2),
            "optimal_spend": np.round(optimal_spend, 2),
            "change_pct": np.round(
                (optimal_spend - current_spend) / np.maximum(current_spend, 1) * 100, 2
            ),
            "current_share_pct": np.round(
                current_spend / np.sum(current_spend) * 100, 2
            ),
            "optimal_share_pct": np.round(
                optimal_spend / np.sum(optimal_spend) * 100, 2
            ),
            "coefficient": np.round(coefficients, 6),
            "current_contribution": np.round(
                coefficients * np.log1p(current_spend), 2
            ),
            "optimal_contribution": np.round(
                coefficients * np.log1p(optimal_spend), 2
            ),
        }
    )
    df_alloc.to_csv(OUTPUT_DIR / "optimal_allocation.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'optimal_allocation.csv'}")

    # Comparison JSON
    comparison = {
        "channels": {},
    }
    for i, ch in enumerate(channels):
        comparison["channels"][ch] = {
            "current_spend": round(float(current_spend[i]), 2),
            "optimal_spend": round(float(optimal_spend[i]), 2),
            "change_pct": round(
                float((optimal_spend[i] - current_spend[i]) / max(current_spend[i], 1) * 100), 2
            ),
        }
    with open(OUTPUT_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {OUTPUT_DIR / 'comparison.json'}")

    # Optimization result JSON
    with open(OUTPUT_DIR / "optimization_result.json", "w", encoding="utf-8") as f:
        json.dump(opt_result, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {OUTPUT_DIR / 'optimization_result.json'}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  Budget Optimization (P3)")
    print("=" * 60)

    summary, elasticities = load_mmm_outputs()

    # Aggregate elasticities across brands (use first brand or average)
    # elasticities.csv has columns: brand_code, variable, elasticity
    first_brand = summary[0]["brand_code"] if summary else elasticities["brand_code"].iloc[0]
    brand_elast = elasticities[elasticities["brand_code"] == first_brand].copy()

    # Map adstock variable names to spend columns
    SPEND_MAP = {
        "tv_grp_adstock": "tv_spend_jpy",
        "digital_spend_jpy_adstock": "digital_spend_jpy",
        "ooh_spend_jpy_adstock": "ooh_spend_jpy",
        "trade_promo_spend_jpy_adstock": "trade_promo_spend_jpy",
        "line_messages_delivered_adstock": "digital_line_spend_jpy",
        "campaign_entries_adstock": "trade_promo_spend_jpy",
    }

    # Fetch actual spend data
    cfg = load_config()
    client = get_supabase_client(cfg)
    df_market = pd.DataFrame(fetch_all(client, "mmm_weekly_market", "*"))
    if client:
        client.close()
    df_brand_market = df_market[df_market["brand_code"] == first_brand]

    channels = []
    coefficients_list = []
    current_spend_list = []
    for _, row in brand_elast.iterrows():
        var = row["variable"]
        spend_col = SPEND_MAP.get(var)
        if spend_col and spend_col in df_brand_market.columns:
            channels.append(var.replace("_adstock", ""))
            coefficients_list.append(float(row["elasticity"]))
            current_spend_list.append(float(df_brand_market[spend_col].sum()))

    if not channels:
        print("ERROR: No matching spend channels found")
        sys.exit(1)

    coefficients = np.array(coefficients_list)
    current_spend = np.array(current_spend_list)

    print(f"\nCurrent allocation:")
    for i, ch in enumerate(channels):
        print(f"  {ch}: spend={current_spend[i]:,.0f}, coeff={coefficients[i]:.6f}")

    # Run optimization
    print("\nRunning optimization (SLSQP)...")
    opt_result = optimize_budget(channels, coefficients, current_spend)

    # Reconstruct optimal spend from result
    # Re-run to get the actual x values
    n = len(channels)
    total_budget = float(np.sum(current_spend))
    bounds = [
        (max(1.0, current_spend[i] * MIN_RATIO), current_spend[i] * MAX_RATIO)
        for i in range(n)
    ]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - total_budget}]

    result = minimize(
        negative_sales,
        current_spend.copy(),
        args=(coefficients,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    optimal_spend = result.x

    save_outputs(channels, coefficients, current_spend, optimal_spend, opt_result)

    # Print summary
    print("\n" + "=" * 60)
    print("  BUDGET OPTIMIZATION SUMMARY")
    print("=" * 60)

    print(f"\nOptimization {'succeeded' if opt_result['success'] else 'FAILED'}: {opt_result['message']}")
    print(f"Total budget: {opt_result['total_budget']:,.0f}")

    print(f"\nCurrent vs Optimal allocation:")
    print(f"  {'Channel':<25} {'Current':>12} {'Optimal':>12} {'Change':>8}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12} {'─' * 8}")
    for i, ch in enumerate(channels):
        change_pct = (optimal_spend[i] - current_spend[i]) / max(current_spend[i], 1) * 100
        print(
            f"  {ch:<25} {current_spend[i]:>12,.0f} {optimal_spend[i]:>12,.0f} {change_pct:>+7.1f}%"
        )

    print(f"\nExpected sales lift: {opt_result['expected_sales_lift_pct']:+.2f}%")
    print(f"Current predicted sales:  {opt_result['current_predicted_sales']:,.2f}")
    print(f"Optimal predicted sales:  {opt_result['optimal_predicted_sales']:,.2f}")


if __name__ == "__main__":
    main()
