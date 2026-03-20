"""
M2: Media ROI Analysis
Reads M1 (MMM) output and computes channel-level ROI metrics.
Output: output/roi/channel_roi.csv, roi_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.data_loader import load_config, fetch_all, get_supabase_client  # noqa: E402


# Mapping from adstock variable names back to spend/volume columns for ROI
CHANNEL_SPEND_MAP = {
    "tv_grp_adstock": "tv_grp",
    "digital_spend_jpy_adstock": "digital_spend_jpy",
    "ooh_spend_jpy_adstock": "ooh_spend_jpy",
    "trade_promo_spend_jpy_adstock": "trade_promo_spend_jpy",
    "line_messages_delivered_adstock": "line_messages_delivered",
    "campaign_entries_adstock": "campaign_entries",
}


def main() -> None:
    # --- Check M1 output exists ---
    project_root = Path(__file__).parent.parent
    mmm_dir = project_root / "output" / "mmm"
    summary_path = mmm_dir / "model_summary.json"
    elast_path = mmm_dir / "elasticities.csv"

    if not summary_path.exists() or not elast_path.exists():
        print("ERROR: MMM output not found. Run mmm.py (M1) first.")
        print(f"  Expected: {summary_path}")
        print(f"  Expected: {elast_path}")
        sys.exit(1)

    # --- Load M1 output ---
    with open(summary_path, encoding="utf-8") as f:
        model_summaries = json.load(f)

    df_elast = pd.read_csv(elast_path)

    # --- Fetch actual spend data for ROI denominator ---
    cfg = load_config()
    client = get_supabase_client(cfg)
    if client is None:
        print("ERROR: Supabase client not configured.")
        sys.exit(1)

    print("Fetching mmm_weekly_market data for spend totals...")
    rows = fetch_all(client, "mmm_weekly_market", "*")
    df_market = pd.DataFrame(rows)
    print(f"  mmm_weekly_market: {len(df_market):,} rows")

    # --- Compute ROI per brand per channel ---
    roi_rows: list[dict] = []
    roi_summary: list[dict] = []

    for model in model_summaries:
        brand = model.get("brand_code", model.get("brand", ""))
        mean_sales = model["mean_sales"]
        coefficients = model["coefficients"]
        elasticities = model["elasticities"]

        df_brand = df_market[df_market["brand_code"] == brand]
        n_weeks = len(df_brand)

        brand_roi: list[dict] = []

        for adstock_var, raw_var in CHANNEL_SPEND_MAP.items():
            if adstock_var not in elasticities:
                continue
            if raw_var not in df_brand.columns:
                continue

            elast = elasticities[adstock_var]
            coeff = coefficients.get(adstock_var, 0.0)

            # Total spend/volume for this channel
            total_spend = df_brand[raw_var].astype(float).sum()
            mean_spend = df_brand[raw_var].astype(float).mean()

            if total_spend == 0:
                continue

            # Incremental sales from 1% increase scenario
            incremental_sales_pct = elast * mean_sales * 0.01  # per week
            incremental_sales_total = incremental_sales_pct * n_weeks

            # Average ROI = total incremental sales / total spend
            avg_roi = (coeff * total_spend) / total_spend if total_spend > 0 else 0.0
            # Simplifies to coefficient, but represents sales per unit of channel

            # Marginal ROI: incremental sales from 1% more spend / 1% of spend
            marginal_spend_1pct = total_spend * 0.01
            marginal_roi = incremental_sales_total / marginal_spend_1pct if marginal_spend_1pct > 0 else 0.0

            entry = {
                "brand": brand,
                "channel": raw_var,
                "elasticity": round(elast, 6),
                "coefficient": round(coeff, 6),
                "total_spend": round(total_spend, 2),
                "mean_weekly_spend": round(mean_spend, 2),
                "incremental_sales_1pct": round(incremental_sales_total, 2),
                "average_roi": round(avg_roi, 4),
                "marginal_roi": round(marginal_roi, 4),
            }
            roi_rows.append(entry)
            brand_roi.append(entry)

        # Sort by marginal ROI for summary
        brand_roi_sorted = sorted(brand_roi, key=lambda x: x["marginal_roi"], reverse=True)
        roi_summary.append({
            "brand": brand,
            "r_squared": model["r_squared"],
            "channel_ranking": [
                {"channel": r["channel"], "marginal_roi": r["marginal_roi"], "elasticity": r["elasticity"]}
                for r in brand_roi_sorted
            ],
        })

    # --- Output ---
    output_dir = project_root / "output" / "roi"
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_csv_path = output_dir / "channel_roi.csv"
    pd.DataFrame(roi_rows).to_csv(roi_csv_path, index=False)
    print(f"\nSaved: {roi_csv_path}")

    roi_json_path = output_dir / "roi_summary.json"
    with open(roi_json_path, "w", encoding="utf-8") as f:
        json.dump(roi_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {roi_json_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Channel ROI Ranking")
    print("=" * 60)
    for s in roi_summary:
        print(f"\n  Brand: {s['brand']} (R²: {s['r_squared']:.4f})")
        for i, ch in enumerate(s["channel_ranking"], 1):
            print(f"    {i}. {ch['channel']}: marginal ROI = {ch['marginal_roi']:.4f}, elasticity = {ch['elasticity']:.4f}")


if __name__ == "__main__":
    main()
