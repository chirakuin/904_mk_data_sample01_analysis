"""
Cohort Retention Analysis (C3)

顧客の first_known_date 月でコホートを定義し、月次リテンション率を追跡する。

Usage:
    uv run python customer/cohort.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from lib.data_loader import fetch_all, get_supabase_client, load_config

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "cohort"

CID_COL = "unified_customer_id"


# ============================================================
# Data Fetch
# ============================================================


def fetch_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch customer_profile and purchase_transaction."""
    client = get_supabase_client(cfg)
    if client is None:
        raise RuntimeError("Supabase client required")
    try:
        cp_rows = fetch_all(client, "customer_profile", f"{CID_COL},first_known_date")
        pt_rows = fetch_all(client, "purchase_transaction", f"{CID_COL},purchase_datetime")
    finally:
        client.close()

    df_cust = pd.DataFrame(cp_rows)
    df_purch = pd.DataFrame(pt_rows)
    print(f"Fetched customer_profile: {len(df_cust):,} rows")
    print(f"Fetched purchase_transaction: {len(df_purch):,} rows")
    return df_cust, df_purch


# ============================================================
# Cohort Analysis
# ============================================================


def build_retention_matrix(
    df_cust: pd.DataFrame, df_purch: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Build cohort retention matrix."""
    # Parse dates
    df_cust = df_cust.copy()
    df_cust["first_known_date"] = pd.to_datetime(df_cust["first_known_date"])
    df_cust["cohort"] = df_cust["first_known_date"].dt.to_period("M")

    df_purch = df_purch.copy()
    df_purch["purchase_date"] = pd.to_datetime(df_purch["purchase_datetime"])
    df_purch["purchase_month"] = df_purch["purchase_date"].dt.to_period("M")

    # Merge cohort info onto purchases
    merged = df_purch.merge(
        df_cust[[CID_COL, "cohort", "first_known_date"]],
        on=CID_COL,
        how="inner",
    )

    # Compute months since cohort start
    merged["months_since"] = (
        merged["purchase_month"].astype(int) - merged["cohort"].astype(int)
    )
    # Keep only non-negative
    merged = merged[merged["months_since"] >= 0]

    # Cohort sizes
    cohort_sizes = df_cust.groupby("cohort")[CID_COL].nunique()

    # Active customers per cohort per month
    active = (
        merged.groupby(["cohort", "months_since"])[CID_COL]
        .nunique()
        .reset_index(name="active_customers")
    )

    # Build retention matrix
    cohorts = sorted(cohort_sizes.index)
    max_months = int(active["months_since"].max()) if len(active) > 0 else 0

    retention_data = []
    for cohort in cohorts:
        row = {"cohort": str(cohort), "cohort_size": int(cohort_sizes[cohort])}
        for m in range(max_months + 1):
            mask = (active["cohort"] == cohort) & (active["months_since"] == m)
            n_active = int(active.loc[mask, "active_customers"].sum()) if mask.any() else 0
            rate = n_active / cohort_sizes[cohort] if cohort_sizes[cohort] > 0 else 0.0
            row[f"month_{m}"] = round(rate, 4)
        retention_data.append(row)

    df_retention = pd.DataFrame(retention_data)

    # Summary: retention at month 1, 3, 6, 12
    summary = {
        "total_cohorts": len(cohorts),
        "total_customers": int(cohort_sizes.sum()),
        "cohorts": [],
    }
    for row in retention_data:
        cohort_summary = {
            "cohort": row["cohort"],
            "size": row["cohort_size"],
        }
        for m in [1, 3, 6, 12]:
            key = f"month_{m}"
            cohort_summary[f"retention_month_{m}"] = row.get(key, None)
        summary["cohorts"].append(cohort_summary)

    return df_retention, summary


# ============================================================
# Output
# ============================================================


def save_outputs(df_retention: pd.DataFrame, summary: dict) -> None:
    """Save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_retention.to_csv(OUTPUT_DIR / "retention_matrix.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'retention_matrix.csv'}")

    with open(OUTPUT_DIR / "cohort_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {OUTPUT_DIR / 'cohort_summary.json'}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  Cohort Retention Analysis (C3)")
    print("=" * 60)

    cfg = load_config()
    df_cust, df_purch = fetch_data(cfg)

    if df_cust.empty:
        print("ERROR: No customer data. Exiting.")
        sys.exit(1)
    if df_purch.empty:
        print("ERROR: No purchase data. Exiting.")
        sys.exit(1)

    print("\nBuilding retention matrix...")
    df_retention, summary = build_retention_matrix(df_cust, df_purch)

    save_outputs(df_retention, summary)

    # Print summary
    print("\n" + "=" * 60)
    print("  COHORT RETENTION SUMMARY")
    print("=" * 60)
    print(f"\nTotal cohorts: {summary['total_cohorts']}")
    print(f"Total customers: {summary['total_customers']:,}")

    print(f"\n{'Cohort':<10} {'Size':>6} {'M1':>7} {'M3':>7} {'M6':>7} {'M12':>7}")
    print(f"{'-'*10} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for c in summary["cohorts"]:
        m1 = f"{c['retention_month_1']:.1%}" if c.get("retention_month_1") is not None else "n/a"
        m3 = f"{c['retention_month_3']:.1%}" if c.get("retention_month_3") is not None else "n/a"
        m6 = f"{c['retention_month_6']:.1%}" if c.get("retention_month_6") is not None else "n/a"
        m12 = f"{c['retention_month_12']:.1%}" if c.get("retention_month_12") is not None else "n/a"
        print(f"{c['cohort']:<10} {c['size']:>6,} {m1:>7} {m3:>7} {m6:>7} {m12:>7}")


if __name__ == "__main__":
    main()
