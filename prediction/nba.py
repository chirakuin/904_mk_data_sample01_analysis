"""
Next Best Action (P2)

顧客サマリーと購買履歴から、ルールベースで顧客ごとのアクション推薦を生成。
クラスタリング結果やLTV予測結果が存在する場合はそれらも統合する。

Usage:
    uv run python prediction/nba.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from lib.data_loader import fetch_all, get_supabase_client, load_config

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "nba"

CLUSTER_PATH = PROJECT_ROOT / "output" / "clustering" / "cluster_assignments.csv"
LTV_PATH = PROJECT_ROOT / "output" / "ltv" / "customer_ltv.csv"

# Rule thresholds
CHURN_HIGH_THRESHOLD = 0.7
ENGAGEMENT_HIGH_THRESHOLD = 70
RECENCY_INACTIVE_DAYS = 60
NEW_CUSTOMER_DAYS = 90

# Reference date for recency calculation (from config or fallback)
REFERENCE_DATE = datetime(2025, 3, 10)


# ============================================================
# Data Fetch
# ============================================================


def fetch_customer_summary(cfg: dict) -> pd.DataFrame:
    """Fetch v_customer_summary with all columns."""
    client = get_supabase_client(cfg)
    if client is None:
        raise RuntimeError("Supabase client required")
    try:
        rows = fetch_all(client, "v_customer_summary", "*")
    finally:
        client.close()
    df = pd.DataFrame(rows)
    print(f"Fetched v_customer_summary: {len(df):,} rows, {len(df.columns)} columns")
    return df


def fetch_purchase_transactions(cfg: dict) -> pd.DataFrame:
    """Fetch purchase_transaction for recency calculation."""
    client = get_supabase_client(cfg)
    if client is None:
        raise RuntimeError("Supabase client required")
    try:
        rows = fetch_all(
            client,
            "purchase_transaction",
            "unified_customer_id,purchase_datetime",
        )
    finally:
        client.close()
    df = pd.DataFrame(rows)
    print(f"Fetched purchase_transaction: {len(df):,} rows")
    return df


# ============================================================
# Optional Data Integration
# ============================================================


def load_cluster_assignments() -> pd.DataFrame | None:
    """Load cluster assignments if available."""
    if CLUSTER_PATH.exists():
        df = pd.read_csv(CLUSTER_PATH)
        print(f"Loaded cluster assignments: {len(df):,} rows")
        return df
    else:
        print(f"WARNING: Cluster assignments not found at {CLUSTER_PATH} (skipping)")
        return None


def load_ltv_predictions() -> pd.DataFrame | None:
    """Load LTV predictions if available."""
    if LTV_PATH.exists():
        df = pd.read_csv(LTV_PATH)
        print(f"Loaded LTV predictions: {len(df):,} rows")
        return df
    else:
        print(f"WARNING: LTV predictions not found at {LTV_PATH} (skipping)")
        return None


# ============================================================
# Preprocessing
# ============================================================


def compute_recency(df_purchase: pd.DataFrame, ref_date: datetime) -> pd.DataFrame:
    """Compute days since last purchase per customer."""
    df = df_purchase.copy()
    df["purchase_datetime"] = pd.to_datetime(df["purchase_datetime"])
    last_purchase = (
        df.groupby("unified_customer_id")["purchase_datetime"]
        .max()
        .reset_index()
    )
    last_purchase.columns = ["unified_customer_id", "last_purchase_date"]
    last_purchase["days_since_last_purchase"] = (
        (ref_date - last_purchase["last_purchase_date"]).dt.total_seconds() / 86400
    ).astype(float)
    return last_purchase


def prepare_customers(
    df_cust: pd.DataFrame,
    df_recency: pd.DataFrame,
    df_clusters: pd.DataFrame | None,
    df_ltv: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge customer data with recency, clusters, and LTV predictions."""
    # Convert all numeric columns to float to avoid int/float merge conflicts
    df = df_cust.copy()
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = df[col].astype(float)

    # Merge recency
    df = df.merge(df_recency, on="unified_customer_id", how="left")
    df["days_since_last_purchase"] = df["days_since_last_purchase"].fillna(9999.0)

    # Merge clusters if available
    if df_clusters is not None and "unified_customer_id" in df_clusters.columns:
        df = df.merge(df_clusters, on="unified_customer_id", how="left")
        print(f"  Merged cluster labels")

    # Merge LTV predictions if available (override clv_12m)
    if df_ltv is not None and "unified_customer_id" in df_ltv.columns:
        ltv_col = None
        for candidate in ["predicted_clv_12m", "clv_12m_predicted", "predicted_ltv"]:
            if candidate in df_ltv.columns:
                ltv_col = candidate
                break
        if ltv_col:
            df = df.merge(
                df_ltv[["unified_customer_id", ltv_col]],
                on="unified_customer_id",
                how="left",
            )
            # Use predicted CLV instead of original
            mask = df[ltv_col].notna()
            if mask.any():
                df.loc[mask, "clv_12m"] = df.loc[mask, ltv_col]
                print(f"  Replaced clv_12m with {ltv_col} for {mask.sum():,} customers")

    # Ensure numeric
    for col in ["churn_risk_score", "clv_12m", "engagement_score", "monthly_spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Parse first_known_date
    if "first_known_date" in df.columns:
        df["first_known_date"] = pd.to_datetime(df["first_known_date"], errors="coerce")

    return df


# ============================================================
# Rule Engine
# ============================================================


def assign_actions(df: pd.DataFrame, ref_date: datetime) -> pd.DataFrame:
    """Apply rule-based action assignment to each customer."""
    clv_median = df["clv_12m"].median() if "clv_12m" in df.columns else 0
    spend_median = df["monthly_spend"].median() if "monthly_spend" in df.columns else 0
    new_cutoff = ref_date - timedelta(days=NEW_CUSTOMER_DAYS)

    actions = []
    for _, row in df.iterrows():
        cid = row.get("unified_customer_id", "")
        churn = float(row.get("churn_risk_score", 0))
        clv = float(row.get("clv_12m", 0))
        engagement = float(row.get("engagement_score", 0))
        monthly = float(row.get("monthly_spend", 0))
        recency_days = float(row.get("days_since_last_purchase", 9999))
        first_date = row.get("first_known_date", None)
        status = str(row.get("status", "")).lower()

        # Rule 1: High churn + high LTV
        if churn > CHURN_HIGH_THRESHOLD and clv > clv_median:
            actions.append({
                "unified_customer_id": cid,
                "action": "retention_coupon",
                "channel": "LINE",
                "message_template": "Special loyalty coupon for valued customer",
                "priority": "high",
                "reason": f"High churn risk ({churn:.2f}) with high LTV ({clv:,.0f})",
            })
            continue

        # Rule 2: High churn + low LTV
        if churn > CHURN_HIGH_THRESHOLD:
            actions.append({
                "unified_customer_id": cid,
                "action": "reactivation_push",
                "channel": "push",
                "message_template": "We miss you! Check out what's new",
                "priority": "medium",
                "reason": f"High churn risk ({churn:.2f}) with low LTV ({clv:,.0f})",
            })
            continue

        # Rule 3: No purchase in 60+ days + active
        if recency_days >= RECENCY_INACTIVE_DAYS and status in ("active", ""):
            actions.append({
                "unified_customer_id": cid,
                "action": "reminder",
                "channel": "email",
                "message_template": "It's been a while! Here are picks for you",
                "priority": "medium",
                "reason": f"No purchase in {int(recency_days)} days, still active",
            })
            continue

        # Rule 4: High engagement + subscription potential
        if engagement > ENGAGEMENT_HIGH_THRESHOLD and monthly > spend_median:
            actions.append({
                "unified_customer_id": cid,
                "action": "cross_sell",
                "channel": "LINE",
                "message_template": "Recommended products based on your favorites",
                "priority": "medium",
                "reason": f"High engagement ({engagement:.0f}) + high spend ({monthly:,.0f})",
            })
            continue

        # Rule 5: New customer
        if first_date is not None and pd.notna(first_date) and first_date >= new_cutoff:
            actions.append({
                "unified_customer_id": cid,
                "action": "onboarding_series",
                "channel": "email",
                "message_template": "Welcome! Here's how to get the most out of your membership",
                "priority": "high",
                "reason": f"New customer (joined {first_date.strftime('%Y-%m-%d') if hasattr(first_date, 'strftime') else first_date})",
            })
            continue

        # Rule 6: Default
        actions.append({
            "unified_customer_id": cid,
            "action": "no_action",
            "channel": "-",
            "message_template": "-",
            "priority": "low",
            "reason": "No action criteria met",
        })

    return pd.DataFrame(actions)


# ============================================================
# Output
# ============================================================


def save_outputs(
    df_actions: pd.DataFrame,
    df_customers: pd.DataFrame,
) -> None:
    """Save all outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Add cluster column if available
    if "cluster" in df_customers.columns:
        cluster_map = df_customers.drop_duplicates("unified_customer_id").set_index("unified_customer_id")["cluster"]
        df_actions["cluster"] = df_actions["unified_customer_id"].map(cluster_map)

    # Customer actions CSV
    df_actions.to_csv(OUTPUT_DIR / "customer_actions.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'customer_actions.csv'}")

    # Action summary JSON
    action_counts = df_actions["action"].value_counts().to_dict()
    priority_counts = df_actions["priority"].value_counts().to_dict()
    channel_counts = df_actions.groupby("action")["channel"].first().to_dict()

    summary = {
        "total_customers": len(df_actions),
        "action_distribution": {
            k: {"count": int(v), "pct": round(v / len(df_actions) * 100, 1)}
            for k, v in action_counts.items()
        },
        "priority_distribution": {
            k: {"count": int(v), "pct": round(v / len(df_actions) * 100, 1)}
            for k, v in priority_counts.items()
        },
        "has_cluster_data": "cluster" in df_actions.columns,
        "has_ltv_prediction": bool(LTV_PATH.exists()),
    }

    with open(OUTPUT_DIR / "action_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {OUTPUT_DIR / 'action_summary.json'}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  Next Best Action (P2)")
    print("=" * 60)

    cfg = load_config()

    # Use reference date from config if available
    ref_date = REFERENCE_DATE
    windows = cfg.get("windows", {})
    if "_data_end_date" in windows:
        ref_date = datetime.combine(windows["_data_end_date"], datetime.min.time())
    print(f"Reference date: {ref_date.date()}")

    print("\nFetching data...")
    df_cust = fetch_customer_summary(cfg)
    df_purchase = fetch_purchase_transactions(cfg)

    if df_cust.empty:
        print("ERROR: No customer data fetched. Exiting.")
        sys.exit(1)

    print("\nLoading optional data sources...")
    df_clusters = load_cluster_assignments()
    df_ltv = load_ltv_predictions()

    print("\nComputing recency...")
    df_recency = compute_recency(df_purchase, ref_date)
    print(f"  Recency computed for {len(df_recency):,} customers")

    print("\nPreparing customer data...")
    df_prepared = prepare_customers(df_cust, df_recency, df_clusters, df_ltv)

    print("\nAssigning actions...")
    df_actions = assign_actions(df_prepared, ref_date)
    print(f"  Actions assigned: {len(df_actions):,} customers")

    save_outputs(df_actions, df_prepared)

    # Print summary
    print("\n" + "=" * 60)
    print("  NBA SUMMARY")
    print("=" * 60)

    action_dist = df_actions["action"].value_counts()
    print(f"\nAction distribution:")
    for action, count in action_dist.items():
        pct = count / len(df_actions) * 100
        print(f"  {action:<25} {count:>6,} ({pct:>5.1f}%)")

    priority_dist = df_actions["priority"].value_counts()
    print(f"\nPriority distribution:")
    for priority, count in priority_dist.items():
        pct = count / len(df_actions) * 100
        print(f"  {priority:<10} {count:>6,} ({pct:>5.1f}%)")

    high_priority = df_actions[df_actions["priority"] == "high"]
    print(f"\nTop priority actions: {len(high_priority):,} customers")
    if not high_priority.empty:
        hp_actions = high_priority["action"].value_counts()
        for action, count in hp_actions.items():
            print(f"  {action}: {count:,}")

    if "cluster" in df_actions.columns:
        print(f"\nActions by cluster:")
        ct = pd.crosstab(df_actions["cluster"], df_actions["action"])
        print(ct.to_string())


if __name__ == "__main__":
    main()
