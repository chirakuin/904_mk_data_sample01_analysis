"""
Segment Transition Analysis (C4)

segment_membership からセグメント間の遷移行列を構築する。
四半期スナップショットを使用。

Usage:
    uv run python customer/segment_transition.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from lib.data_loader import fetch_all, get_supabase_client, load_config

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "segment_transition"

CID_COL = "unified_customer_id"


# ============================================================
# Data Fetch
# ============================================================


def fetch_segment_data(cfg: dict) -> pd.DataFrame:
    """Fetch segment_membership table."""
    client = get_supabase_client(cfg)
    if client is None:
        raise RuntimeError("Supabase client required")
    try:
        rows = fetch_all(
            client,
            "segment_membership",
            f"{CID_COL},segment_name,entered_at,exited_at,is_current_member",
        )
    finally:
        client.close()
    df = pd.DataFrame(rows)
    print(f"Fetched segment_membership: {len(df):,} rows")
    return df


# ============================================================
# Transition Analysis
# ============================================================


def build_quarterly_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Build quarterly snapshots: which segments each customer belongs to at each quarter."""
    df = df.copy()
    df["entered_at"] = pd.to_datetime(df["entered_at"])
    df["exited_at"] = pd.to_datetime(df["exited_at"], errors="coerce")

    # Determine date range
    min_date = df["entered_at"].min()
    max_date = df["entered_at"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        return pd.DataFrame()

    # Generate quarter-end dates
    quarters = pd.date_range(
        start=min_date.to_period("Q").start_time,
        end=max_date + pd.offsets.QuarterEnd(1),
        freq="QE",
    )

    snapshot_rows = []
    for q_date in quarters:
        # Active memberships at this quarter-end
        active = df[
            (df["entered_at"] <= q_date)
            & ((df["exited_at"].isna()) | (df["exited_at"] > q_date))
        ]
        for _, row in active.iterrows():
            snapshot_rows.append(
                {
                    CID_COL: row[CID_COL],
                    "quarter": str(q_date.to_period("Q")),
                    "segment_name": row["segment_name"],
                }
            )

    return pd.DataFrame(snapshot_rows)


def build_transition_matrix(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Build transition matrix from quarterly snapshots."""
    if snapshots.empty:
        return pd.DataFrame(), []

    quarters = sorted(snapshots["quarter"].unique())
    if len(quarters) < 2:
        return pd.DataFrame(), []

    # For each customer, get their segment set at each quarter
    # A customer may belong to multiple segments; use the primary one (first alphabetically)
    pivot = (
        snapshots.sort_values("segment_name")
        .drop_duplicates(subset=[CID_COL, "quarter"], keep="first")
        .pivot(index=CID_COL, columns="quarter", values="segment_name")
    )

    # Count transitions between consecutive quarters
    transition_counter: Counter = Counter()
    total_from: Counter = Counter()

    for i in range(len(quarters) - 1):
        q_curr = quarters[i]
        q_next = quarters[i + 1]
        if q_curr not in pivot.columns or q_next not in pivot.columns:
            continue

        mask = pivot[q_curr].notna() & pivot[q_next].notna()
        for _, row in pivot[mask].iterrows():
            seg_from = row[q_curr]
            seg_to = row[q_next]
            transition_counter[(seg_from, seg_to)] += 1
            total_from[seg_from] += 1

    # Build matrix
    all_segments = sorted(set(s for pair in transition_counter for s in pair))
    matrix_rows = []
    for seg_from in all_segments:
        row = {"from_segment": seg_from}
        for seg_to in all_segments:
            count = transition_counter.get((seg_from, seg_to), 0)
            prob = count / total_from[seg_from] if total_from[seg_from] > 0 else 0.0
            row[seg_to] = round(prob, 4)
        matrix_rows.append(row)

    df_matrix = pd.DataFrame(matrix_rows)

    # Build flow data (top transitions by volume)
    flow_data = []
    for (seg_from, seg_to), count in transition_counter.most_common():
        prob = count / total_from[seg_from] if total_from[seg_from] > 0 else 0.0
        flow_data.append(
            {
                "from_segment": seg_from,
                "to_segment": seg_to,
                "count": count,
                "probability": round(prob, 4),
            }
        )

    return df_matrix, flow_data


# ============================================================
# Output
# ============================================================


def save_outputs(df_matrix: pd.DataFrame, flow_data: list[dict]) -> None:
    """Save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_matrix.to_csv(OUTPUT_DIR / "transition_matrix.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'transition_matrix.csv'}")

    flow_output = {
        "total_transitions": sum(f["count"] for f in flow_data),
        "unique_transitions": len(flow_data),
        "flows": flow_data,
    }
    with open(OUTPUT_DIR / "segment_flow.json", "w", encoding="utf-8") as f:
        json.dump(flow_output, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'segment_flow.json'}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  Segment Transition Analysis (C4)")
    print("=" * 60)

    cfg = load_config()
    df = fetch_segment_data(cfg)

    if df.empty:
        print("ERROR: No segment data. Exiting.")
        sys.exit(1)

    print("\nBuilding quarterly snapshots...")
    snapshots = build_quarterly_snapshots(df)
    if snapshots.empty:
        print("ERROR: No snapshots generated. Exiting.")
        sys.exit(1)
    print(f"  Quarterly snapshots: {len(snapshots):,} rows")
    print(f"  Quarters: {sorted(snapshots['quarter'].unique())}")

    print("\nBuilding transition matrix...")
    df_matrix, flow_data = build_transition_matrix(snapshots)

    if df_matrix.empty:
        print("WARNING: No transitions found (need at least 2 quarters of data).")
        save_outputs(df_matrix, flow_data)
        return

    save_outputs(df_matrix, flow_data)

    # Print summary
    print("\n" + "=" * 60)
    print("  SEGMENT TRANSITION SUMMARY")
    print("=" * 60)

    total = sum(f["count"] for f in flow_data)
    print(f"\nTotal transitions: {total:,}")
    print(f"Unique transition types: {len(flow_data)}")

    print(f"\nTop 10 transitions by volume:")
    print(f"  {'From':<25} {'To':<25} {'Count':>7} {'Prob':>7}")
    print(f"  {'-'*25} {'-'*25} {'-'*7} {'-'*7}")
    for f in flow_data[:10]:
        print(
            f"  {f['from_segment']:<25} {f['to_segment']:<25} "
            f"{f['count']:>7,} {f['probability']:>7.1%}"
        )


if __name__ == "__main__":
    main()
