"""
J4: Funnel Analysis (Digital + LINE)

デジタル行動ログのセッションベースファネルと
LINEインタラクションの配信ファネルを分析する。

Usage:
    uv run python journey/funnel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.data_loader import (
    load_config,
    fetch_all,
    get_supabase_client,
)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "funnel"

# Digital funnel stages (ordered)
DIGITAL_FUNNEL_STAGES = ["page_view", "product_click", "add_to_cart", "purchase"]

# LINE funnel stages (ordered)
LINE_FUNNEL_STAGES = ["message_delivered", "message_opened", "message_clicked"]


# ============================================================
# 1. DATA FETCHING
# ============================================================


def fetch_digital_and_line(
    cfg: dict, client
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch digital_behavior_log and line_interaction."""
    ds = cfg["data_source"]

    if ds["type"] == "supabase":
        print("Fetching digital_behavior_log...")
        digital_rows = fetch_all(
            client,
            "digital_behavior_log",
            "unified_customer_id,event_name,session_id,event_datetime",
        )
        print(f"  digital_behavior_log: {len(digital_rows):,} rows")

        print("Fetching line_interaction...")
        line_rows = fetch_all(
            client,
            "line_interaction",
            "unified_customer_id,event_type,event_datetime",
        )
        print(f"  line_interaction: {len(line_rows):,} rows")
    else:
        csv_dir = Path(ds.get("csv_dir", "./data"))
        if not csv_dir.is_absolute():
            csv_dir = Path(__file__).parent.parent / csv_dir

        df_d = pd.read_csv(csv_dir / "digital_behavior_log.csv")
        cols_d = [
            "unified_customer_id",
            "event_name",
            "session_id",
            "event_datetime",
        ]
        digital_rows = df_d[
            [c for c in cols_d if c in df_d.columns]
        ].to_dict("records")
        print(f"  digital_behavior_log: {len(digital_rows):,} rows")

        df_l = pd.read_csv(csv_dir / "line_interaction.csv")
        cols_l = ["unified_customer_id", "event_type", "event_datetime"]
        line_rows = df_l[
            [c for c in cols_l if c in df_l.columns]
        ].to_dict("records")
        print(f"  line_interaction: {len(line_rows):,} rows")

    return pd.DataFrame(digital_rows), pd.DataFrame(line_rows)


# ============================================================
# 2. DIGITAL FUNNEL (SESSION-BASED)
# ============================================================


def compute_digital_funnel(df_digital: pd.DataFrame) -> pd.DataFrame:
    """Compute session-based digital funnel.

    For each session, check which funnel stages exist.
    A session 'reaches' a stage if that event_name appears in the session.
    """
    if df_digital.empty:
        return pd.DataFrame(
            columns=["stage", "sessions", "conversion_rate", "step_conversion_rate"]
        )

    # Group events by session
    session_events = (
        df_digital.groupby("session_id")["event_name"]
        .apply(set)
        .to_dict()
    )

    total_sessions = len(session_events)
    print(f"\n  Total sessions: {total_sessions:,}")

    # Count sessions reaching each stage
    stage_counts = []
    for stage in DIGITAL_FUNNEL_STAGES:
        count = sum(1 for events in session_events.values() if stage in events)
        stage_counts.append({"stage": stage, "sessions": count})

    df_funnel = pd.DataFrame(stage_counts)

    # Overall conversion rate (from total sessions)
    df_funnel["conversion_rate"] = df_funnel["sessions"] / total_sessions

    # Step conversion rate (from previous stage)
    step_rates = []
    for i, row in df_funnel.iterrows():
        if i == 0:
            step_rates.append(1.0)  # First stage relative to itself
        else:
            prev = df_funnel.iloc[i - 1]["sessions"]
            step_rates.append(row["sessions"] / prev if prev > 0 else 0.0)
    df_funnel["step_conversion_rate"] = step_rates

    # Round for readability
    df_funnel["conversion_rate"] = df_funnel["conversion_rate"].round(6)
    df_funnel["step_conversion_rate"] = df_funnel["step_conversion_rate"].round(6)

    return df_funnel


# ============================================================
# 3. LINE FUNNEL
# ============================================================


def compute_line_funnel(df_line: pd.DataFrame) -> pd.DataFrame:
    """Compute LINE interaction funnel.

    Count unique customers at each stage.
    A customer 'reaches' a stage if they have that event_type.
    """
    if df_line.empty:
        return pd.DataFrame(
            columns=["stage", "customers", "conversion_rate", "step_conversion_rate"]
        )

    # Count unique customers per event type
    customer_events = (
        df_line.groupby("unified_customer_id")["event_type"]
        .apply(set)
        .to_dict()
    )

    total_customers = len(customer_events)
    print(f"\n  LINE customers: {total_customers:,}")

    stage_counts = []
    for stage in LINE_FUNNEL_STAGES:
        count = sum(
            1 for events in customer_events.values() if stage in events
        )
        stage_counts.append({"stage": stage, "customers": count})

    df_funnel = pd.DataFrame(stage_counts)

    # Overall conversion rate
    df_funnel["conversion_rate"] = df_funnel["customers"] / total_customers

    # Step conversion rate
    step_rates = []
    for i, row in df_funnel.iterrows():
        if i == 0:
            step_rates.append(1.0)
        else:
            prev = df_funnel.iloc[i - 1]["customers"]
            step_rates.append(row["customers"] / prev if prev > 0 else 0.0)
    df_funnel["step_conversion_rate"] = step_rates

    df_funnel["conversion_rate"] = df_funnel["conversion_rate"].round(6)
    df_funnel["step_conversion_rate"] = df_funnel["step_conversion_rate"].round(6)

    return df_funnel


# ============================================================
# 4. MAIN
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  J4: Funnel Analysis (Digital + LINE)")
    print("=" * 60)

    cfg = load_config()
    client = get_supabase_client(cfg)

    df_digital, df_line = fetch_digital_and_line(cfg, client)
    if client:
        client.close()

    # Digital funnel
    print("\n--- Digital Funnel (Session-based) ---")
    df_digital_funnel = compute_digital_funnel(df_digital)

    # LINE funnel
    print("\n--- LINE Funnel (Customer-based) ---")
    df_line_funnel = compute_line_funnel(df_line)

    # Build summary JSON
    summary = {
        "digital_funnel": {
            "total_sessions": int(
                df_digital.groupby("session_id").ngroups
            )
            if not df_digital.empty
            else 0,
            "stages": [],
        },
        "line_funnel": {
            "total_customers": int(df_line["unified_customer_id"].nunique())
            if not df_line.empty
            else 0,
            "stages": [],
        },
    }

    for _, r in df_digital_funnel.iterrows():
        summary["digital_funnel"]["stages"].append(
            {
                "stage": r["stage"],
                "sessions": int(r["sessions"]),
                "conversion_rate": float(r["conversion_rate"]),
                "step_conversion_rate": float(r["step_conversion_rate"]),
            }
        )

    for _, r in df_line_funnel.iterrows():
        summary["line_funnel"]["stages"].append(
            {
                "stage": r["stage"],
                "customers": int(r["customers"]),
                "conversion_rate": float(r["conversion_rate"]),
                "step_conversion_rate": float(r["step_conversion_rate"]),
            }
        )

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_digital_funnel.to_csv(OUTPUT_DIR / "digital_funnel.csv", index=False)
    df_line_funnel.to_csv(OUTPUT_DIR / "line_funnel.csv", index=False)

    with open(OUTPUT_DIR / "funnel_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")

    # Print summary
    print("\n" + "=" * 60)
    print("  DIGITAL FUNNEL")
    print("=" * 60)
    print(
        f"\n  {'Stage':<20} {'Sessions':>10} {'Conv Rate':>10} {'Step Rate':>10}"
    )
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for _, r in df_digital_funnel.iterrows():
        print(
            f"  {r['stage']:<20} {r['sessions']:>10,} "
            f"{r['conversion_rate']:>10.2%} {r['step_conversion_rate']:>10.2%}"
        )

    # Drop-off summary
    if len(df_digital_funnel) >= 2:
        first_stage = df_digital_funnel.iloc[0]["sessions"]
        last_stage = df_digital_funnel.iloc[-1]["sessions"]
        overall_conv = last_stage / first_stage if first_stage > 0 else 0
        print(f"\n  Overall funnel conversion: {overall_conv:.2%}")
        print(f"  Total drop-off: {first_stage - last_stage:,} sessions")

    print("\n" + "=" * 60)
    print("  LINE FUNNEL")
    print("=" * 60)
    print(
        f"\n  {'Stage':<25} {'Customers':>10} {'Conv Rate':>10} {'Step Rate':>10}"
    )
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for _, r in df_line_funnel.iterrows():
        print(
            f"  {r['stage']:<25} {r['customers']:>10,} "
            f"{r['conversion_rate']:>10.2%} {r['step_conversion_rate']:>10.2%}"
        )

    if len(df_line_funnel) >= 2:
        first_stage = df_line_funnel.iloc[0]["customers"]
        last_stage = df_line_funnel.iloc[-1]["customers"]
        overall_conv = last_stage / first_stage if first_stage > 0 else 0
        print(f"\n  Overall funnel conversion: {overall_conv:.2%}")
        print(f"  Total drop-off: {first_stage - last_stage:,} customers")

    print("=" * 60)


if __name__ == "__main__":
    main()
