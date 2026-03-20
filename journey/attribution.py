"""
J2: Markov Chain Attribution Analysis

タッチポイント系列からマルコフ連鎖モデルを構築し、
各チャネルの除去効果（removal effect）に基づくアトリビューションを算出する。
比較用にラストタッチ・リニアアトリビューションも計算。

Usage:
    uv run python journey/attribution.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.data_loader import (
    load_config,
    fetch_all,
    fetch_data,
    get_supabase_client,
)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "attribution"


# ============================================================
# 1. TOUCHPOINT EXTRACTION (reuses config pattern from golden_path)
# ============================================================


def build_touchpoint_sequences(
    cfg: dict,
    dfs: dict[str, pd.DataFrame],
    df_purchase: pd.DataFrame,
) -> dict[str, list[tuple[str, datetime]]]:
    """Build unified touchpoint sequences per customer.

    Returns dict: customer_id -> sorted list of (code, datetime).
    Purchase events are appended as "PURCHASE".
    """
    tp_sources_cfg = cfg["data_source"]["tables"].get("touchpoint_sources", {})
    tp_reverse = cfg["_tp_reverse"]
    suppress_codes = cfg["_suppress_codes"]

    # Collect (customer_id, datetime, code)
    records: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

    for source_key, source_cfg in tp_sources_cfg.items():
        if source_key not in dfs:
            continue
        df_source = dfs[source_key]
        cid_col = source_cfg["customer_id_col"]
        dt_col = source_cfg["datetime_col"]
        classify_col = source_cfg.get("classify_col")
        reverse_map = tp_reverse.get(source_key, {})

        for _, row in df_source.iterrows():
            if classify_col:
                event_val = str(row.get(classify_col, ""))
                code = reverse_map.get(event_val)
                if code is None:
                    continue
            elif "__ALL__" in reverse_map:
                code = reverse_map["__ALL__"]
            else:
                continue
            if code in suppress_codes:
                continue
            try:
                dt = datetime.fromisoformat(
                    str(row[dt_col]).replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                continue
            records[str(row[cid_col])].append((dt, code))

    # Add purchase events
    pt_cfg = cfg["data_source"]["tables"]["purchase"]
    for _, row in df_purchase.iterrows():
        try:
            dt = datetime.fromisoformat(
                str(row[pt_cfg["datetime_col"]]).replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            continue
        records[str(row[pt_cfg["customer_id_col"]])].append((dt, "PURCHASE"))

    # Sort by datetime and return
    sequences: dict[str, list[tuple[str, datetime]]] = {}
    for cid, events in records.items():
        events.sort(key=lambda x: x[0])
        sequences[cid] = [(code, dt) for dt, code in events]

    return sequences


# ============================================================
# 2. MARKOV CHAIN ATTRIBUTION
# ============================================================


def build_transition_matrix(
    sequences: dict[str, list[tuple[str, datetime]]],
) -> tuple[dict[tuple[str, str], int], set[str]]:
    """Build raw transition counts from touchpoint sequences.

    Each journey: START -> touchpoints... -> PURCHASE or NULL.
    """
    transitions: Counter = Counter()
    channels: set[str] = set()

    for cid, seq in sequences.items():
        codes = [tp[0] for tp in seq]
        # Split into sub-journeys ending at PURCHASE or end of sequence
        journey: list[str] = ["START"]
        for code in codes:
            if code == "PURCHASE":
                journey.append("PURCHASE")
                # Record transitions for this journey
                for i in range(len(journey) - 1):
                    transitions[(journey[i], journey[i + 1])] += 1
                    if journey[i] not in ("START", "PURCHASE", "NULL"):
                        channels.add(journey[i])
                # Start new journey
                journey = ["START"]
            else:
                journey.append(code)
                channels.add(code)

        # Journey that didn't end in purchase -> NULL
        if len(journey) > 1:
            journey.append("NULL")
            for i in range(len(journey) - 1):
                transitions[(journey[i], journey[i + 1])] += 1
                if journey[i] not in ("START", "PURCHASE", "NULL"):
                    channels.add(journey[i])

    return transitions, channels


def compute_conversion_rate(
    transitions: dict[tuple[str, str], int],
    channels: set[str],
) -> float:
    """Compute overall conversion rate from transition matrix via absorbing Markov chain.

    States: START, channels, PURCHASE (absorbing), NULL (absorbing).
    Returns P(absorption into PURCHASE | starting from START).
    """
    all_states = ["START"] + sorted(channels)
    absorbing = {"PURCHASE", "NULL"}
    n = len(all_states)

    # Build transition probability matrix for transient states
    # Q[i][j] = P(go to transient state j | in transient state i)
    # R[i][k] = P(go to absorbing state k | in transient state i)
    state_idx = {s: i for i, s in enumerate(all_states)}

    # Outgoing counts per state
    out_counts: dict[str, int] = defaultdict(int)
    for (src, dst), cnt in transitions.items():
        out_counts[src] += cnt

    Q = [[0.0] * n for _ in range(n)]
    R_purchase = [0.0] * n  # P(-> PURCHASE | state i)
    R_null = [0.0] * n  # P(-> NULL | state i)

    for i, state in enumerate(all_states):
        total = out_counts.get(state, 0)
        if total == 0:
            # Dead-end state -> treat as going to NULL
            R_null[i] = 1.0
            continue
        for j, other in enumerate(all_states):
            cnt = transitions.get((state, other), 0)
            Q[i][j] = cnt / total
        R_purchase[i] = transitions.get((state, "PURCHASE"), 0) / total
        R_null[i] = transitions.get((state, "NULL"), 0) / total

    # Solve (I - Q) * N = I  ->  N = (I - Q)^{-1}
    # Then absorption probabilities B = N * R
    # We only need row 0 (START state) absorption into PURCHASE

    # Use iterative method (power series) for simplicity
    # P(PURCHASE | START) via simulation of absorbing chain
    # More robust: direct solve with numpy-free approach

    # Gaussian elimination to solve (I - Q) x = R_purchase
    # where x[i] = P(eventually reach PURCHASE | start at state i)
    # x[i] = R_purchase[i] + sum_j Q[i][j] * x[j]
    # => (I - Q) x = R_purchase

    # Build augmented matrix
    A = [[0.0] * (n + 1) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = (1.0 if i == j else 0.0) - Q[i][j]
        A[i][n] = R_purchase[i]

    # Gaussian elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(A[row][col]) > abs(A[max_row][col]):
                max_row = row
        A[col], A[max_row] = A[max_row], A[col]

        pivot = A[col][col]
        if abs(pivot) < 1e-12:
            continue
        for j in range(col, n + 1):
            A[col][j] /= pivot
        for row in range(n):
            if row == col:
                continue
            factor = A[row][col]
            for j in range(col, n + 1):
                A[row][j] -= factor * A[col][j]

    # x[i] = A[i][n]
    start_idx = state_idx["START"]
    return A[start_idx][n]


def markov_removal_effect(
    transitions: dict[tuple[str, str], int],
    channels: set[str],
) -> dict[str, float]:
    """Compute removal effect for each channel.

    Removal effect = 1 - P(conversion without channel) / P(conversion with all channels).
    """
    base_rate = compute_conversion_rate(transitions, channels)
    if base_rate <= 0:
        return {ch: 0.0 for ch in channels}

    removal_effects: dict[str, float] = {}
    for ch in channels:
        # Remove channel: redirect all transitions to/from ch
        modified = {}
        for (src, dst), cnt in transitions.items():
            if src == ch or dst == ch:
                continue
            modified[(src, dst)] = cnt

        # Transitions that went through ch now go to NULL
        for (src, dst), cnt in transitions.items():
            if dst == ch:
                modified[(src, "NULL")] = modified.get((src, "NULL"), 0) + cnt

        reduced_channels = channels - {ch}
        reduced_rate = compute_conversion_rate(modified, reduced_channels)
        removal_effects[ch] = 1.0 - (reduced_rate / base_rate) if base_rate > 0 else 0.0

    return removal_effects


def markov_attribution(
    sequences: dict[str, list[tuple[str, datetime]]],
) -> pd.DataFrame:
    """Compute Markov Chain attribution weights."""
    transitions, channels = build_transition_matrix(sequences)

    if not channels:
        return pd.DataFrame(columns=["channel", "removal_effect", "attribution_weight"])

    removal_effects = markov_removal_effect(transitions, channels)

    total_effect = sum(max(0, v) for v in removal_effects.values())
    rows = []
    for ch in sorted(channels):
        effect = max(0, removal_effects.get(ch, 0))
        weight = effect / total_effect if total_effect > 0 else 0.0
        rows.append(
            {
                "channel": ch,
                "removal_effect": round(effect, 6),
                "attribution_weight": round(weight, 6),
            }
        )

    return pd.DataFrame(rows).sort_values("attribution_weight", ascending=False)


# ============================================================
# 3. LAST-TOUCH & LINEAR ATTRIBUTION
# ============================================================


def last_touch_attribution(
    sequences: dict[str, list[tuple[str, datetime]]],
) -> pd.DataFrame:
    """Last-touch attribution: credit to last non-PURCHASE touchpoint before conversion."""
    channel_credits: Counter = Counter()
    total_conversions = 0

    for cid, seq in sequences.items():
        codes = [tp[0] for tp in seq]
        # Find each PURCHASE and attribute to preceding touchpoint
        for i, code in enumerate(codes):
            if code == "PURCHASE":
                total_conversions += 1
                # Look backward for last non-PURCHASE touchpoint
                for j in range(i - 1, -1, -1):
                    if codes[j] != "PURCHASE":
                        channel_credits[codes[j]] += 1
                        break

    rows = []
    for ch, cnt in channel_credits.most_common():
        rows.append(
            {
                "channel": ch,
                "conversions": cnt,
                "attribution_weight": round(cnt / total_conversions, 6)
                if total_conversions > 0
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def linear_attribution(
    sequences: dict[str, list[tuple[str, datetime]]],
) -> pd.DataFrame:
    """Linear attribution: equal credit to all touchpoints in path to conversion."""
    channel_credits: Counter = Counter()
    total_credit = 0.0

    for cid, seq in sequences.items():
        codes = [tp[0] for tp in seq]
        # Split into sub-journeys ending at PURCHASE
        journey: list[str] = []
        for code in codes:
            if code == "PURCHASE":
                # Distribute credit equally among touchpoints in this journey
                non_purchase = [c for c in journey if c != "PURCHASE"]
                if non_purchase:
                    credit = 1.0 / len(non_purchase)
                    for c in non_purchase:
                        channel_credits[c] += credit
                    total_credit += 1.0
                journey = []
            else:
                journey.append(code)

    rows = []
    for ch in sorted(channel_credits.keys(), key=lambda x: -channel_credits[x]):
        rows.append(
            {
                "channel": ch,
                "weighted_conversions": round(channel_credits[ch], 4),
                "attribution_weight": round(
                    channel_credits[ch] / total_credit, 6
                )
                if total_credit > 0
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 4. TRANSITION MATRIX EXPORT
# ============================================================


def export_transition_matrix(
    sequences: dict[str, list[tuple[str, datetime]]],
) -> pd.DataFrame:
    """Export full transition matrix with probabilities."""
    transitions, channels = build_transition_matrix(sequences)
    all_states = ["START"] + sorted(channels) + ["PURCHASE", "NULL"]

    out_counts: dict[str, int] = defaultdict(int)
    for (src, dst), cnt in transitions.items():
        out_counts[src] += cnt

    rows = []
    for src in all_states:
        total = out_counts.get(src, 0)
        if total == 0:
            continue
        for dst in all_states:
            cnt = transitions.get((src, dst), 0)
            if cnt > 0:
                rows.append(
                    {
                        "from_state": src,
                        "to_state": dst,
                        "count": cnt,
                        "probability": round(cnt / total, 6),
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# 5. MAIN
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  J2: Markov Chain Attribution Analysis")
    print("=" * 60)

    cfg = load_config()
    client = get_supabase_client(cfg)
    dfs = fetch_data(cfg, client)
    if client:
        client.close()

    df_purchase = dfs["purchase"]
    print(f"\nPurchase transactions: {len(df_purchase):,}")

    # Build touchpoint sequences
    print("\nBuilding touchpoint sequences...")
    sequences = build_touchpoint_sequences(cfg, dfs, df_purchase)
    n_customers = len(sequences)
    n_conversions = sum(
        1
        for seq in sequences.values()
        if any(tp[0] == "PURCHASE" for tp in seq)
    )
    print(f"  Customers with touchpoints: {n_customers:,}")
    print(f"  Customers with conversions: {n_conversions:,}")

    # Markov attribution
    print("\nComputing Markov Chain attribution...")
    df_markov = markov_attribution(sequences)

    # Last-touch attribution
    print("Computing last-touch attribution...")
    df_last_touch = last_touch_attribution(sequences)

    # Linear attribution
    print("Computing linear attribution...")
    df_linear = linear_attribution(sequences)

    # Transition matrix
    print("Building transition matrix...")
    df_transitions = export_transition_matrix(sequences)

    # Comparison table
    all_channels = sorted(
        set(df_markov["channel"].tolist())
        | set(df_last_touch["channel"].tolist())
        | set(df_linear["channel"].tolist())
    )

    markov_weights = dict(
        zip(df_markov["channel"], df_markov["attribution_weight"])
    )
    lt_weights = dict(
        zip(df_last_touch["channel"], df_last_touch["attribution_weight"])
    )
    lin_weights = dict(
        zip(df_linear["channel"], df_linear["attribution_weight"])
    )

    comparison_rows = []
    for ch in all_channels:
        comparison_rows.append(
            {
                "channel": ch,
                "markov_weight": markov_weights.get(ch, 0.0),
                "last_touch_weight": lt_weights.get(ch, 0.0),
                "linear_weight": lin_weights.get(ch, 0.0),
            }
        )
    df_comparison = pd.DataFrame(comparison_rows).sort_values(
        "markov_weight", ascending=False
    )

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_markov.to_csv(OUTPUT_DIR / "markov_attribution.csv", index=False)
    df_comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    df_transitions.to_csv(OUTPUT_DIR / "transition_matrix.csv", index=False)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")

    # Print summary
    print("\n" + "=" * 60)
    print("  ATTRIBUTION COMPARISON")
    print("=" * 60)
    print(
        f"\n  {'Channel':<16} {'Markov':>10} {'Last-Touch':>12} {'Linear':>10}"
    )
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*10}")
    for _, r in df_comparison.iterrows():
        print(
            f"  {r['channel']:<16} {r['markov_weight']:>10.4f} "
            f"{r['last_touch_weight']:>12.4f} {r['linear_weight']:>10.4f}"
        )

    print(f"\n  Total channels: {len(all_channels)}")
    print(f"  Transition matrix: {len(df_transitions)} edges")
    print("=" * 60)


if __name__ == "__main__":
    main()
