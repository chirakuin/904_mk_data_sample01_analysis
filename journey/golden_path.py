"""
Golden Path Analysis v2 - 汎用ゴールデンパス分析フレームワーク

config.yaml で定義されたデータソース・タッチポイント・アウトカムに基づき、
高価値顧客に共通するタッチポイント順序（ゴールデンパス）を特定する。

特徴:
  - 時系列リーク防止（観察窓/成果判定窓分離）
  - 統計的厳密性（Fisher検定・95%CI・Bootstrap安定性）
  - 日次/週次の2粒度比較
  - PURCHASE除外モードによるナーチャリング導線分析
  - config.yaml による設定外出し（データソース・タッチポイント・アウトカム）
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from itertools import islice
from pathlib import Path

import httpx
import pandas as pd
from scipy.stats import fisher_exact

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.data_loader import (
    load_config,
    fetch_all,
    fetch_data,
    get_supabase_client,
    ensure_first_date_col,
)

# ============================================================
# 1. CONFIGURATION (config.yaml driven)
# ============================================================

CFG = load_config()

# Derived constants
OBSERVATION_DAYS = CFG["windows"]["_observation_days"]
OUTCOME_DAYS = CFG["windows"]["_outcome_days"]
TOTAL_WINDOW = CFG["windows"]["_total_window"]
DATA_END_DATE = CFG["windows"]["_data_end_date"]
ELIGIBILITY_CUTOFF = CFG["windows"]["_eligibility_cutoff"]

MIN_PATH_LENGTH = CFG["analysis"]["min_path_length"]
BOOTSTRAP_ITER = CFG["analysis"]["bootstrap_iterations"]
BOOTSTRAP_SAMPLE_RATIO = CFG["analysis"]["bootstrap_sample_ratio"]
STABILITY_THRESHOLD = CFG["analysis"]["stability_threshold"]
N_GRAM_SIZES = CFG["analysis"]["ngram_sizes"]
FIRST_N_SIZES = CFG["analysis"]["first_n_sizes"]
TOP_K_REPORT = CFG["analysis"]["top_k_report"]

OUTPUT_DIR = Path(CFG["output"]["dir"])
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = Path(__file__).parent.parent / OUTPUT_DIR

OUTCOME_THRESHOLD = CFG["outcome"]["threshold"]


def min_support(n_eligible: int) -> int:
    floor = CFG["analysis"]["min_support_floor"]
    ratio = CFG["analysis"]["min_support_ratio"]
    return max(floor, int(n_eligible * ratio))


# ============================================================
# 3. WINDOW ASSIGNMENT
# ============================================================


def assign_windows(df_customer: pd.DataFrame) -> pd.DataFrame:
    """Compute observation/outcome windows per customer and filter eligible."""
    first_date_col = CFG["data_source"]["tables"]["customer"]["first_date_col"]
    if first_date_col not in df_customer.columns:
        raise ValueError(
            f"{first_date_col} not found in customer data. "
            f"Check config.yaml data_source.tables.customer.first_date_col"
        )

    df = df_customer.copy()
    df["first_known_date"] = pd.to_datetime(df[first_date_col]).dt.date

    # Eligibility: first_known_date <= ELIGIBILITY_CUTOFF
    df = df[df["first_known_date"] <= ELIGIBILITY_CUTOFF].copy()

    df["obs_start"] = df["first_known_date"]
    df["obs_end"] = df["first_known_date"].apply(
        lambda d: d + timedelta(days=OBSERVATION_DAYS - 1)
    )
    df["out_start"] = df["first_known_date"].apply(
        lambda d: d + timedelta(days=OBSERVATION_DAYS)
    )
    df["out_end"] = df["first_known_date"].apply(
        lambda d: d + timedelta(days=TOTAL_WINDOW - 1)
    )

    print(f"\nWindow assignment:")
    print(f"  Eligible customers: {len(df):,}")
    print(f"  Eligibility cutoff: first_known_date <= {ELIGIBILITY_CUTOFF}")
    return df


# ============================================================
# 4. OUTCOME LABELING
# ============================================================


def label_outcomes(
    df_customers: pd.DataFrame, df_purchase: pd.DataFrame
) -> pd.DataFrame:
    """Label customers as outcome=1 (threshold+ purchases in outcome window) or 0."""
    pt_cfg = CFG["data_source"]["tables"]["purchase"]
    cid_col = pt_cfg["customer_id_col"]
    dt_col = pt_cfg["datetime_col"]

    df_p = df_purchase.copy()
    df_p["purchase_date"] = pd.to_datetime(df_p[dt_col]).dt.date

    # Merge with customer windows
    cust_cid_col = CFG["data_source"]["tables"]["customer"]["customer_id_col"]
    merged = df_p.merge(
        df_customers[[cust_cid_col, "out_start", "out_end"]],
        left_on=cid_col,
        right_on=cust_cid_col,
        how="inner",
    )

    # Filter to outcome window
    in_window = merged[
        (merged["purchase_date"] >= merged["out_start"])
        & (merged["purchase_date"] <= merged["out_end"])
    ]

    # Count purchases per customer in outcome window
    purchase_counts = (
        in_window.groupby(cust_cid_col).size().reset_index(name="outcome_purchases")
    )

    df_out = df_customers.merge(purchase_counts, on=cust_cid_col, how="left")
    df_out["outcome_purchases"] = df_out["outcome_purchases"].fillna(0).astype(int)
    df_out["outcome"] = (df_out["outcome_purchases"] >= OUTCOME_THRESHOLD).astype(int)

    n_pos = df_out["outcome"].sum()
    n_neg = len(df_out) - n_pos
    print(f"\nOutcome labeling:")
    print(f"  Outcome=1 ({OUTCOME_THRESHOLD}+ purchases): {n_pos:,} ({n_pos/len(df_out)*100:.1f}%)")
    print(f"  Outcome=0 (0-{OUTCOME_THRESHOLD-1} purchases): {n_neg:,} ({n_neg/len(df_out)*100:.1f}%)")

    return df_out


# ============================================================
# 5. TOUCHPOINT EXTRACTION
# ============================================================

def extract_touchpoints(
    dfs: dict[str, pd.DataFrame],
    df_customers: pd.DataFrame,
) -> pd.DataFrame:
    """Extract and unify touchpoints from all tables within observation windows.
    Driven by config.yaml touchpoint_mapping."""
    cust_cid_col = CFG["data_source"]["tables"]["customer"]["customer_id_col"]
    cw = df_customers.drop_duplicates(cust_cid_col).set_index(cust_cid_col)
    customer_windows = cw[["obs_start", "obs_end"]].to_dict("index")

    suppress_codes = CFG["_suppress_codes"]
    tp_sources_cfg = CFG["data_source"]["tables"].get("touchpoint_sources", {})
    tp_reverse = CFG["_tp_reverse"]

    records: list[dict] = []

    def add_events(customer_id: str, dt_str: str, code: str) -> None:
        if code in suppress_codes:
            return
        if customer_id not in customer_windows:
            return
        w = customer_windows[customer_id]
        try:
            dt = datetime.fromisoformat(str(dt_str).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return
        event_date = dt.date()
        if w["obs_start"] <= event_date <= w["obs_end"]:
            records.append(
                {
                    "unified_customer_id": customer_id,
                    "event_datetime": dt,
                    "event_date": event_date,
                    "code": code,
                }
            )

    # Process each touchpoint source from config
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
            add_events(str(row[cid_col]), str(row[dt_col]), code)

    # Purchase touchpoints (from purchase table, always mapped to "PURCHASE")
    pt_cfg = CFG["data_source"]["tables"]["purchase"]
    for _, row in dfs["purchase"].iterrows():
        add_events(
            str(row[pt_cfg["customer_id_col"]]),
            str(row[pt_cfg["datetime_col"]]),
            "PURCHASE",
        )

    df_tp = pd.DataFrame(records)
    print(f"\nTouchpoint extraction:")
    print(f"  Total touchpoint events: {len(df_tp):,}")
    if len(df_tp) > 0:
        print(f"  Code distribution:")
        for code, cnt in df_tp["code"].value_counts().items():
            print(f"    {code}: {cnt:,}")

    return df_tp


# ============================================================
# 6. SEQUENCE BUILDING
# ============================================================


def build_sequences(
    df_touchpoints: pd.DataFrame,
    df_customers: pd.DataFrame,
    granularity: str = "daily",
) -> dict[str, list[str]]:
    """Build deduplicated touchpoint sequences per customer.

    Args:
        granularity: 'daily' or 'weekly'

    Returns:
        Dict mapping customer_id → ordered list of touchpoint codes
    """
    if df_touchpoints.empty:
        return {}

    customer_obs_start = df_customers.set_index("unified_customer_id")[
        "obs_start"
    ].to_dict()

    df = df_touchpoints.copy()

    if granularity == "weekly":
        df["period"] = df.apply(
            lambda r: (r["event_date"] - customer_obs_start[r["unified_customer_id"]]).days // 7,
            axis=1,
        )
    else:  # daily
        df["period"] = df["event_date"]

    # Sort by period then by exact timestamp
    df = df.sort_values(["unified_customer_id", "period", "event_datetime"])

    # Deduplicate: same customer, same period, same code → keep first
    df = df.drop_duplicates(
        subset=["unified_customer_id", "period", "code"], keep="first"
    )

    # Build sequences
    sequences: dict[str, list[str]] = {}
    for cid, group in df.groupby("unified_customer_id"):
        sequences[cid] = group["code"].tolist()

    return sequences


# ============================================================
# 7. PATH EXTRACTION
# ============================================================


def extract_first_n(sequences: dict[str, list[str]], n: int) -> dict[str, list[str]]:
    """Extract first-N touchpoints per customer."""
    return {
        cid: seq[:n] for cid, seq in sequences.items() if len(seq) >= n
    }


def extract_ngrams(sequences: dict[str, list[str]], n: int) -> dict[str, list[tuple[str, ...]]]:
    """Extract all N-grams per customer."""
    result: dict[str, list[tuple[str, ...]]] = {}
    for cid, seq in sequences.items():
        if len(seq) < n:
            continue
        grams = []
        for i in range(len(seq) - n + 1):
            grams.append(tuple(seq[i : i + n]))
        result[cid] = grams
    return result


def count_paths(
    customer_paths: dict[str, list[str] | tuple[str, ...]],
    customer_outcomes: dict[str, int],
    min_sup: int,
    is_ngram: bool = False,
) -> pd.DataFrame:
    """Count path frequency in outcome=1 and outcome=0 groups."""
    pos_counter: Counter = Counter()
    neg_counter: Counter = Counter()
    n_pos = sum(1 for v in customer_outcomes.values() if v == 1)
    n_neg = sum(1 for v in customer_outcomes.values() if v == 0)

    for cid, paths in customer_paths.items():
        outcome = customer_outcomes.get(cid)
        if outcome is None:
            continue
        if is_ngram:
            # paths is list of tuples; count unique paths per customer
            unique_paths = set(paths)
            for p in unique_paths:
                if outcome == 1:
                    pos_counter[p] += 1
                else:
                    neg_counter[p] += 1
        else:
            # paths is a single sequence (list of str)
            p = tuple(paths)
            if outcome == 1:
                pos_counter[p] += 1
            else:
                neg_counter[p] += 1

    # Combine
    all_paths = set(pos_counter.keys()) | set(neg_counter.keys())
    rows = []
    for p in all_paths:
        a = pos_counter.get(p, 0)  # outcome=1, has path
        b = neg_counter.get(p, 0)  # outcome=0, has path
        total = a + b
        if total < min_sup:
            continue
        rows.append(
            {
                "path": " → ".join(p),
                "path_tuple": p,
                "count_pos": a,
                "count_neg": b,
                "count_total": total,
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 8. STATISTICAL ANALYSIS
# ============================================================


def compute_metrics(df_paths: pd.DataFrame) -> pd.DataFrame:
    """Compute support, lift, diff, OR, 95% CI, p-value for each path."""
    if df_paths.empty:
        return df_paths

    df = df_paths.copy()
    n_pos = df["n_pos"].iloc[0]
    n_neg = df["n_neg"].iloc[0]

    df["support_pos"] = df["count_pos"] / n_pos
    df["support_neg"] = df["count_neg"] / n_neg

    # Lift: support_pos / support_neg (avoid div by zero)
    df["lift"] = df.apply(
        lambda r: r["support_pos"] / r["support_neg"] if r["support_neg"] > 0 else float("inf"),
        axis=1,
    )

    # Difference
    df["diff"] = df["support_pos"] - df["support_neg"]

    # Odds ratio and Fisher exact test
    or_vals = []
    ci_lo_vals = []
    ci_hi_vals = []
    p_vals = []

    for _, r in df.iterrows():
        a = r["count_pos"]  # outcome=1, has path
        b = r["count_neg"]  # outcome=0, has path
        c = n_pos - a  # outcome=1, no path
        d = n_neg - b  # outcome=0, no path

        # Odds ratio
        if b * c > 0:
            odds_ratio = (a * d) / (b * c)
        else:
            odds_ratio = float("inf")

        # 95% CI for log(OR)
        if a > 0 and b > 0 and c > 0 and d > 0:
            se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            ln_or = math.log(odds_ratio)
            ci_lo = math.exp(ln_or - 1.96 * se)
            ci_hi = math.exp(ln_or + 1.96 * se)
        else:
            ci_lo = float("nan")
            ci_hi = float("nan")

        # Fisher exact test
        table = [[a, b], [c, d]]
        try:
            _, p_val = fisher_exact(table, alternative="two-sided")
        except ValueError:
            p_val = float("nan")

        or_vals.append(odds_ratio)
        ci_lo_vals.append(ci_lo)
        ci_hi_vals.append(ci_hi)
        p_vals.append(p_val)

    df["odds_ratio"] = or_vals
    df["ci_95_lo"] = ci_lo_vals
    df["ci_95_hi"] = ci_hi_vals
    df["p_value"] = p_vals

    return df


def bootstrap_stability(
    sequences: dict[str, list[str] | list[tuple[str, ...]]],
    customer_outcomes: dict[str, int],
    top_paths: list[tuple[str, ...]],
    n_iter: int = BOOTSTRAP_ITER,
    sample_ratio: float = BOOTSTRAP_SAMPLE_RATIO,
    min_sup: int = 10,
    is_ngram: bool = False,
) -> dict[tuple[str, ...], float]:
    """Bootstrap stability: fraction of iterations where path appears in top-K."""
    if not top_paths:
        return {}

    customer_ids = list(sequences.keys())
    n_sample = int(len(customer_ids) * sample_ratio)
    top_k = len(top_paths)
    top_set = set(top_paths)

    appearance_count: Counter = Counter()

    for i in range(n_iter):
        random.seed(42 + i)
        sampled = random.sample(customer_ids, n_sample)
        sampled_seqs = {cid: sequences[cid] for cid in sampled}
        sampled_outcomes = {cid: customer_outcomes[cid] for cid in sampled if cid in customer_outcomes}

        df_counts = count_paths(sampled_seqs, sampled_outcomes, min_sup, is_ngram=is_ngram)
        if df_counts.empty:
            continue
        df_metrics = compute_metrics(df_counts)
        df_metrics = df_metrics.sort_values("lift", ascending=False).head(top_k)

        for p in df_metrics["path_tuple"]:
            if p in top_set:
                appearance_count[p] += 1

    return {p: appearance_count.get(p, 0) / n_iter for p in top_paths}


def compute_transition_matrix(
    sequences: dict[str, list[str]],
    customer_outcomes: dict[str, int],
) -> pd.DataFrame:
    """Compute transition probabilities P(next|current) for each outcome group."""
    transitions: dict[int, Counter] = {0: Counter(), 1: Counter()}
    from_counts: dict[int, Counter] = {0: Counter(), 1: Counter()}

    for cid, seq in sequences.items():
        outcome = customer_outcomes.get(cid)
        if outcome is None:
            continue
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            transitions[outcome][pair] += 1
            from_counts[outcome][seq[i]] += 1

    # All unique codes
    all_codes = sorted(
        set(c for seq in sequences.values() for c in seq)
    )

    rows = []
    for src in all_codes:
        for dst in all_codes:
            pair = (src, dst)
            cnt_pos = transitions[1].get(pair, 0)
            cnt_neg = transitions[0].get(pair, 0)
            from_pos = from_counts[1].get(src, 0)
            from_neg = from_counts[0].get(src, 0)
            prob_pos = cnt_pos / from_pos if from_pos > 0 else 0.0
            prob_neg = cnt_neg / from_neg if from_neg > 0 else 0.0
            rows.append(
                {
                    "from": src,
                    "to": dst,
                    "prob_pos": round(prob_pos, 4),
                    "prob_neg": round(prob_neg, 4),
                    "prob_diff": round(prob_pos - prob_neg, 4),
                    "count_pos": cnt_pos,
                    "count_neg": cnt_neg,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 9. OUTPUT
# ============================================================


def save_outputs(
    df_customers: pd.DataFrame,
    results_daily: dict,
    results_weekly: dict,
    transition_daily: pd.DataFrame,
    transition_weekly: pd.DataFrame,
    sequences_daily: dict[str, list[str]],
    sequences_weekly: dict[str, list[str]],
    granularity_comparison: pd.DataFrame,
) -> None:
    """Save all outputs to the output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Customer journeys
    journey_rows = []
    for cid, seq in sequences_daily.items():
        row = df_customers[df_customers["unified_customer_id"] == cid]
        if row.empty:
            continue
        r = row.iloc[0]
        journey_rows.append(
            {
                "unified_customer_id": cid,
                "outcome": r["outcome"],
                "outcome_purchases": r["outcome_purchases"],
                "engagement_score": r.get("engagement_score"),
                "clv_12m": r.get("clv_12m"),
                "sequence_daily": " → ".join(seq),
                "sequence_weekly": " → ".join(sequences_weekly.get(cid, [])),
                "path_length_daily": len(seq),
                "path_length_weekly": len(sequences_weekly.get(cid, [])),
            }
        )
    pd.DataFrame(journey_rows).to_csv(
        OUTPUT_DIR / "customer_journeys.csv", index=False
    )

    # Path comparisons
    for gran, res in [("daily", results_daily), ("weekly", results_weekly)]:
        all_dfs = []
        for label, df in res.items():
            if df.empty:
                continue
            df_out = df.drop(columns=["path_tuple", "n_pos", "n_neg"], errors="ignore").copy()
            df_out["extraction_type"] = label
            all_dfs.append(df_out)
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(
                OUTPUT_DIR / f"path_comparison_{gran}.csv", index=False
            )

    # Transition matrices
    transition_daily.to_csv(OUTPUT_DIR / "transition_matrix_daily.csv", index=False)
    transition_weekly.to_csv(OUTPUT_DIR / "transition_matrix_weekly.csv", index=False)

    # Granularity comparison
    granularity_comparison.to_csv(
        OUTPUT_DIR / "granularity_comparison.csv", index=False
    )

    # JSON summary
    summary = build_json_summary(
        df_customers, results_daily, results_weekly, transition_daily
    )
    with open(OUTPUT_DIR / "golden_paths_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nOutputs saved to {OUTPUT_DIR}/")


def build_json_summary(
    df_customers: pd.DataFrame,
    results_daily: dict,
    results_weekly: dict,
    transition_daily: pd.DataFrame,
) -> dict:
    """Build JSON summary structure."""
    n_total = len(df_customers)
    n_pos = int(df_customers["outcome"].sum())
    n_neg = n_total - n_pos

    def top_paths(results: dict, k: int = 10) -> list[dict]:
        # Use first-3 as primary ranking
        key = "first_3"
        if key not in results or results[key].empty:
            # Fallback to any available
            for k2 in results:
                if not results[k2].empty:
                    key = k2
                    break
            else:
                return []
        df = results[key].sort_values("lift", ascending=False).head(k)
        out = []
        for _, r in df.iterrows():
            out.append(
                {
                    "path": r["path"],
                    "lift": round(r["lift"], 2) if r["lift"] != float("inf") else "inf",
                    "support_pos": round(r["support_pos"], 4),
                    "support_neg": round(r["support_neg"], 4),
                    "diff": round(r["diff"], 4),
                    "odds_ratio": round(r["odds_ratio"], 2) if r["odds_ratio"] != float("inf") else "inf",
                    "p_value": round(r["p_value"], 6) if not math.isnan(r["p_value"]) else None,
                    "stability": round(r.get("stability", 0), 2),
                }
            )
        return out

    # Top transition diffs
    top_transitions = (
        transition_daily.sort_values("prob_diff", ascending=False, key=abs)
        .head(10)
        .to_dict("records")
    )

    return {
        "config": {
            "observation_days": OBSERVATION_DAYS,
            "outcome_days": OUTCOME_DAYS,
            "suppress_codes": list(CFG["_suppress_codes"]),
            "eligibility_cutoff": str(ELIGIBILITY_CUTOFF),
            "data_end_date": str(DATA_END_DATE),
        },
        "population": {
            "eligible_customers": n_total,
            "outcome_positive": n_pos,
            "outcome_negative": n_neg,
            "outcome_rate": round(n_pos / n_total, 4) if n_total > 0 else 0,
        },
        "top_golden_paths_daily": top_paths(results_daily),
        "top_golden_paths_weekly": top_paths(results_weekly),
        "top_transition_diffs": top_transitions,
    }


# ============================================================
# 10. REPORTING (stdout)
# ============================================================


def print_report(
    df_customers: pd.DataFrame,
    results_daily: dict,
    results_weekly: dict,
    transition_daily: pd.DataFrame,
    granularity_comparison: pd.DataFrame,
) -> None:
    """Print summary report to stdout."""
    print("\n" + "=" * 70)
    print("  GOLDEN PATH ANALYSIS v2 — SUMMARY REPORT")
    print("=" * 70)

    n_total = len(df_customers)
    n_pos = int(df_customers["outcome"].sum())
    n_neg = n_total - n_pos
    print(f"\nPopulation: {n_total:,} eligible customers")
    print(f"  Outcome=1 (2+ repeat purchases): {n_pos:,} ({n_pos/n_total*100:.1f}%)")
    print(f"  Outcome=0: {n_neg:,} ({n_neg/n_total*100:.1f}%)")

    for gran, results in [("DAILY", results_daily), ("WEEKLY", results_weekly)]:
        print(f"\n--- Top Golden Paths ({gran}) ---")
        key = "first_3"
        if key not in results or results[key].empty:
            for k2 in results:
                if not results[k2].empty:
                    key = k2
                    break
            else:
                print("  No paths found.")
                continue

        df = results[key].sort_values("lift", ascending=False).head(10)
        print(f"  Extraction: {key}")
        print(
            f"  {'#':<3} {'Path':<45} {'Lift':>6} {'Sup+':>6} {'Sup-':>6} {'OR':>7} {'p':>8} {'Stab':>5}"
        )
        print(f"  {'-'*3} {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*5}")
        for i, (_, r) in enumerate(df.iterrows(), 1):
            lift_s = f"{r['lift']:.2f}" if r["lift"] != float("inf") else "inf"
            or_s = f"{r['odds_ratio']:.2f}" if r["odds_ratio"] != float("inf") else "inf"
            p_s = f"{r['p_value']:.4f}" if not math.isnan(r["p_value"]) else "n/a"
            stab_s = f"{r.get('stability', 0):.0%}" if "stability" in r else "n/a"
            print(
                f"  {i:<3} {r['path']:<45} {lift_s:>6} {r['support_pos']:>6.3f} "
                f"{r['support_neg']:>6.3f} {or_s:>7} {p_s:>8} {stab_s:>5}"
            )

    # Granularity comparison
    if not granularity_comparison.empty:
        print(f"\n--- Granularity Comparison (Daily vs Weekly) ---")
        overlap = granularity_comparison["in_both"].sum()
        total = len(granularity_comparison)
        print(f"  Top-{TOP_K_REPORT} overlap: {overlap}/{total} paths appear in both")

    # Top transition diffs
    print(f"\n--- Top Intervention Points (Transition Probability Diffs) ---")
    top_trans = transition_daily.sort_values("prob_diff", ascending=False, key=abs).head(10)
    print(f"  {'From':<12} {'To':<12} {'P(pos)':>7} {'P(neg)':>7} {'Diff':>7}")
    print(f"  {'-'*12} {'-'*12} {'-'*7} {'-'*7} {'-'*7}")
    for _, r in top_trans.iterrows():
        print(
            f"  {r['from']:<12} {r['to']:<12} {r['prob_pos']:>7.3f} "
            f"{r['prob_neg']:>7.3f} {r['prob_diff']:>+7.3f}"
        )

    print("\n" + "=" * 70)


# ============================================================
# 11. MAIN ORCHESTRATION
# ============================================================


def run_analysis_for_granularity(
    df_touchpoints: pd.DataFrame,
    df_customers: pd.DataFrame,
    customer_outcomes: dict[str, int],
    granularity: str,
    msup: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, list[str]]]:
    """Run full path analysis for a given granularity."""
    print(f"\n{'='*40}")
    print(f"  Granularity: {granularity.upper()}")
    print(f"{'='*40}")

    sequences = build_sequences(df_touchpoints, df_customers, granularity)

    # Filter short sequences
    valid_seqs = {
        cid: seq for cid, seq in sequences.items() if len(seq) >= MIN_PATH_LENGTH
    }
    excluded = len(sequences) - len(valid_seqs)
    print(f"  Sequences with {MIN_PATH_LENGTH}+ touchpoints: {len(valid_seqs):,} (excluded {excluded:,})")

    results: dict[str, pd.DataFrame] = {}

    # First-N extraction
    for n in FIRST_N_SIZES:
        first_n = extract_first_n(valid_seqs, n)
        if not first_n:
            results[f"first_{n}"] = pd.DataFrame()
            continue
        df_counts = count_paths(first_n, customer_outcomes, msup, is_ngram=False)
        if df_counts.empty:
            results[f"first_{n}"] = pd.DataFrame()
            continue
        df_metrics = compute_metrics(df_counts)
        # Bootstrap stability for top paths
        top = df_metrics.sort_values("lift", ascending=False).head(TOP_K_REPORT)
        top_path_tuples = top["path_tuple"].tolist()
        stability = bootstrap_stability(
            first_n, customer_outcomes, top_path_tuples, min_sup=msup, is_ngram=False
        )
        df_metrics["stability"] = df_metrics["path_tuple"].map(stability).fillna(0)
        results[f"first_{n}"] = df_metrics
        print(f"  first-{n}: {len(df_metrics)} paths above support threshold")

    # N-gram extraction
    for n in N_GRAM_SIZES:
        ngrams = extract_ngrams(valid_seqs, n)
        if not ngrams:
            results[f"ngram_{n}"] = pd.DataFrame()
            continue
        df_counts = count_paths(ngrams, customer_outcomes, msup, is_ngram=True)
        if df_counts.empty:
            results[f"ngram_{n}"] = pd.DataFrame()
            continue
        df_metrics = compute_metrics(df_counts)
        top = df_metrics.sort_values("lift", ascending=False).head(TOP_K_REPORT)
        top_path_tuples = top["path_tuple"].tolist()
        stability = bootstrap_stability(
            ngrams, customer_outcomes, top_path_tuples, min_sup=msup, is_ngram=True
        )
        df_metrics["stability"] = df_metrics["path_tuple"].map(stability).fillna(0)
        results[f"ngram_{n}"] = df_metrics
        print(f"  {n}-gram: {len(df_metrics)} paths above support threshold")

    # Transition matrix
    transition = compute_transition_matrix(valid_seqs, customer_outcomes)

    return results, transition, sequences


def compare_granularity(
    results_daily: dict[str, pd.DataFrame],
    results_weekly: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare top-K paths between daily and weekly granularity."""
    key = "first_3"
    rows = []

    for k in [key]:
        if k not in results_daily or k not in results_weekly:
            continue
        if results_daily[k].empty or results_weekly[k].empty:
            continue

        daily_top = set(
            results_daily[k].sort_values("lift", ascending=False).head(TOP_K_REPORT)["path"]
        )
        weekly_top = set(
            results_weekly[k].sort_values("lift", ascending=False).head(TOP_K_REPORT)["path"]
        )

        all_paths = daily_top | weekly_top
        for p in sorted(all_paths):
            rows.append(
                {
                    "extraction": k,
                    "path": p,
                    "in_daily_top": p in daily_top,
                    "in_weekly_top": p in weekly_top,
                    "in_both": p in daily_top and p in weekly_top,
                }
            )

    return pd.DataFrame(rows)


def run_mode(
    label: str,
    df_touchpoints: pd.DataFrame,
    df_customers: pd.DataFrame,
    customer_outcomes: dict[str, int],
    msup: int,
    output_subdir: Path,
) -> None:
    """Run full analysis (daily+weekly) for a given touchpoint set and save/report."""
    print(f"\n{'#'*70}")
    print(f"  MODE: {label}")
    print(f"{'#'*70}")

    tp_codes = sorted(df_touchpoints["code"].unique())
    print(f"  Touchpoint codes: {', '.join(tp_codes)}")
    print(f"  Events: {len(df_touchpoints):,}")

    results_daily, transition_daily, seqs_daily = run_analysis_for_granularity(
        df_touchpoints, df_customers, customer_outcomes, "daily", msup
    )
    results_weekly, transition_weekly, seqs_weekly = run_analysis_for_granularity(
        df_touchpoints, df_customers, customer_outcomes, "weekly", msup
    )

    granularity_comparison = compare_granularity(results_daily, results_weekly)

    # Save to subdir
    orig_output = OUTPUT_DIR
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Inline save (reuse save logic but to subdir)
    # Customer journeys
    journey_rows = []
    for cid, seq in seqs_daily.items():
        row = df_customers[df_customers["unified_customer_id"] == cid]
        if row.empty:
            continue
        r = row.iloc[0]
        journey_rows.append(
            {
                "unified_customer_id": cid,
                "outcome": r["outcome"],
                "outcome_purchases": r["outcome_purchases"],
                "engagement_score": r.get("engagement_score"),
                "clv_12m": r.get("clv_12m"),
                "sequence_daily": " → ".join(seq),
                "sequence_weekly": " → ".join(seqs_weekly.get(cid, [])),
                "path_length_daily": len(seq),
                "path_length_weekly": len(seqs_weekly.get(cid, [])),
            }
        )
    pd.DataFrame(journey_rows).to_csv(output_subdir / "customer_journeys.csv", index=False)

    for gran, res in [("daily", results_daily), ("weekly", results_weekly)]:
        all_dfs = []
        for extraction_label, df in res.items():
            if df.empty:
                continue
            df_out = df.drop(columns=["path_tuple", "n_pos", "n_neg"], errors="ignore").copy()
            df_out["extraction_type"] = extraction_label
            all_dfs.append(df_out)
        if all_dfs:
            pd.concat(all_dfs, ignore_index=True).to_csv(
                output_subdir / f"path_comparison_{gran}.csv", index=False
            )

    transition_daily.to_csv(output_subdir / "transition_matrix_daily.csv", index=False)
    transition_weekly.to_csv(output_subdir / "transition_matrix_weekly.csv", index=False)
    granularity_comparison.to_csv(output_subdir / "granularity_comparison.csv", index=False)

    summary = build_json_summary(df_customers, results_daily, results_weekly, transition_daily)
    summary["mode"] = label
    with open(output_subdir / "golden_paths_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n  Outputs saved to {output_subdir}/")

    # Print report
    print_report(df_customers, results_daily, results_weekly, transition_daily, granularity_comparison)


def main() -> None:
    # Fetch data
    client = get_supabase_client(CFG)
    dfs = fetch_data(CFG, client)
    ensure_first_date_col(CFG, dfs, client)
    if client:
        client.close()

    cust_cid_col = CFG["data_source"]["tables"]["customer"]["customer_id_col"]

    # Window assignment
    df_customers = assign_windows(dfs["customer"])

    # Outcome labeling
    df_customers = label_outcomes(df_customers, dfs["purchase"])

    # Touchpoint extraction (full)
    df_touchpoints = extract_touchpoints(dfs, df_customers)

    if df_touchpoints.empty:
        print("No touchpoints found. Exiting.")
        sys.exit(1)

    # Customer outcomes dict
    customer_outcomes = dict(
        zip(df_customers[cust_cid_col], df_customers["outcome"])
    )

    msup = min_support(len(df_customers))
    print(f"\nMin support threshold: {msup}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # === MODE 1: Full (all touchpoints including PURCHASE) ===
    if CFG["output"]["run_full_mode"]:
        run_mode(
            "FULL (all touchpoints)",
            df_touchpoints,
            df_customers,
            customer_outcomes,
            msup,
            OUTPUT_DIR / "full",
        )

    # === MODE 2: Non-purchase (PURCHASE excluded from paths) ===
    if CFG["output"]["run_no_purchase_mode"]:
        df_tp_no_purchase = df_touchpoints[df_touchpoints["code"] != "PURCHASE"].copy()
        print(f"\n  PURCHASE excluded: {len(df_touchpoints) - len(df_tp_no_purchase):,} events removed")

        run_mode(
            "NON-PURCHASE (nurturing paths only)",
            df_tp_no_purchase,
            df_customers,
            customer_outcomes,
            msup,
            OUTPUT_DIR / "no_purchase",
        )


if __name__ == "__main__":
    main()
