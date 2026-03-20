"""
M3: Campaign Effectiveness Analysis
Propensity Score Matching to compare campaign participants vs non-participants.
Output: output/campaign/campaign_effects.csv, matching_quality.csv, summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.data_loader import load_config, fetch_all, get_supabase_client  # noqa: E402


# Covariates for propensity score model
PSM_COVARIATES = ["engagement_score", "clv_12m", "monthly_spend", "lifetime_points_earned"]


def compute_propensity_scores(
    df_treat: pd.DataFrame, df_control: pd.DataFrame, covariates: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, LogisticRegression]:
    """Fit logistic regression and return dataframes with propensity scores."""
    df_t = df_treat[covariates].copy().fillna(0)
    df_c = df_control[covariates].copy().fillna(0)

    X = pd.concat([df_t, df_c], ignore_index=True)
    y = np.array([1] * len(df_t) + [0] * len(df_c))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    probs = model.predict_proba(X_scaled)[:, 1]

    df_treat = df_treat.copy()
    df_control = df_control.copy()
    df_treat["propensity_score"] = probs[: len(df_t)]
    df_control["propensity_score"] = probs[len(df_t):]

    return df_treat, df_control, model


def nearest_neighbor_match(
    df_treat: pd.DataFrame, df_control: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1:1 nearest neighbor matching on propensity score."""
    if df_control.empty or df_treat.empty:
        return df_treat, df_control

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(df_control[["propensity_score"]])
    distances, indices = nn.kneighbors(df_treat[["propensity_score"]])

    matched_control_idx = indices.flatten()
    # Remove duplicate matches (keep closest)
    unique_matches: dict[int, tuple[int, float]] = {}
    for t_idx, (c_idx, dist) in enumerate(zip(matched_control_idx, distances.flatten())):
        if c_idx not in unique_matches or dist < unique_matches[c_idx][1]:
            unique_matches[c_idx] = (t_idx, dist)

    matched_control_positions = list(unique_matches.keys())
    matched_treat_positions = [unique_matches[c][0] for c in matched_control_positions]

    df_treat_matched = df_treat.iloc[matched_treat_positions].copy()
    df_control_matched = df_control.iloc[matched_control_positions].copy()

    return df_treat_matched, df_control_matched


def main() -> None:
    cfg = load_config()
    client = get_supabase_client(cfg)
    if client is None:
        print("ERROR: Supabase client not configured.")
        sys.exit(1)

    # --- Fetch data ---
    print("Fetching data...")
    cp_rows = fetch_all(client, "campaign_participation", "unified_customer_id,campaign_id,entry_datetime")
    print(f"  campaign_participation: {len(cp_rows):,} rows")

    tx_rows = fetch_all(client, "purchase_transaction", "unified_customer_id,purchase_datetime,total_amount")
    print(f"  purchase_transaction: {len(tx_rows):,} rows")

    cust_rows = fetch_all(client, "v_customer_summary", "*")
    print(f"  v_customer_summary: {len(cust_rows):,} rows")

    df_cp = pd.DataFrame(cp_rows)
    df_tx = pd.DataFrame(tx_rows)
    df_cust = pd.DataFrame(cust_rows)

    if df_cp.empty or df_tx.empty or df_cust.empty:
        print("ERROR: One or more tables returned no data.")
        sys.exit(1)

    # Parse dates
    df_cp["entry_datetime"] = pd.to_datetime(df_cp["entry_datetime"])
    df_tx["purchase_datetime"] = pd.to_datetime(df_tx["purchase_datetime"])

    # Ensure numeric covariates
    for col in PSM_COVARIATES:
        if col in df_cust.columns:
            df_cust[col] = pd.to_numeric(df_cust[col], errors="coerce").fillna(0)

    # Customer ID col
    cid = "unified_customer_id"

    # Get unique campaigns
    campaigns = df_cp["campaign_id"].unique()
    print(f"\nCampaigns found: {len(campaigns)}")

    # --- Per-campaign analysis ---
    effect_rows: list[dict] = []
    matching_quality_rows: list[dict] = []
    summary_results: list[dict] = []

    for campaign_id in campaigns:
        df_camp = df_cp[df_cp["campaign_id"] == campaign_id]
        participants = set(df_camp[cid].unique())

        # Campaign end date (last entry + a small buffer)
        campaign_end = df_camp["entry_datetime"].max()
        post_start = campaign_end
        post_end = post_start + pd.Timedelta(days=30)

        # Treatment group
        treat_ids = participants
        # Control group: all non-participants
        all_customers = set(df_cust[cid].unique())
        control_ids = all_customers - treat_ids

        if len(treat_ids) < 5 or len(control_ids) < 5:
            continue

        # Build treatment and control dataframes with covariates
        df_treat = df_cust[df_cust[cid].isin(treat_ids)].copy()
        df_control = df_cust[df_cust[cid].isin(control_ids)].copy()

        # Check that covariates exist
        available_covs = [c for c in PSM_COVARIATES if c in df_cust.columns]
        if len(available_covs) < 2:
            print(f"  Campaign {campaign_id}: insufficient covariates, skipping")
            continue

        # Propensity score matching
        try:
            df_treat, df_control, _ = compute_propensity_scores(df_treat, df_control, available_covs)
            df_treat_m, df_control_m = nearest_neighbor_match(df_treat, df_control)
        except Exception as e:
            print(f"  Campaign {campaign_id}: matching failed ({e}), skipping")
            continue

        if df_treat_m.empty or df_control_m.empty:
            continue

        # Post-campaign purchase behavior
        df_post_tx = df_tx[
            (df_tx["purchase_datetime"] >= post_start) & (df_tx["purchase_datetime"] <= post_end)
        ]

        # Treatment outcomes
        treat_tx = df_post_tx[df_post_tx[cid].isin(set(df_treat_m[cid]))]
        treat_agg = treat_tx.groupby(cid).agg(
            purchase_count=("total_amount", "count"),
            purchase_amount=("total_amount", "sum"),
        ).reindex(df_treat_m[cid].values).fillna(0)

        # Control outcomes
        ctrl_tx = df_post_tx[df_post_tx[cid].isin(set(df_control_m[cid]))]
        ctrl_agg = ctrl_tx.groupby(cid).agg(
            purchase_count=("total_amount", "count"),
            purchase_amount=("total_amount", "sum"),
        ).reindex(df_control_m[cid].values).fillna(0)

        # Compute differences
        mean_treat_count = treat_agg["purchase_count"].mean()
        mean_ctrl_count = ctrl_agg["purchase_count"].mean()
        mean_treat_amount = treat_agg["purchase_amount"].mean()
        mean_ctrl_amount = ctrl_agg["purchase_amount"].mean()

        count_diff = mean_treat_count - mean_ctrl_count
        amount_diff = mean_treat_amount - mean_ctrl_amount
        lift_count = count_diff / mean_ctrl_count if mean_ctrl_count > 0 else np.nan
        lift_amount = amount_diff / mean_ctrl_amount if mean_ctrl_amount > 0 else np.nan

        # T-test for significance
        t_stat_count, p_val_count = stats.ttest_ind(
            treat_agg["purchase_count"], ctrl_agg["purchase_count"], equal_var=False
        )
        t_stat_amount, p_val_amount = stats.ttest_ind(
            treat_agg["purchase_amount"], ctrl_agg["purchase_amount"], equal_var=False
        )

        effect_rows.append({
            "campaign_id": campaign_id,
            "n_treatment": len(df_treat_m),
            "n_control": len(df_control_m),
            "mean_purchase_count_treat": round(mean_treat_count, 4),
            "mean_purchase_count_ctrl": round(mean_ctrl_count, 4),
            "purchase_count_diff": round(count_diff, 4),
            "purchase_count_lift": round(lift_count, 4) if not np.isnan(lift_count) else None,
            "purchase_count_pvalue": round(p_val_count, 6),
            "mean_purchase_amount_treat": round(mean_treat_amount, 2),
            "mean_purchase_amount_ctrl": round(mean_ctrl_amount, 2),
            "purchase_amount_diff": round(amount_diff, 2),
            "purchase_amount_lift": round(lift_amount, 4) if not np.isnan(lift_amount) else None,
            "purchase_amount_pvalue": round(p_val_amount, 6),
        })

        # Matching quality: covariate balance
        for cov in available_covs:
            t_vals = df_treat_m[cov].astype(float)
            c_vals = df_control_m[cov].astype(float)
            smd = (t_vals.mean() - c_vals.mean()) / np.sqrt((t_vals.std() ** 2 + c_vals.std() ** 2) / 2) if (t_vals.std() + c_vals.std()) > 0 else 0.0
            matching_quality_rows.append({
                "campaign_id": campaign_id,
                "covariate": cov,
                "mean_treatment": round(t_vals.mean(), 4),
                "mean_control": round(c_vals.mean(), 4),
                "std_mean_diff": round(abs(smd), 4),
            })

        summary_results.append({
            "campaign_id": campaign_id,
            "n_matched_pairs": len(df_treat_m),
            "purchase_count_lift": round(lift_count, 4) if not np.isnan(lift_count) else None,
            "purchase_amount_lift": round(lift_amount, 4) if not np.isnan(lift_amount) else None,
            "count_significant": p_val_count < 0.05,
            "amount_significant": p_val_amount < 0.05,
        })

    # --- Output ---
    output_dir = Path(__file__).parent.parent / "output" / "campaign"
    output_dir.mkdir(parents=True, exist_ok=True)

    effects_path = output_dir / "campaign_effects.csv"
    pd.DataFrame(effect_rows).to_csv(effects_path, index=False)
    print(f"\nSaved: {effects_path}")

    quality_path = output_dir / "matching_quality.csv"
    pd.DataFrame(matching_quality_rows).to_csv(quality_path, index=False)
    print(f"Saved: {quality_path}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {summary_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Campaign Effectiveness Summary")
    print("=" * 60)
    for s in summary_results:
        sig_marker = lambda p: "*" if p else ""
        print(
            f"  Campaign {s['campaign_id']}: "
            f"n={s['n_matched_pairs']} pairs, "
            f"count lift={s['purchase_count_lift']}, "
            f"amount lift={s['purchase_amount_lift']}, "
            f"count sig={s['count_significant']}, amount sig={s['amount_significant']}"
        )

    if not summary_results:
        print("  No campaigns had sufficient data for analysis.")


if __name__ == "__main__":
    main()
