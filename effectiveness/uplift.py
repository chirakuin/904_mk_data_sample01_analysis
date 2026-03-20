"""
M4: Uplift Analysis (CATE by Engagement Quartile)
Identifies which customer segments benefit most from campaign treatment.
Output: output/uplift/uplift_by_segment.csv, ate_summary.json
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


# Covariates for propensity score
PSM_COVARIATES = ["engagement_score", "clv_12m", "monthly_spend", "lifetime_points_earned"]

# Features for uplift segmentation
UPLIFT_FEATURES = ["engagement_score", "clv_12m", "monthly_spend", "age_band", "gender"]


def compute_propensity_scores(
    df_treat: pd.DataFrame, df_control: pd.DataFrame, covariates: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    return df_treat, df_control


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

    cid = "unified_customer_id"

    # --- Pick the largest campaign by participation count ---
    campaign_sizes = df_cp.groupby("campaign_id")[cid].nunique().sort_values(ascending=False)
    target_campaign = campaign_sizes.index[0]
    target_size = campaign_sizes.iloc[0]
    print(f"\nLargest campaign: {target_campaign} ({target_size} participants)")

    df_camp = df_cp[df_cp["campaign_id"] == target_campaign]
    participants = set(df_camp[cid].unique())

    # Campaign period
    campaign_end = df_camp["entry_datetime"].max()
    post_start = campaign_end
    post_end = post_start + pd.Timedelta(days=30)

    # Treatment / Control
    all_customers = set(df_cust[cid].unique())
    control_ids = all_customers - participants

    df_treat = df_cust[df_cust[cid].isin(participants)].copy()
    df_control = df_cust[df_cust[cid].isin(control_ids)].copy()

    print(f"  Treatment: {len(df_treat)}, Control pool: {len(df_control)}")

    # --- Propensity Score Matching ---
    available_covs = [c for c in PSM_COVARIATES if c in df_cust.columns]
    df_treat, df_control = compute_propensity_scores(df_treat, df_control, available_covs)
    df_treat_m, df_control_m = nearest_neighbor_match(df_treat, df_control)
    print(f"  Matched pairs: {len(df_treat_m)}")

    if df_treat_m.empty or df_control_m.empty:
        print("ERROR: Matching produced no pairs.")
        sys.exit(1)

    # --- Compute binary outcome: purchased within 30 days ---
    df_post_tx = df_tx[
        (df_tx["purchase_datetime"] >= post_start) & (df_tx["purchase_datetime"] <= post_end)
    ]
    purchasers_post = set(df_post_tx[cid].unique())

    df_treat_m["purchased"] = df_treat_m[cid].isin(purchasers_post).astype(int)
    df_control_m["purchased"] = df_control_m[cid].isin(purchasers_post).astype(int)

    # --- Overall ATE ---
    ate = df_treat_m["purchased"].mean() - df_control_m["purchased"].mean()
    t_stat, p_val = stats.ttest_ind(df_treat_m["purchased"], df_control_m["purchased"], equal_var=False)

    print(f"\n  Overall ATE: {ate:.4f} (p={p_val:.4f})")

    # --- CATE by engagement_score quartile ---
    if "engagement_score" not in df_treat_m.columns:
        print("ERROR: engagement_score not available for subgroup analysis.")
        sys.exit(1)

    # Assign quartiles based on combined distribution
    all_scores = pd.concat([df_treat_m["engagement_score"], df_control_m["engagement_score"]])
    quartile_bins = all_scores.quantile([0, 0.25, 0.5, 0.75, 1.0]).values.copy()
    quartile_bins[0] = -np.inf
    quartile_bins[-1] = np.inf
    labels = ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]

    df_treat_m["eng_quartile"] = pd.cut(
        df_treat_m["engagement_score"], bins=quartile_bins, labels=labels, include_lowest=True
    )
    df_control_m["eng_quartile"] = pd.cut(
        df_control_m["engagement_score"], bins=quartile_bins, labels=labels, include_lowest=True
    )

    uplift_rows: list[dict] = []
    for q in labels:
        t_q = df_treat_m[df_treat_m["eng_quartile"] == q]
        c_q = df_control_m[df_control_m["eng_quartile"] == q]

        if len(t_q) < 5 or len(c_q) < 5:
            continue

        cate = t_q["purchased"].mean() - c_q["purchased"].mean()
        t_s, p_v = stats.ttest_ind(t_q["purchased"], c_q["purchased"], equal_var=False)

        uplift_rows.append({
            "engagement_quartile": q,
            "n_treatment": len(t_q),
            "n_control": len(c_q),
            "treatment_purchase_rate": round(t_q["purchased"].mean(), 4),
            "control_purchase_rate": round(c_q["purchased"].mean(), 4),
            "cate": round(cate, 4),
            "p_value": round(p_v, 6),
            "significant": p_v < 0.05,
        })

    # --- Output ---
    output_dir = Path(__file__).parent.parent / "output" / "uplift"
    output_dir.mkdir(parents=True, exist_ok=True)

    uplift_csv_path = output_dir / "uplift_by_segment.csv"
    pd.DataFrame(uplift_rows).to_csv(uplift_csv_path, index=False)
    print(f"\nSaved: {uplift_csv_path}")

    ate_summary = {
        "campaign_id": target_campaign,
        "n_treatment_matched": int(len(df_treat_m)),
        "n_control_matched": int(len(df_control_m)),
        "overall_ate": round(ate, 4),
        "overall_p_value": round(p_val, 6),
        "overall_significant": bool(p_val < 0.05),
        "treatment_purchase_rate": round(df_treat_m["purchased"].mean(), 4),
        "control_purchase_rate": round(df_control_m["purchased"].mean(), 4),
        "uplift_by_quartile": uplift_rows,
    }

    ate_json_path = output_dir / "ate_summary.json"
    with open(ate_json_path, "w", encoding="utf-8") as f:
        json.dump(ate_summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {ate_json_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Uplift Analysis Summary")
    print("=" * 60)
    print(f"  Campaign: {target_campaign}")
    print(f"  Overall ATE: {ate:.4f} (p={p_val:.4f}, {'significant' if p_val < 0.05 else 'not significant'})")
    print(f"\n  Uplift by Engagement Quartile:")
    for row in uplift_rows:
        sig = "*" if row["significant"] else ""
        print(
            f"    {row['engagement_quartile']}: "
            f"CATE={row['cate']:.4f} "
            f"(treat={row['treatment_purchase_rate']:.2%}, ctrl={row['control_purchase_rate']:.2%}) "
            f"p={row['p_value']:.4f}{sig}"
        )

    if not uplift_rows:
        print("  No quartile segments had sufficient data.")


if __name__ == "__main__":
    main()
