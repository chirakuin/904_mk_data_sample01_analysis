"""
Customer Clustering Analysis (C1 + C2)

v_customer_summary からクラスタリングを実行し、ペルソナを生成する。
KMeans (k=4,5,6 からシルエットスコアで選択) + DBSCAN 比較。

Usage:
    uv run python customer/clustering.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from lib.data_loader import fetch_all, get_supabase_client, load_config

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "clustering"

FEATURE_COLS = [
    "engagement_score",
    "churn_risk_score",
    "clv_12m",
    "monthly_spend",
    "health_consciousness",
    "lifetime_points_earned",
]

PROFILE_COLS = [
    "unified_customer_id",
    "engagement_score",
    "churn_risk_score",
    "clv_12m",
    "monthly_spend",
    "health_consciousness",
    "lifetime_points_earned",
    "prefecture",
    "membership_tier",
    "gender",
]


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


# ============================================================
# Preprocessing
# ============================================================


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Fill missing values and scale features."""
    df_work = df.copy()
    for col in FEATURE_COLS:
        if col not in df_work.columns:
            print(f"  WARNING: {col} not found, filling with 0")
            df_work[col] = 0.0
        else:
            median_val = df_work[col].median()
            n_missing = df_work[col].isna().sum()
            if n_missing > 0:
                print(f"  {col}: {n_missing} missing values filled with median ({median_val:.2f})")
            df_work[col] = df_work[col].fillna(median_val)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_work[FEATURE_COLS].values)
    return df_work, X, scaler


# ============================================================
# Clustering
# ============================================================


def run_kmeans(X: np.ndarray) -> tuple[int, np.ndarray, dict]:
    """Run KMeans for k=4,5,6 and select best by silhouette score."""
    results = {}
    for k in [4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        results[k] = {"labels": labels, "silhouette": score, "inertia": km.inertia_}
        print(f"  KMeans k={k}: silhouette={score:.4f}, inertia={km.inertia_:.1f}")

    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"  -> Best k={best_k} (silhouette={results[best_k]['silhouette']:.4f})")

    silhouette_report = {
        str(k): {
            "silhouette_score": round(v["silhouette"], 4),
            "inertia": round(v["inertia"], 2),
        }
        for k, v in results.items()
    }
    silhouette_report["best_k"] = best_k

    return best_k, results[best_k]["labels"], silhouette_report


def run_dbscan(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run DBSCAN as comparison."""
    db = DBSCAN(eps=1.5, min_samples=10)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    if n_clusters >= 2:
        mask = labels != -1
        score = silhouette_score(X[mask], labels[mask]) if mask.sum() > 1 else -1.0
    else:
        score = -1.0

    info = {
        "n_clusters": n_clusters,
        "n_noise": int(n_noise),
        "silhouette_score": round(score, 4),
    }
    print(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise points, silhouette={score:.4f}")
    return labels, info


# ============================================================
# Persona Generation
# ============================================================


def generate_profiles(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    """Generate cluster profiles / personas."""
    df_c = df.copy()
    df_c["cluster"] = labels
    profiles = []

    for cid in sorted(df_c["cluster"].unique()):
        group = df_c[df_c["cluster"] == cid]
        profile: dict = {
            "cluster": int(cid),
            "size": len(group),
            "pct": round(len(group) / len(df_c) * 100, 1),
        }

        # Numeric feature stats
        for col in FEATURE_COLS:
            if col in group.columns:
                profile[f"{col}_mean"] = round(float(group[col].mean()), 2)
                profile[f"{col}_median"] = round(float(group[col].median()), 2)

        # Categorical top values
        if "prefecture" in group.columns:
            top_pref = group["prefecture"].mode()
            profile["top_prefecture"] = str(top_pref.iloc[0]) if len(top_pref) > 0 else "N/A"

        if "membership_tier" in group.columns:
            top_tier = group["membership_tier"].mode()
            profile["top_membership_tier"] = str(top_tier.iloc[0]) if len(top_tier) > 0 else "N/A"

        if "gender" in group.columns:
            top_gender = group["gender"].mode()
            profile["top_gender"] = str(top_gender.iloc[0]) if len(top_gender) > 0 else "N/A"

        profiles.append(profile)

    return profiles


# ============================================================
# Output
# ============================================================


def save_outputs(
    df: pd.DataFrame,
    kmeans_labels: np.ndarray,
    profiles: list[dict],
    silhouette_report: dict,
    dbscan_info: dict,
) -> None:
    """Save all outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cluster assignments
    df_out = df[["unified_customer_id"]].copy()
    df_out["cluster"] = kmeans_labels
    df_out.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'cluster_assignments.csv'}")

    # Cluster profiles
    output_profiles = {
        "kmeans_profiles": profiles,
        "dbscan_comparison": dbscan_info,
    }
    with open(OUTPUT_DIR / "cluster_profiles.json", "w", encoding="utf-8") as f:
        json.dump(output_profiles, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {OUTPUT_DIR / 'cluster_profiles.json'}")

    # Silhouette report
    with open(OUTPUT_DIR / "silhouette_report.json", "w", encoding="utf-8") as f:
        json.dump(silhouette_report, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'silhouette_report.json'}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  Customer Clustering Analysis (C1 + C2)")
    print("=" * 60)

    cfg = load_config()
    df = fetch_customer_summary(cfg)

    if df.empty:
        print("ERROR: No data fetched. Exiting.")
        sys.exit(1)

    print("\nPreparing features...")
    df, X, scaler = prepare_features(df)

    print("\nRunning KMeans...")
    best_k, kmeans_labels, silhouette_report = run_kmeans(X)

    print("\nRunning DBSCAN (comparison)...")
    dbscan_labels, dbscan_info = run_dbscan(X)

    print("\nGenerating cluster profiles...")
    profiles = generate_profiles(df, kmeans_labels)

    save_outputs(df, kmeans_labels, profiles, silhouette_report, dbscan_info)

    # Print summary
    print("\n" + "=" * 60)
    print("  CLUSTERING SUMMARY")
    print("=" * 60)
    print(f"\nBest KMeans: k={best_k}, silhouette={silhouette_report[str(best_k)]['silhouette_score']:.4f}")
    print(f"\nCluster sizes:")
    for p in profiles:
        print(f"  Cluster {p['cluster']}: {p['size']:,} customers ({p['pct']}%)")
    print(f"\nCluster profiles:")
    for p in profiles:
        print(f"\n  Cluster {p['cluster']}:")
        for col in FEATURE_COLS:
            mean_key = f"{col}_mean"
            if mean_key in p:
                print(f"    {col}: mean={p[mean_key]}, median={p[f'{col}_median']}")
        for cat_key in ["top_prefecture", "top_membership_tier", "top_gender"]:
            if cat_key in p:
                print(f"    {cat_key}: {p[cat_key]}")

    print(f"\nDBSCAN comparison: {dbscan_info['n_clusters']} clusters, "
          f"{dbscan_info['n_noise']} noise, silhouette={dbscan_info['silhouette_score']:.4f}")


if __name__ == "__main__":
    main()
