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

# クラスタリング特徴量: 嗜好・属性軸のみ
# 除外するもの:
#   金額系（clv_12m, monthly_spend, lifetime_points_earned）→ 結果指標
#   エンゲージメント系（engagement_score, churn_risk_score）→ 当然の結果で分離するだけ
# 採用するもの:
#   「何が好きか」「どんな人か」→ 施策の方向性が決まる軸
FEATURE_COLS_NUMERIC = [
    "health_consciousness",
]
FEATURE_COLS_BINARY = [
    "is_alcohol_eligible",  # 酒類適格（行動を大きく分ける軸）
]
FEATURE_COLS_CATEGORICAL = [
    "gender",               # 性別（商品選好に直結）
    "preferred_category",   # 嗜好カテゴリ（最頻購買カテゴリ）
    "registration_source",  # 登録チャネル（チャネル親和性の代理変数）
]
# 内部的に結合して使う
FEATURE_COLS = FEATURE_COLS_NUMERIC + FEATURE_COLS_BINARY + FEATURE_COLS_CATEGORICAL

# プロファイル用（クラスタの特徴記述に使う。金額はここで確認）
PROFILE_COLS = [
    "unified_customer_id",
    "engagement_score",
    "churn_risk_score",
    "health_consciousness",
    "is_alcohol_eligible",
    "clv_12m",
    "monthly_spend",
    "lifetime_points_earned",
    "prefecture",
    "membership_tier",
    "gender",
    "age_band",
    "registration_source",
    "preferred_category",
    "rfm_segment",
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

    # --- 数値特徴量 ---
    for col in FEATURE_COLS_NUMERIC:
        if col not in df_work.columns:
            print(f"  WARNING: {col} not found, filling with 0")
            df_work[col] = 0.0
        else:
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")
            median_val = df_work[col].median()
            n_missing = df_work[col].isna().sum()
            if n_missing > 0:
                print(f"  {col}: {n_missing} missing → median ({median_val:.2f})")
            df_work[col] = df_work[col].fillna(median_val)

    # --- バイナリ特徴量 ---
    for col in FEATURE_COLS_BINARY:
        if col in df_work.columns:
            df_work[col] = df_work[col].map(
                {"true": 1, "True": 1, True: 1, "false": 0, "False": 0, False: 0}
            ).fillna(0).astype(float)
        else:
            df_work[col] = 0.0

    # --- カテゴリ特徴量 → one-hot ---
    for col in FEATURE_COLS_CATEGORICAL:
        if col not in df_work.columns:
            print(f"  WARNING: {col} not found, skipping")
            continue
        df_work[col] = df_work[col].fillna("unknown")

    dummies = pd.get_dummies(df_work[FEATURE_COLS_CATEGORICAL], prefix=FEATURE_COLS_CATEGORICAL, drop_first=False)
    print(f"  One-hot encoded: {FEATURE_COLS_CATEGORICAL} → {len(dummies.columns)} columns")

    # 結合
    feature_matrix = pd.concat([
        df_work[FEATURE_COLS_NUMERIC + FEATURE_COLS_BINARY].reset_index(drop=True),
        dummies.reset_index(drop=True),
    ], axis=1)

    all_feature_names = list(feature_matrix.columns)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix.values)
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

        # Numeric feature stats (clustering features + result metrics)
        numeric_cols = FEATURE_COLS_NUMERIC + FEATURE_COLS_BINARY + [
            "engagement_score", "churn_risk_score",  # 結果指標として表示
            "clv_12m", "monthly_spend", "lifetime_points_earned",
        ]
        for col in numeric_cols:
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors="coerce")
                profile[f"{col}_mean"] = round(float(vals.mean()), 2)
                profile[f"{col}_median"] = round(float(vals.median()), 2)

        # Categorical top values
        for col in FEATURE_COLS_CATEGORICAL + ["prefecture", "membership_tier", "age_band", "rfm_segment"]:
            if col in group.columns:
                top_val = group[col].mode()
                profile[f"top_{col}"] = str(top_val.iloc[0]) if len(top_val) > 0 else "N/A"

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
        # Numeric stats
        for col in FEATURE_COLS_NUMERIC + FEATURE_COLS_BINARY + [
            "engagement_score", "churn_risk_score", "clv_12m", "monthly_spend",
        ]:
            mean_key = f"{col}_mean"
            if mean_key in p:
                print(f"    {col}: mean={p[mean_key]}, median={p.get(f'{col}_median', 'N/A')}")
        # Categorical top values
        for key, val in p.items():
            if key.startswith("top_"):
                print(f"    {key}: {val}")

    print(f"\nDBSCAN comparison: {dbscan_info['n_clusters']} clusters, "
          f"{dbscan_info['n_noise']} noise, silhouette={dbscan_info['silhouette_score']:.4f}")


if __name__ == "__main__":
    main()
