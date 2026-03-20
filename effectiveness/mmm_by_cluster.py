"""
M1b: クラスタ別MMM + ROI分析

クラスタリング結果（output/clustering/cluster_assignments.csv）と購買データを結合し、
クラスタ別の週次売上を構築してMMMを回す。クラスタごとのメディア弾性値・ROIを比較。

依存: customer/clustering.py の実行済み出力
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.data_loader import fetch_all, get_supabase_client, load_config

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "roi_by_cluster"
CLUSTER_FILE = PROJECT_ROOT / "output" / "clustering" / "cluster_assignments.csv"

# --- Adstock ---

def adstock(series: pd.Series, decay: float) -> pd.Series:
    result = np.zeros(len(series))
    result[0] = series.iloc[0]
    for i in range(1, len(series)):
        result[i] = series.iloc[i] + decay * result[i - 1]
    return pd.Series(result, index=series.index)


MEDIA_VARS = {
    "tv_grp": 0.7,
    "digital_spend_jpy": 0.4,
    "ooh_spend_jpy": 0.5,
    "trade_promo_spend_jpy": 0.5,
    "line_messages_delivered": 0.5,
    "campaign_entries": 0.5,
}
CONTROL_VARS = ["avg_temperature", "seasonal_index", "competitor_tv_grp"]


# --- Build cluster-level weekly sales ---

def build_cluster_weekly_sales(
    df_purchases: pd.DataFrame,
    df_clusters: pd.DataFrame,
    df_products: pd.DataFrame,
) -> pd.DataFrame:
    """購買データにクラスタラベルを結合し、クラスタ×ブランド×週の売上を集計。"""

    # クラスタラベル結合
    df = df_purchases.merge(
        df_clusters[["unified_customer_id", "cluster"]],
        on="unified_customer_id",
        how="inner",
    )

    # ブランドコード結合
    df = df.merge(
        df_products[["product_id", "brand_code"]],
        on="product_id",
        how="left",
    )

    # 週開始日（月曜基準）
    df["purchase_dt"] = pd.to_datetime(df["purchase_datetime"])
    df["week_start_date"] = df["purchase_dt"].apply(
        lambda d: (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")
    )

    # クラスタ×ブランド×週 で集計
    grouped = (
        df.groupby(["cluster", "brand_code", "week_start_date"])
        .agg(
            sales_volume=("quantity", "sum"),
            sales_amount=("total_amount", "sum"),
        )
        .reset_index()
    )

    return grouped


# --- Run MMM per cluster×brand ---

def run_mmm_for_segment(
    df_segment: pd.DataFrame,
    df_market: pd.DataFrame,
    segment_label: str,
    brand: str,
) -> dict | None:
    """クラスタの週次売上 + 全体のメディア投下データでMMMを実行。"""

    # メディア投下量はブランド単位（クラスタで分けられない）
    df_media = df_market[df_market["brand_code"] == brand][
        ["week_start_date"] + list(MEDIA_VARS.keys()) + CONTROL_VARS
    ].drop_duplicates("week_start_date")

    # 結合
    df = df_segment.merge(df_media, on="week_start_date", how="inner")
    df = df.sort_values("week_start_date").reset_index(drop=True)

    if len(df) < 20:
        return None

    y = df["sales_volume"].astype(float)
    if y.sum() == 0:
        return None

    # Build X
    X_parts: dict[str, pd.Series] = {}
    for var, decay_rate in MEDIA_VARS.items():
        if var in df.columns:
            X_parts[f"{var}_adstock"] = adstock(df[var].astype(float).fillna(0), decay_rate)
    for var in CONTROL_VARS:
        if var in df.columns:
            X_parts[var] = df[var].astype(float).fillna(0)

    X = pd.DataFrame(X_parts)
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    mean_y = y.mean()
    elasticities = {}
    for col in X.columns:
        if col == "const":
            continue
        elasticities[col] = float(model.params[col] * X[col].mean() / mean_y) if mean_y != 0 else 0.0

    return {
        "segment": segment_label,
        "brand_code": brand,
        "n_weeks": len(df),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "mean_weekly_sales": round(float(mean_y), 1),
        "elasticities": elasticities,
    }


# --- Main ---

def main() -> None:
    print("=" * 60)
    print("  Cluster-level MMM + ROI Analysis")
    print("=" * 60)

    # Check cluster file
    if not CLUSTER_FILE.exists():
        print(f"ERROR: {CLUSTER_FILE} not found.")
        print("Please run customer/clustering.py first.")
        sys.exit(1)

    df_clusters = pd.read_csv(CLUSTER_FILE)
    n_clusters = df_clusters["cluster"].nunique()
    print(f"Loaded cluster assignments: {len(df_clusters):,} customers, {n_clusters} clusters")

    # Fetch data
    cfg = load_config()
    client = get_supabase_client(cfg)

    print("Fetching data...")
    purchases = pd.DataFrame(fetch_all(client, "purchase_transaction",
        "unified_customer_id,product_id,purchase_datetime,quantity,total_amount"))
    print(f"  purchase_transaction: {len(purchases):,} rows")

    products = pd.DataFrame(fetch_all(client, "product_master", "product_id,brand_code"))
    print(f"  product_master: {len(products):,} rows")

    market = pd.DataFrame(fetch_all(client, "mmm_weekly_market", "*"))
    print(f"  mmm_weekly_market: {len(market):,} rows")

    client.close()

    # Build cluster-level weekly sales
    print("\nBuilding cluster-level weekly sales...")
    df_cluster_sales = build_cluster_weekly_sales(purchases, df_clusters, products)
    print(f"  Cluster × brand × week: {len(df_cluster_sales):,} rows")

    # Identify top brands
    top_brands = (
        df_cluster_sales.groupby("brand_code")["sales_volume"]
        .sum().sort_values(ascending=False).head(5).index.tolist()
    )
    print(f"  Top 5 brands: {top_brands}")

    # Run MMM per cluster × brand
    print("\nRunning MMM per cluster × brand...")
    all_results: list[dict] = []
    all_roi_rows: list[dict] = []

    cluster_labels = sorted(df_clusters["cluster"].unique())

    for cluster_id in cluster_labels:
        cluster_name = f"Cluster_{cluster_id}"
        df_c = df_cluster_sales[df_cluster_sales["cluster"] == cluster_id]

        for brand in top_brands:
            df_cb = df_c[df_c["brand_code"] == brand][["week_start_date", "sales_volume"]].copy()

            result = run_mmm_for_segment(df_cb, market, cluster_name, brand)
            if result is None:
                continue

            all_results.append(result)

            # Compute ROI per channel
            mean_sales = result["mean_weekly_sales"]
            brand_market = market[market["brand_code"] == brand]

            for var_adstock, elast in result["elasticities"].items():
                raw_var = var_adstock.replace("_adstock", "")
                if raw_var in brand_market.columns:
                    mean_spend = brand_market[raw_var].astype(float).mean()
                    if mean_spend > 0:
                        marginal_roi = elast * mean_sales / mean_spend
                    else:
                        marginal_roi = 0.0
                else:
                    marginal_roi = 0.0

                all_roi_rows.append({
                    "cluster": cluster_name,
                    "brand_code": brand,
                    "channel": raw_var,
                    "elasticity": round(elast, 4),
                    "marginal_roi": round(marginal_roi, 4),
                    "r_squared": result["r_squared"],
                })

    # --- Output ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Results JSON
    with open(OUTPUT_DIR / "cluster_mmm_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_DIR / 'cluster_mmm_summary.json'}")

    # ROI CSV
    df_roi = pd.DataFrame(all_roi_rows)
    df_roi.to_csv(OUTPUT_DIR / "cluster_roi.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'cluster_roi.csv'}")

    # --- Comparison pivot: channel × cluster ---
    if not df_roi.empty:
        # Aggregate across brands (mean elasticity per cluster × channel)
        pivot = (
            df_roi.groupby(["cluster", "channel"])["elasticity"]
            .mean().reset_index()
            .pivot(index="channel", columns="cluster", values="elasticity")
            .fillna(0)
        )
        pivot.to_csv(OUTPUT_DIR / "elasticity_comparison.csv")
        print(f"Saved: {OUTPUT_DIR / 'elasticity_comparison.csv'}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  CLUSTER × CHANNEL ROI COMPARISON")
    print("=" * 60)

    if df_roi.empty:
        print("  No results (insufficient data per cluster×brand)")
        return

    # Show per-cluster top channel
    for cluster_name in sorted(df_roi["cluster"].unique()):
        df_c = df_roi[df_roi["cluster"] == cluster_name]
        mean_r2 = df_c.groupby("brand_code")["r_squared"].first().mean()
        print(f"\n  {cluster_name} (avg R²: {mean_r2:.3f}):")

        # Aggregate across brands
        channel_avg = df_c.groupby("channel").agg(
            avg_elasticity=("elasticity", "mean"),
            avg_roi=("marginal_roi", "mean"),
        ).sort_values("avg_elasticity", ascending=False)

        for ch, row in channel_avg.iterrows():
            marker = "+" if row["avg_elasticity"] > 0 else " "
            print(f"    {marker} {ch:<30} elasticity={row['avg_elasticity']:>+7.3f}  ROI={row['avg_roi']:>+8.4f}")

    # Cross-cluster comparison for top channel
    print(f"\n  --- キャンペーンの弾性値をクラスタ間で比較 ---")
    cp_data = df_roi[df_roi["channel"] == "campaign_entries"]
    if not cp_data.empty:
        cp_by_cluster = cp_data.groupby("cluster")["elasticity"].mean()
        for cluster, elast in cp_by_cluster.items():
            print(f"    {cluster}: {elast:+.3f}")


if __name__ == "__main__":
    main()
