"""
Churn Prediction Analysis (C6)

v_customer_summary + purchase_transaction から解約予測モデルを構築する。
RandomForestClassifier で特徴量重要度と予測結果を出力。

Usage:
    uv run python customer/churn.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from lib.data_loader import fetch_all, get_supabase_client, load_config

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "churn"

CID_COL = "unified_customer_id"
CHURN_DAYS = 90  # No purchase in last N days = churned

FEATURE_COLS = [
    "engagement_score",
    "clv_12m",
    "monthly_spend",
    "churn_risk_score",
    "health_consciousness",
    "lifetime_points_earned",
    "purchase_count",
    "avg_purchase_amount",
    "days_since_last_purchase",
]


# ============================================================
# Data Fetch
# ============================================================


def fetch_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch v_customer_summary and purchase_transaction."""
    client = get_supabase_client(cfg)
    if client is None:
        raise RuntimeError("Supabase client required")
    try:
        cs_rows = fetch_all(client, "v_customer_summary", "*")
        pt_rows = fetch_all(
            client,
            "purchase_transaction",
            f"{CID_COL},purchase_datetime,total_amount",
        )
    finally:
        client.close()

    df_cust = pd.DataFrame(cs_rows)
    df_purch = pd.DataFrame(pt_rows)
    print(f"Fetched v_customer_summary: {len(df_cust):,} rows")
    print(f"Fetched purchase_transaction: {len(df_purch):,} rows")
    return df_cust, df_purch


# ============================================================
# Feature Engineering
# ============================================================


def engineer_features(
    df_cust: pd.DataFrame, df_purch: pd.DataFrame
) -> pd.DataFrame:
    """Build features and churn label."""
    df_purch = df_purch.copy()
    df_purch["purchase_datetime"] = pd.to_datetime(df_purch["purchase_datetime"])
    df_purch["total_amount"] = pd.to_numeric(df_purch["total_amount"], errors="coerce").fillna(0)

    # Data end date
    data_end = df_purch["purchase_datetime"].max()
    churn_cutoff = data_end - pd.Timedelta(days=CHURN_DAYS)
    print(f"\nData end date: {data_end.date()}")
    print(f"Churn cutoff: {churn_cutoff.date()} (no purchase after this = churned)")

    # Purchase aggregates per customer
    purch_agg = (
        df_purch.groupby(CID_COL)
        .agg(
            purchase_count=("purchase_datetime", "count"),
            avg_purchase_amount=("total_amount", "mean"),
            last_purchase_date=("purchase_datetime", "max"),
        )
        .reset_index()
    )
    purch_agg["days_since_last_purchase"] = (
        data_end - purch_agg["last_purchase_date"]
    ).dt.days

    # Merge with customer summary
    df = df_cust.merge(purch_agg, on=CID_COL, how="left")

    # Fill missing purchase features (customers with no purchases)
    df["purchase_count"] = df["purchase_count"].fillna(0).astype(int)
    df["avg_purchase_amount"] = df["avg_purchase_amount"].fillna(0.0)
    df["days_since_last_purchase"] = df["days_since_last_purchase"].fillna(
        (data_end - data_end.replace(year=data_end.year - 2)).days  # max possible
    )

    # Churn label
    df["churned"] = (df["days_since_last_purchase"] > CHURN_DAYS).astype(int)

    # Fill missing feature columns with median
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  WARNING: {col} not found, filling with 0")
            df[col] = 0.0
        else:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                median_val = df[col].median()
                print(f"  {col}: {n_missing} missing values filled with median ({median_val:.2f})")
                df[col] = df[col].fillna(median_val)

    n_churned = df["churned"].sum()
    n_active = len(df) - n_churned
    print(f"\nChurn distribution:")
    print(f"  Churned: {n_churned:,} ({n_churned/len(df)*100:.1f}%)")
    print(f"  Active: {n_active:,} ({n_active/len(df)*100:.1f}%)")

    return df


# ============================================================
# Model Training
# ============================================================


def train_model(df: pd.DataFrame) -> tuple[RandomForestClassifier, dict, pd.DataFrame]:
    """Train RandomForest and return model, metrics, predictions."""
    X = df[FEATURE_COLS].values
    y = df["churned"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "churn_rate_train": round(float(y_train.mean()), 4),
        "churn_rate_test": round(float(y_test.mean()), 4),
    }

    print(f"\nModel performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    # Full predictions
    df_pred = df[[CID_COL]].copy()
    df_pred["churn_probability"] = clf.predict_proba(X)[:, 1]
    df_pred["churn_predicted"] = clf.predict(X)
    df_pred["churned_actual"] = df["churned"].values

    return clf, metrics, df_pred


# ============================================================
# Output
# ============================================================


def save_outputs(
    clf: RandomForestClassifier,
    metrics: dict,
    df_pred: pd.DataFrame,
) -> None:
    """Save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Feature importance
    fi = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    fi["importance"] = fi["importance"].round(4)
    fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'feature_importance.csv'}")

    # Predictions
    df_pred.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'predictions.csv'}")

    # Metrics
    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_DIR / 'model_metrics.json'}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  Churn Prediction Analysis (C6)")
    print("=" * 60)

    cfg = load_config()
    df_cust, df_purch = fetch_data(cfg)

    if df_cust.empty:
        print("ERROR: No customer data. Exiting.")
        sys.exit(1)
    if df_purch.empty:
        print("ERROR: No purchase data. Exiting.")
        sys.exit(1)

    print("\nEngineering features...")
    df = engineer_features(df_cust, df_purch)

    print("\nTraining model...")
    clf, metrics, df_pred = train_model(df)

    save_outputs(clf, metrics, df_pred)

    # Print summary
    print("\n" + "=" * 60)
    print("  CHURN PREDICTION SUMMARY")
    print("=" * 60)

    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\nTop 5 feature importances:")
    fi = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
    for feat, imp in fi[:5]:
        print(f"  {feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
