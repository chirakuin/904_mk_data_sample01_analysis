"""
J3: Basket Analysis (Association Rules + Sequential Basket)

購買トランザクションから商品間の関連ルール（Apriori）と
時系列的な逐次購買パターン（30日以内の次回購買）を分析する。

Usage:
    uv run python journey/basket.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Add project root to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.data_loader import (
    load_config,
    fetch_all,
    get_supabase_client,
)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "basket"

# Parameters
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.1
METRIC = "lift"
SEQUENTIAL_WINDOW_DAYS = 30


# ============================================================
# 1. DATA FETCHING
# ============================================================


def fetch_purchase_and_product(
    cfg: dict, client
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch purchase_transaction and product_master."""
    ds = cfg["data_source"]

    if ds["type"] == "supabase":
        print("Fetching purchase_transaction...")
        purchase_rows = fetch_all(
            client,
            "purchase_transaction",
            "unified_customer_id,product_id,purchase_datetime",
        )
        print(f"  purchase_transaction: {len(purchase_rows):,} rows")

        print("Fetching product_master...")
        product_rows = fetch_all(
            client,
            "product_master",
            "product_id,product_name,brand_name,category_l2",
        )
        print(f"  product_master: {len(product_rows):,} rows")
    else:
        csv_dir = Path(ds.get("csv_dir", "./data"))
        if not csv_dir.is_absolute():
            csv_dir = Path(__file__).parent.parent / csv_dir

        df_p = pd.read_csv(csv_dir / "purchase_transaction.csv")
        cols_p = ["unified_customer_id", "product_id", "purchase_datetime"]
        purchase_rows = df_p[[c for c in cols_p if c in df_p.columns]].to_dict(
            "records"
        )
        print(f"  purchase_transaction: {len(purchase_rows):,} rows")

        df_m = pd.read_csv(csv_dir / "product_master.csv")
        cols_m = ["product_id", "product_name", "brand_name", "category_l2"]
        product_rows = df_m[[c for c in cols_m if c in df_m.columns]].to_dict(
            "records"
        )
        print(f"  product_master: {len(product_rows):,} rows")

    return pd.DataFrame(purchase_rows), pd.DataFrame(product_rows)


# ============================================================
# 2. ASSOCIATION RULES (APRIORI)
# ============================================================


def compute_association_rules(
    df_purchase: pd.DataFrame, df_product: pd.DataFrame
) -> pd.DataFrame:
    """Build customer-product matrix and compute association rules."""
    # Merge to get product names
    df = df_purchase.merge(df_product[["product_id", "product_name"]], on="product_id", how="left")
    df = df.dropna(subset=["product_name"])

    # Build transactions: list of product sets per customer
    transactions = (
        df.groupby("unified_customer_id")["product_name"]
        .apply(lambda x: list(set(x)))
        .tolist()
    )

    print(f"\n  Unique customers with purchases: {len(transactions):,}")
    print(f"  Unique products: {df['product_name'].nunique()}")

    # TransactionEncoder + Apriori
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    print(f"  Running Apriori (min_support={MIN_SUPPORT}, max_len=2)...")
    frequent_items = apriori(
        df_encoded, min_support=MIN_SUPPORT, use_colnames=True, max_len=2
    )
    print(f"  Frequent itemsets: {len(frequent_items)}")

    if frequent_items.empty:
        print("  No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(frequent_items, metric=METRIC, min_threshold=1.0)
    # Filter by confidence
    rules = rules[rules["confidence"] >= MIN_CONFIDENCE].copy()

    # Format antecedents/consequents as strings
    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(x))
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(x))
    )

    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    print(f"  Association rules (confidence >= {MIN_CONFIDENCE}): {len(rules)}")

    return rules


# ============================================================
# 3. SEQUENTIAL BASKET ANALYSIS
# ============================================================


def compute_sequential_rules(
    df_purchase: pd.DataFrame,
    df_product: pd.DataFrame,
    window_days: int = SEQUENTIAL_WINDOW_DAYS,
) -> pd.DataFrame:
    """Compute P(product B | bought product A within N days).

    For each customer, look at pairs where product B was purchased
    within `window_days` days after product A.
    """
    df = df_purchase.merge(
        df_product[["product_id", "product_name"]], on="product_id", how="left"
    )
    df = df.dropna(subset=["product_name"])
    df["purchase_dt"] = pd.to_datetime(df["purchase_datetime"])
    df = df.sort_values(["unified_customer_id", "purchase_dt"])

    # Count: how many customers bought product A
    product_buyers: defaultdict[str, set] = defaultdict(set)
    # Count: how many customers bought B within N days after A
    sequential_pairs: defaultdict[tuple[str, str], set] = defaultdict(set)

    print(f"\n  Computing sequential patterns (window={window_days} days)...")

    for cid, group in df.groupby("unified_customer_id"):
        # Deduplicate: keep first purchase per product per customer for sequential analysis
        seen_products = set()
        purchases = []
        for _, row in group.iterrows():
            prod = row["product_name"]
            product_buyers[prod].add(cid)
            purchases.append((prod, row["purchase_dt"]))

        # Only check unique product pairs within window (skip same-product repeats)
        for i, (prod_a, dt_a) in enumerate(purchases):
            seen_in_window = set()
            for j in range(i + 1, len(purchases)):
                prod_b, dt_b = purchases[j]
                delta = (dt_b - dt_a).days
                if delta > window_days:
                    break
                if prod_a != prod_b and prod_b not in seen_in_window:
                    sequential_pairs[(prod_a, prod_b)].add(cid)
                    seen_in_window.add(prod_b)

    # Build rules
    rows = []
    for (prod_a, prod_b), customers_ab in sequential_pairs.items():
        n_a = len(product_buyers[prod_a])
        n_ab = len(customers_ab)
        n_b = len(product_buyers[prod_b])
        total_customers = len(
            set().union(*product_buyers.values()) if product_buyers else set()
        )

        support_a = n_a / total_customers if total_customers > 0 else 0
        support_b = n_b / total_customers if total_customers > 0 else 0
        support_ab = n_ab / total_customers if total_customers > 0 else 0
        confidence = n_ab / n_a if n_a > 0 else 0
        lift = confidence / support_b if support_b > 0 else 0

        if n_ab >= 5:  # Minimum pair frequency
            rows.append(
                {
                    "antecedent": prod_a,
                    "consequent": prod_b,
                    "support_a": round(support_a, 6),
                    "support_b": round(support_b, 6),
                    "support_ab": round(support_ab, 6),
                    "confidence": round(confidence, 6),
                    "lift": round(lift, 4),
                    "count_a": n_a,
                    "count_ab": n_ab,
                    "window_days": window_days,
                }
            )

    df_seq = pd.DataFrame(rows)
    if not df_seq.empty:
        df_seq = df_seq.sort_values("lift", ascending=False).reset_index(drop=True)

    print(f"  Sequential rules (min count >= 5): {len(df_seq)}")
    return df_seq


# ============================================================
# 4. MAIN
# ============================================================


def main() -> None:
    print("=" * 60)
    print("  J3: Basket Analysis (Association Rules + Sequential)")
    print("=" * 60)

    cfg = load_config()
    client = get_supabase_client(cfg)

    df_purchase, df_product = fetch_purchase_and_product(cfg, client)
    if client:
        client.close()

    # Association rules
    print("\n--- Association Rules (Apriori) ---")
    df_rules = compute_association_rules(df_purchase, df_product)

    # Sequential basket analysis
    print("\n--- Sequential Basket Analysis ---")
    df_sequential = compute_sequential_rules(df_purchase, df_product)

    # Top pairs JSON
    top_pairs = []
    if not df_rules.empty:
        for _, r in df_rules.head(20).iterrows():
            top_pairs.append(
                {
                    "antecedents": r["antecedents_str"],
                    "consequents": r["consequents_str"],
                    "support": round(r["support"], 4),
                    "confidence": round(r["confidence"], 4),
                    "lift": round(r["lift"], 4),
                }
            )

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not df_rules.empty:
        # Select output columns
        out_cols = [
            "antecedents_str",
            "consequents_str",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
            "zhangs_metric",
        ]
        existing_cols = [c for c in out_cols if c in df_rules.columns]
        df_rules[existing_cols].to_csv(
            OUTPUT_DIR / "association_rules.csv", index=False
        )

    if not df_sequential.empty:
        df_sequential.to_csv(OUTPUT_DIR / "sequential_rules.csv", index=False)

    with open(OUTPUT_DIR / "top_pairs.json", "w", encoding="utf-8") as f:
        json.dump(
            {"association_top_pairs": top_pairs}, f, ensure_ascii=False, indent=2
        )

    print(f"\nOutputs saved to {OUTPUT_DIR}/")

    # Print summary
    print("\n" + "=" * 60)
    print("  TOP 10 ASSOCIATION RULES (by lift)")
    print("=" * 60)
    if not df_rules.empty:
        print(
            f"\n  {'#':<3} {'Antecedents':<30} {'Consequents':<30} "
            f"{'Lift':>6} {'Conf':>6} {'Supp':>6}"
        )
        print(f"  {'-'*3} {'-'*30} {'-'*30} {'-'*6} {'-'*6} {'-'*6}")
        for i, (_, r) in enumerate(df_rules.head(10).iterrows(), 1):
            print(
                f"  {i:<3} {r['antecedents_str']:<30} {r['consequents_str']:<30} "
                f"{r['lift']:>6.2f} {r['confidence']:>6.3f} {r['support']:>6.3f}"
            )
    else:
        print("  No association rules found.")

    print("\n" + "=" * 60)
    print("  TOP 10 SEQUENTIAL RULES (by lift)")
    print("=" * 60)
    if not df_sequential.empty:
        print(
            f"\n  {'#':<3} {'Antecedent':<30} {'Consequent':<30} "
            f"{'Lift':>6} {'Conf':>6} {'Count':>6}"
        )
        print(f"  {'-'*3} {'-'*30} {'-'*30} {'-'*6} {'-'*6} {'-'*6}")
        for i, (_, r) in enumerate(df_sequential.head(10).iterrows(), 1):
            print(
                f"  {i:<3} {r['antecedent']:<30} {r['consequent']:<30} "
                f"{r['lift']:>6.2f} {r['confidence']:>6.3f} {r['count_ab']:>6}"
            )
    else:
        print("  No sequential rules found.")

    print("=" * 60)


if __name__ == "__main__":
    main()
