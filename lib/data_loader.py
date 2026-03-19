"""
共通データローダー: config.yaml の読み込み + Supabase/CSV データ取得

全分析スクリプトから共有される。使い方:

    from lib.data_loader import load_config, fetch_data, get_supabase_client

    cfg = load_config()
    client = get_supabase_client(cfg)
    dfs = fetch_data(cfg, client)
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PAGE_SIZE = 1000


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and validate config.yaml."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Parse windows
    w = cfg.get("windows", {})
    if w:
        w["_observation_days"] = int(w.get("observation_days", 60))
        w["_outcome_days"] = int(w.get("outcome_days", 90))
        w["_total_window"] = w["_observation_days"] + w["_outcome_days"]
        w["_data_end_date"] = date.fromisoformat(w.get("data_end_date", "2025-03-10"))
        w["_eligibility_cutoff"] = w["_data_end_date"] - timedelta(days=w["_total_window"])

    # Parse analysis defaults
    a = cfg.setdefault("analysis", {})
    a.setdefault("min_support_floor", 10)
    a.setdefault("min_support_ratio", 0.005)
    a.setdefault("ngram_sizes", [3, 5])
    a.setdefault("first_n_sizes", [3, 5])
    a.setdefault("top_k_report", 20)
    a.setdefault("bootstrap_iterations", 100)
    a.setdefault("bootstrap_sample_ratio", 0.8)
    a.setdefault("stability_threshold", 0.70)
    a.setdefault("min_path_length", 3)

    # Parse output
    o = cfg.setdefault("output", {})
    o.setdefault("dir", "./output")
    o.setdefault("run_full_mode", True)
    o.setdefault("run_no_purchase_mode", True)

    # Build reverse touchpoint mapping: {source: {event_value: CODE}}
    tp_map = cfg.get("touchpoint_mapping", {})
    cfg["_tp_reverse"] = {}
    for source, code_map in tp_map.items():
        reverse = {}
        for code, values in code_map.items():
            if values == "all":
                reverse["__ALL__"] = code
            else:
                for v in values:
                    reverse[v] = code
        cfg["_tp_reverse"][source] = reverse

    cfg["_suppress_codes"] = set(cfg.get("suppress_codes", []))

    return cfg


def get_supabase_client(cfg: dict) -> httpx.Client | None:
    """Create httpx client for Supabase, or None if data_source is CSV."""
    if cfg["data_source"]["type"] != "supabase":
        return None
    load_dotenv(PROJECT_ROOT / ".env")
    return httpx.Client(timeout=60)


def _supabase_headers() -> dict[str, str]:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }


def fetch_all(
    client: httpx.Client,
    table: str,
    select: str,
    filters: dict[str, str] | None = None,
) -> list[dict]:
    """Paginated fetch from Supabase REST API."""
    url = os.environ.get("SUPABASE_URL", "")
    headers = {**_supabase_headers(), "Range-Unit": "items", "Prefer": "count=exact"}
    all_rows: list[dict] = []
    offset = 0
    while True:
        params: dict[str, str] = {
            "select": select,
            "limit": str(PAGE_SIZE),
            "offset": str(offset),
        }
        if filters:
            params.update(filters)
        resp = client.get(f"{url}/rest/v1/{table}", headers=headers, params=params)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return all_rows


def _fetch_csv(source: str, select: str, csv_dir: Path) -> list[dict]:
    """Load data from a CSV file."""
    csv_path = csv_dir / source if not source.endswith(".csv") else csv_dir / source
    if not csv_path.exists():
        csv_path = csv_dir / f"{source}.csv"
    df = pd.read_csv(csv_path)
    if select != "*":
        cols = [c.strip() for c in select.split(",")]
        df = df[[c for c in cols if c in df.columns]]
    return df.to_dict("records")


def fetch_data(cfg: dict, client: httpx.Client | None) -> dict[str, pd.DataFrame]:
    """Fetch all required tables from configured data source."""
    ds = cfg["data_source"]
    source_type = ds["type"]
    csv_dir = Path(ds.get("csv_dir", "./data"))
    if not csv_dir.is_absolute():
        csv_dir = PROJECT_ROOT / csv_dir

    print(f"Fetching data from {source_type}...")

    def fetch_table(source: str, select: str) -> list[dict]:
        if source_type == "supabase":
            return fetch_all(client, source, select)
        else:
            return _fetch_csv(source, select, csv_dir)

    tables_cfg = ds["tables"]
    dfs: dict[str, pd.DataFrame] = {}

    # Customer table
    ct = tables_cfg["customer"]
    rows = fetch_table(ct["source"], ct["select"])
    dfs["customer"] = pd.DataFrame(rows)
    print(f"  {ct['source']}: {len(rows):,} rows")

    # Purchase table
    pt = tables_cfg["purchase"]
    rows = fetch_table(pt["source"], pt["select"])
    dfs["purchase"] = pd.DataFrame(rows)
    print(f"  {pt['source']}: {len(rows):,} rows")

    # Touchpoint sources
    for key, ts in tables_cfg.get("touchpoint_sources", {}).items():
        rows = fetch_table(ts["source"], ts["select"])
        dfs[key] = pd.DataFrame(rows)
        print(f"  {ts['source']}: {len(rows):,} rows")

    return dfs


def ensure_first_date_col(
    cfg: dict, dfs: dict[str, pd.DataFrame], client: httpx.Client | None
) -> None:
    """If first_date_col is missing from customer data, fetch from customer_profile."""
    first_date_col = cfg["data_source"]["tables"]["customer"]["first_date_col"]
    cust_cid_col = cfg["data_source"]["tables"]["customer"]["customer_id_col"]

    if first_date_col not in dfs["customer"].columns:
        if cfg["data_source"]["type"] == "supabase" and client:
            print(f"  {first_date_col} not in customer data, fetching from customer_profile...")
            cp_rows = fetch_all(client, "customer_profile", f"{cust_cid_col},{first_date_col}")
            df_cp = pd.DataFrame(cp_rows)
            dfs["customer"] = dfs["customer"].merge(df_cp, on=cust_cid_col, how="left")
        else:
            raise ValueError(f"{first_date_col} not found in customer data.")
