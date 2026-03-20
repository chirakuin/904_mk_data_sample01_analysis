# CDP Analysis Framework

CDPデータに対する17の分析を統合的に実行するフレームワーク。`config.yaml` を編集するだけで異なるドメイン・データソースに適用可能。

## 分析一覧

### 顧客理解

| # | 分析 | スクリプト | 概要 |
|---|------|-----------|------|
| C1 | クラスタリング | `customer/clustering.py` | KMeans（k=4,5,6自動選択）+ DBSCAN比較 |
| C2 | ペルソナ分析 | （C1に統合） | クラスタの定性的な命名・特徴記述 |
| C3 | コホート分析 | `customer/cohort.py` | 獲得月別のリテンションカーブ |
| C4 | セグメント遷移 | `customer/segment_transition.py` | 四半期スナップショットの遷移行列 |
| C5 | LTV予測 | `customer/ltv.py` | BG/NBD + Gamma-Gamma |
| C6 | 離脱予測 | `customer/churn.py` | RandomForest + 特徴量重要度 |

### パス・ジャーニー分析

| # | 分析 | スクリプト | 概要 |
|---|------|-----------|------|
| J1 | ゴールデンパス | `journey/golden_path.py` | 高価値顧客の共通タッチポイント順序 |
| J2 | アトリビューション | `journey/attribution.py` | マルコフ連鎖 + ラストタッチ + リニア |
| J3 | バスケット分析 | `journey/basket.py` | Apriori併買 + 逐次購買パターン |
| J4 | ファネル分析 | `journey/funnel.py` | デジタルファネル + LINE配信ファネル |

### マーケティング効果測定

| # | 分析 | スクリプト | 概要 |
|---|------|-----------|------|
| M1 | MMM | `effectiveness/mmm.py` | OLS回帰 + Adstock変換（ブランド別） |
| M1b | クラスタ別MMM | `effectiveness/mmm_by_cluster.py` | クラスタごとのメディア弾性値・ROI比較 |
| M2 | ROI予測 | `effectiveness/roi.py` | チャネル別ROI算出 |
| M3 | CP効果分析 | `effectiveness/campaign.py` | 傾向スコアマッチング |
| M4 | 増分効果測定 | `effectiveness/uplift.py` | CATE by Engagement Quartile |

### 予測・最適化

| # | 分析 | スクリプト | 概要 |
|---|------|-----------|------|
| P1 | 需要予測 | `prediction/demand.py` | SARIMAX + 外部変数 |
| P2 | NBA | `prediction/nba.py` | ルールベース推薦（全分析結果を統合） |
| P3 | 予算配分最適化 | `prediction/budget.py` | scipy.optimize (SLSQP) |

## クイックスタート

```bash
cd A03_CDP_analysis
cp .env.example .env    # Supabase利用時のみ
uv sync
```

### 個別実行

```bash
uv run python customer/clustering.py
uv run python journey/golden_path.py
uv run python effectiveness/mmm.py
# ... 各スクリプトは引数なしで独立実行可能
```

### Claude Code スキルによる実行

```
/analyze 1          # 番号で指定
/analyze クラスタ    # キーワードで指定
/analyze 19         # 顧客理解パック（C1-C6一括）
/analyze 22         # フル分析（全スクリプト）
```

分析メニューの全リストは [ANALYSIS_CATALOG.md](ANALYSIS_CATALOG.md) を参照。

### レポート生成

```
/report-technical clustering   # 技術レポート生成
/report-executive clustering   # エグゼクティブ概要版に変換
```

## 実行順序と依存関係

依存する分析がある場合、先に実行が必要。`/analyze` スキルは依存を自動解決する。

```
Phase 1: clustering, cohort, segment_transition, basket, funnel
Phase 2: ltv, churn, attribution, golden_path
Phase 3: mmm, campaign, uplift
Phase 4: roi, demand, budget
Phase 5: nba（最後、他の結果を参照）
```

## プロジェクト構成

```
A03_CDP_analysis/
├── config.yaml                      # 共通設定（データソース・接続情報）
├── pyproject.toml
├── .env / .env.example
│
├── lib/
│   └── data_loader.py               # 共通データローダー（Supabase/CSV両対応）
│
├── customer/                         # 顧客理解（C1-C6）
│   ├── clustering.py, cohort.py, segment_transition.py
│   ├── ltv.py, churn.py
│
├── journey/                          # パス・ジャーニー（J1-J4）
│   ├── golden_path.py, attribution.py
│   ├── basket.py, funnel.py
│
├── effectiveness/                    # 効果測定（M1-M4）
│   ├── mmm.py, mmm_by_cluster.py, roi.py
│   ├── campaign.py, uplift.py
│
├── prediction/                       # 予測・最適化（P1-P3）
│   ├── demand.py, budget.py, nba.py
│
├── other/
│   ├── REPORT.md                     # 技術レポート
│   └── REPORT_EXECUTIVE.md           # エグゼクティブレポート
│
├── ANALYSIS_CATALOG.md              # 分析カタログ（全体設計図）
│
└── output/                          # 分析結果（JSON/CSV）
    ├── clustering/, cohort/, segment_transition/
    ├── ltv/, churn/
    ├── attribution/, basket/, funnel/
    ├── mmm/, roi/, roi_by_cluster/
    ├── campaign/, uplift/
    ├── demand/, budget/, nba/
```

## 設定（config.yaml）

### データソース切り替え

```yaml
data_source:
  type: supabase  # supabase | csv
  csv_dir: ./data

  tables:
    customer:
      source: v_customer_summary      # テーブル名 or CSVファイル名
      customer_id_col: unified_customer_id
      first_date_col: first_known_date
    purchase:
      source: purchase_transaction
      customer_id_col: unified_customer_id
      datetime_col: purchase_datetime
    touchpoint_sources:
      digital:
        source: digital_behavior_log
        customer_id_col: unified_customer_id
        datetime_col: event_datetime
        classify_col: event_name
      # ... 他ソースも同様に定義
```

### 他プロジェクトへの適用

1. `config.yaml` の `data_source.tables` を自社のテーブル/カラム名に変更
2. `touchpoint_mapping` でイベント値→タッチポイントコードを定義
3. `outcome` で成果の定義（購買回数、金額等）を設定
4. `windows` で観察窓・判定窓の長さを調整

業界別の設定例は [ANALYSIS_CATALOG.md](ANALYSIS_CATALOG.md) を参照。

## 技術スタック

- **Python 3.10+** / uv
- pandas, scikit-learn, scipy, statsmodels, lifetimes, mlxtend
- データソース: Supabase REST API / CSV

## 関連プロジェクト

| プロジェクト | 役割 |
|-------------|------|
| A01 (CDP Data Generator) | 合成CDPデータ生成（18エンティティ・491K行） |
| **A03 (CDP Analysis)** | 本プロジェクト。17分析の実行・レポート生成 |
| A04 (CDP Person Model) | SDA — 分析結果を基にした顧客意思決定モデル |

## Claude Code メモリ

このプロジェクトの学習事項・作業履歴は以下に永続保存される:
`~/.claude/projects/-Users-cc-Documents-Code-000-business-030-MK-A01-CDP-data-generator/memory/MEMORY.md`
