# Golden Path Analysis - 汎用ゴールデンパス分析フレームワーク

高価値顧客に共通するタッチポイント順序（ゴールデンパス）を特定する汎用分析ツール。`config.yaml` を編集するだけで、異なるドメイン・データソースに適用可能。

## 特徴

- **config.yaml 駆動**: データソース・タッチポイント定義・アウトカム定義を設定ファイルで管理
- **複数データソース対応**: Supabase REST API / CSV ファイル
- **時系列リーク防止**: 観察窓 / 成果判定窓を分離
- **統計的厳密性**: Fisher正確検定、95%CI、Bootstrap安定性
- **2粒度比較**: 日次・週次の結果を照合
- **2モード**: FULL（全タッチポイント）+ NO_PURCHASE（ナーチャリング導線分析）
- **遷移確率分析**: 成果群 vs 非成果群の遷移確率差 → 介入ポイント候補

## クイックスタート

```bash
cd A03_CDP_analysis
cp .env.example .env    # Supabase利用時のみ
uv sync
uv run python golden_path_analysis.py
```

## 他プロジェクトへの適用方法

### 1. config.yaml を編集

```yaml
# データソースを変更（Supabase or CSV）
data_source:
  type: csv                    # supabase → csv に変更
  csv_dir: ./data              # CSVファイルの格納先

  tables:
    customer:
      source: customers.csv    # ファイル名 or テーブル名
      customer_id_col: user_id # 顧客IDカラム名
      first_date_col: created_at  # 初回観測日カラム名

    purchase:
      source: orders.csv
      customer_id_col: user_id
      datetime_col: order_datetime

    touchpoint_sources:
      # 必要なソースだけ定義（不要なものは削除可）
      email:
        source: email_events.csv
        customer_id_col: user_id
        datetime_col: sent_at
        classify_col: event_type
```

### 2. タッチポイント分類を定義

```yaml
touchpoint_mapping:
  email:
    EMAIL_OPEN:
      - opened
    EMAIL_CLK:
      - clicked
  # classify_col がないソースは "all" で全行に同一コードを付与
  webinar:
    WEBINAR: all
```

### 3. アウトカムを定義

```yaml
outcome:
  metric: purchase_count   # 判定窓内の購買回数
  threshold: 2             # 2回以上 → 成果群
```

### 4. ウィンドウ長を調整

```yaml
windows:
  observation_days: 60     # 観察窓（タッチポイント収集期間）
  outcome_days: 90         # 成果判定窓
  data_end_date: "2025-03-10"  # データ終端日
```

## 業界別の設定例

### EC / D2C
```yaml
windows:
  observation_days: 30
  outcome_days: 60
touchpoint_mapping:
  web:
    BROWSE: [page_view, search]
    ENGAGE: [add_to_cart, wishlist]
  email:
    EMAIL_OPEN: [opened]
    EMAIL_CLK: [clicked]
  sns:
    SNS_CLK: [link_click]
outcome:
  threshold: 2
```

### SaaS
```yaml
windows:
  observation_days: 14
  outcome_days: 30
touchpoint_mapping:
  app:
    LOGIN: [login]
    FEATURE_USE: [feature_activated, report_created]
    INVITE: [team_invite_sent]
  email:
    ONBOARD_OPEN: [onboarding_email_opened]
    ONBOARD_CLK: [onboarding_email_clicked]
outcome:
  metric: purchase_count  # login_count等に差し替え可能
  threshold: 5
```

### ゲーム
```yaml
windows:
  observation_days: 7
  outcome_days: 14
touchpoint_mapping:
  game:
    TUTORIAL: [tutorial_complete]
    PLAY: [stage_clear, pvp_match]
    SOCIAL: [friend_add, guild_join]
  push:
    PUSH_OPEN: [push_opened]
outcome:
  threshold: 3  # 課金3回以上
```

## 設計思想: ウィンドウによる時系列リーク防止

```
顧客ごとの相対タイムライン:

|← 観察窓 →|← 成果判定窓 →|
Day 0       Day N          Day N+M
(first_date)

観察窓内の行動 → パス分析対象
判定窓内の成果 → アウトカムラベル
```

観察窓と判定窓を分離することで「未来の情報で過去を評価する」リークを防止。

## 出力ファイル

FULLモードとNO_PURCHASEモードそれぞれに以下を出力:

```
output/
├── full/
│   ├── golden_paths_summary.json   # 全体サマリ
│   ├── path_comparison_daily.csv   # パス統計（support, lift, OR, CI, p値, 安定性）
│   ├── path_comparison_weekly.csv
│   ├── customer_journeys.csv       # 顧客別シーケンス
│   ├── transition_matrix_daily.csv # 遷移確率行列
│   ├── transition_matrix_weekly.csv
│   └── granularity_comparison.csv  # 日次vs週次の一致度
└── no_purchase/
    └── (同上)
```

## config.yaml パラメータ一覧

### data_source

| キー | 説明 | 例 |
|------|------|-----|
| `type` | データソース種別 | `supabase` / `csv` |
| `csv_dir` | CSVディレクトリ（type=csv時） | `./data` |
| `tables.customer.source` | 顧客テーブル/ファイル名 | `v_customer_summary` |
| `tables.customer.customer_id_col` | 顧客IDカラム | `unified_customer_id` |
| `tables.customer.first_date_col` | 初回観測日カラム | `first_known_date` |
| `tables.purchase.source` | 購買テーブル/ファイル名 | `purchase_transaction` |
| `tables.purchase.datetime_col` | 購買日時カラム | `purchase_datetime` |
| `tables.touchpoint_sources.*` | タッチポイントソース定義 | 複数定義可能 |

### touchpoint_mapping

| キー | 説明 |
|------|------|
| `{source}.{CODE}` | イベント値のリスト → タッチポイントコードにマッピング |
| `{source}.{CODE}: all` | classify_col なしの場合、全行にこのコードを付与 |

### windows

| キー | デフォルト | 説明 |
|------|-----------|------|
| `observation_days` | 60 | 観察窓（日数） |
| `outcome_days` | 90 | 成果判定窓（日数） |
| `data_end_date` | 2025-03-10 | データ終端日 |

### outcome

| キー | デフォルト | 説明 |
|------|-----------|------|
| `metric` | purchase_count | 成果指標 |
| `threshold` | 2 | 成果群の閾値（以上） |

### analysis

| キー | デフォルト | 説明 |
|------|-----------|------|
| `min_path_length` | 3 | 最低パス長 |
| `min_support_ratio` | 0.005 | 最低サポート比率 |
| `min_support_floor` | 10 | 最低サポート下限 |
| `ngram_sizes` | [3, 5] | N-gramサイズ |
| `first_n_sizes` | [3, 5] | First-N抽出サイズ |
| `top_k_report` | 20 | レポート上位K件 |
| `bootstrap_iterations` | 100 | Bootstrap反復回数 |
| `bootstrap_sample_ratio` | 0.8 | Bootstrapサンプル比率 |

### output

| キー | デフォルト | 説明 |
|------|-----------|------|
| `dir` | ./output | 出力ディレクトリ |
| `run_full_mode` | true | FULLモード実行 |
| `run_no_purchase_mode` | true | PURCHASE除外モード実行 |

### suppress_codes

パスから除外するタッチポイントコードのリスト（例: `AD_IMP`）。

## 統計指標

各パスについて以下を算出:

| 指標 | 説明 |
|------|------|
| Support (成果群/非成果群) | 各群でのパス出現率 |
| Lift | 成果群Support / 非成果群Support |
| 差分 | 成果群Support - 非成果群Support |
| オッズ比 (OR) | 関連の強さ |
| 95% CI | オッズ比の信頼区間 |
| p値 | Fisher正確検定（有意性） |
| 安定性 | Bootstrap 100回での再現率 |

## Claude Code メモリ

このプロジェクトの学習事項・作業履歴は以下に永続保存される:
`~/.claude/projects/-Users-cc-Documents-Code-000-business-030-MK-A01-CDP-data-generator/memory/MEMORY.md`
