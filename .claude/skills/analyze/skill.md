---
name: analyze
description: CDP分析実行スキル。A03の分析スクリプトを番号またはキーワードで選択・実行し、結果をレポートにまとめる。
---

# CDP分析実行

A03_CDP_analysis の分析スクリプトを実行するスキル。

## プロジェクトパス

- A03: `/Users/cc/Documents/Code/000_business/030_MK/A03_CDP_analysis/`
- Python実行: `cd /Users/cc/Documents/Code/000_business/030_MK/A03_CDP_analysis && uv run python <module>/<script>.py`

## 分析メニュー

ユーザーが番号、キーワード、または質問で分析を指定する。以下のマッピングに従い、該当スクリプトを実行する。

### 顧客を知る

| # | 問い | スクリプト |
|---|------|-----------|
| 1 | うちの顧客はどんなタイプがいるか？ | `customer/clustering.py` |
| 2 | 顧客の将来的な価値はどれくらいか？ | `customer/ltv.py` |
| 3 | 離脱しそうな顧客は誰か？ | `customer/churn.py` |
| 4 | 新規顧客はどれくらい定着しているか？ | `customer/cohort.py` |
| 5 | 顧客のステータスはどう変化しているか？ | `customer/segment_transition.py` |

### 購買を知る

| # | 問い | スクリプト |
|---|------|-----------|
| 6 | よく一緒に買われる商品の組み合わせは？ | `journey/basket.py` |
| 7 | ある商品を買った人が次に何を買うか？ | `journey/basket.py` |

### チャネルを知る

| # | 問い | スクリプト |
|---|------|-----------|
| 8 | どのチャネルが購買に最も貢献しているか？ | `journey/attribution.py` |
| 9 | サイト上のどこで離脱しているか？ | `journey/funnel.py` |
| 10 | LINEメッセージの開封→クリック率は？ | `journey/funnel.py` |
| 11 | 顧客が購買に至る典型的な行動パターンは？ | `journey/golden_path.py` |

### 施策の効果を知る

| # | 問い | スクリプト |
|---|------|-----------|
| 12 | キャンペーンは購買を増やしたか？ | `effectiveness/campaign.py` |
| 13 | 施策がなかった場合と比べてどれだけ効果があったか？ | `effectiveness/uplift.py` |
| 14 | 各メディアの売上への貢献度は？ | `effectiveness/mmm.py` |
| 15 | 各チャネルの投資対効果は？ | `effectiveness/mmm.py` → `effectiveness/roi.py` |

### 未来を予測する

| # | 問い | スクリプト |
|---|------|-----------|
| 16 | 来月の売上はどれくらいか？ | `prediction/demand.py` |
| 17 | 広告予算をどう配分すべきか？ | `effectiveness/mmm.py` → `prediction/budget.py` |
| 18 | 次に各顧客に何をすべきか？ | `customer/clustering.py` → `customer/ltv.py` → `customer/churn.py` → `prediction/nba.py` |

### パック実行

| # | 内容 | 実行する分析 |
|---|------|------------|
| 19 | 顧客理解パック | 1+2+3+4+5 |
| 20 | チャネル分析パック | 8+9+10+11 |
| 21 | 効果測定パック | 12+13+14+15 |
| 22 | フル分析（全て実行） | 全スクリプト |

### 追加分析

| キーワード | スクリプト |
|-----------|-----------|
| クラスタ別MMM | `effectiveness/mmm_by_cluster.py` |

## 依存関係の自動解決

実行順序（依存があれば先に実行）:

```
Phase 1: clustering, cohort, segment_transition, basket, funnel
Phase 2: ltv, churn, attribution, golden_path
Phase 3: mmm, campaign, uplift
Phase 4: roi, demand, budget
Phase 5: nba（最後、他の結果を参照）
```

## 実行手順

1. ユーザーの入力（番号、キーワード、質問文）から該当分析を特定
2. 依存する分析の出力が `output/` に存在するか確認。なければ先に実行
3. `cd /Users/cc/Documents/Code/000_business/030_MK/A03_CDP_analysis && uv run python <module>/<script>.py` で実行
4. `output/` 配下の結果ファイル（JSON/CSV）を読み取り、要点をユーザーに報告
5. 必要に応じて `/report-technical` または `/report-executive` でレポート生成を提案
