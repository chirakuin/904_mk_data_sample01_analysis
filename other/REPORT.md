# ゴールデンパス分析 v2 レポート

**分析日**: 2026-03-20
**対象データ**: CDP合成データ（Supabase、5,000顧客・約32万行）
**目的**: 高価値顧客に共通するタッチポイント順序（ゴールデンパス）の特定

---

## 1. 分析設計

### ウィンドウ設計（時系列リーク防止）

```
顧客ごとの相対タイムライン:

|← 観察窓 60日 →|← 成果判定窓 90日 →|
Day 0            Day 60              Day 150
(first_known_date)
```

- **観察窓**: タッチポイント収集期間（60日）
- **成果判定窓**: アウトカム判定期間（90日）
- 観察窓の行動を使って判定窓の成果を予測する設計により、**将来情報のリーク**を防止

### アウトカム定義

- **成果群（Outcome=1）**: 判定窓90日間で **2回以上購買**
- **非成果群（Outcome=0）**: 判定窓で0〜1回購買
- 単一軸による対称的な群分け（全適格顧客がどちらかに分類）

### タッチポイント分類（10コード）

| コード | ソース | 条件 |
|--------|--------|------|
| AD_CLK | ad_exposure | click |
| AD_CVR | ad_exposure | conversion |
| LINE_ADD | line_interaction | friend_add |
| LINE_OPEN | line_interaction | message_opened |
| LINE_CLK | line_interaction | message_clicked |
| DIG_BROWSE | digital_behavior_log | page_view, search, recipe_view |
| DIG_ENGAGE | digital_behavior_log | product_click, add_to_cart, video_play等 |
| PURCHASE | purchase_transaction | 全行 |
| CAMPAIGN | campaign_participation | 全行 |
| VENDING | vending_machine_event | purchase |

※ AD_IMP（広告impression）は高頻度・低シグナルのため除外

---

## 2. 対象母集団

| 指標 | 値 |
|------|-----|
| 全顧客 | 5,000人 |
| 適格顧客（150日ウィンドウ確保可能） | 4,221人 |
| 成果群（Outcome=1） | 1,980人（46.9%） |
| 非成果群（Outcome=0） | 2,241人（53.1%） |

### 群間プロファイル比較

| 指標 | 成果群 | 非成果群 |
|------|--------|---------|
| エンゲージメントスコア（平均） | 34.6 | 13.2 |
| CLV_12M（平均） | ¥33,668 | ¥7,590 |
| CLV_12M（中央値） | ¥15,802 | ¥1,016 |
| 判定窓内の購買回数（平均） | 7.0回 | 0.2回 |
| 観察窓内パス長（平均） | 7.5 | 7.5 |
| タッチポイント種類数（平均） | 2.9 | 2.8 |

**所見**: エンゲージメントスコアとCLVで群間に大きな差がある一方、パス長やタッチポイント種類数にはほぼ差がない。これは合成データの `engagement_score` ハブ変数構造を反映している。

---

## 3. 分析結果

### 3.1 遷移確率行列（スクリーニング）

全タッチポイント含むFULLモードの遷移確率差分上位:

| 遷移 | 成果群 | 非成果群 | 差分 | サンプル数 |
|------|--------|---------|------|-----------|
| CAMPAIGN → PURCHASE | 52.9% | 10.3% | **+42.6%** | 十分 |
| AD_CLK → PURCHASE | 68.4% | 26.3% | **+42.1%** | 中程度 |
| PURCHASE → PURCHASE | 62.6% | 21.1% | **+41.5%** | 十分 |
| LINE_ADD → PURCHASE | 51.2% | 9.8% | **+41.4%** | 中程度 |
| LINE_OPEN → PURCHASE | 48.7% | 10.1% | **+38.6%** | 中程度 |
| DIG_ENGAGE → PURCHASE | 46.7% | 10.8% | **+35.8%** | 十分 |
| VENDING → PURCHASE | 49.1% | 14.9% | **+34.2%** | 少数 |
| AD_CLK → DIG_BROWSE | 23.2% | 63.2% | **-40.0%** | 中程度 |

**読み方**: 成果群ではあらゆるタッチポイントから購買への遷移確率が高い。特に**能動的アクション**（広告クリック、LINE追加、CP参加）からの購買転換率の差が大きい。

逆方向の注目点として、非成果群では AD_CLK → DIG_BROWSE（63.2%）が高い。広告をクリックしても購買ではなくサイト閲覧に留まる傾向。

### 3.2 ゴールデンパス（FULLモード）

#### 日次粒度 Top 10（first-3抽出）

| # | パス | Lift | Support(+) | Support(-) | OR | p値 | 安定性 |
|---|------|------|-----------|-----------|-----|------|--------|
| 1 | PURCHASE → PURCHASE → PURCHASE | 24.76 | 23.2% | 0.9% | 31.93 | <0.0001 | 100% |
| 2 | PURCHASE → CAMPAIGN → PURCHASE | 6.11 | 2.2% | 0.4% | 6.23 | <0.0001 | 100% |
| 3 | DIG_BROWSE → PURCHASE → PURCHASE | 5.42 | 4.5% | 0.8% | 5.63 | <0.0001 | 100% |
| 4 | PURCHASE → PURCHASE → CAMPAIGN | 4.99 | 2.6% | 0.5% | 5.10 | <0.0001 | 100% |
| 5 | PURCHASE → DIG_BROWSE → PURCHASE | 4.41 | 3.2% | 0.7% | 4.53 | <0.0001 | 100% |
| 6 | PURCHASE → PURCHASE → DIG_BROWSE | 4.26 | 5.3% | 1.2% | 4.44 | <0.0001 | 100% |
| 7 | PURCHASE → DIG_BROWSE → DIG_ENGAGE | 2.40 | 5.9% | 2.4% | 2.49 | <0.0001 | 100% |
| 8 | DIG_BROWSE → DIG_ENGAGE → PURCHASE | 1.97 | 6.1% | 3.1% | 2.04 | <0.0001 | 100% |
| 9 | PURCHASE → DIG_BROWSE → CAMPAIGN | 1.68 | 1.1% | 0.6% | 1.69 | 0.1909 | 87% |
| 10 | DIG_BROWSE → PURCHASE → DIG_BROWSE | 1.63 | 3.0% | 1.8% | 1.65 | 0.0259 | 100% |

**所見**: 上位6パスはすべてPURCHASEが複数回含まれる。#8の `DIG_BROWSE → DIG_ENGAGE → PURCHASE`（サイト閲覧→商品エンゲージ→購買）が施策的に最も示唆的。

#### 日次 vs 週次の一致度

Top-20パスのうち **19/21が両粒度で共通**（90%一致）。粒度による結果のブレは小さい。

### 3.3 ゴールデンパス（PURCHASE除外モード）

#### 日次粒度 Top 10（first-3抽出）

| # | パス | Lift | Support(+) | Support(-) | OR | p値 | 安定性 |
|---|------|------|-----------|-----------|-----|------|--------|
| 1 | DIG_BROWSE → DIG_BROWSE → CAMPAIGN | 1.45 | 1.4% | 0.9% | 1.46 | 0.2652 | 100% |
| 2 | CAMPAIGN → DIG_BROWSE → DIG_ENGAGE | 1.37 | 2.8% | 2.1% | 1.38 | 0.1547 | 100% |
| 3 | DIG_BROWSE → DIG_ENGAGE → CAMPAIGN | 1.35 | 3.5% | 2.6% | 1.37 | 0.1155 | 100% |
| 4 | DIG_BROWSE → CAMPAIGN → CAMPAIGN | 1.13 | 1.2% | 1.0% | 1.13 | 0.7480 | 100% |
| 5 | CAMPAIGN × 3 | 0.89 | 0.6% | 0.6% | 0.89 | 0.8299 | 1% |
| 6 | CAMPAIGN → DIG_BROWSE → CAMPAIGN | 0.86 | 0.8% | 0.9% | 0.86 | 0.7203 | 99% |
| 7 | CAMPAIGN → CAMPAIGN → DIG_BROWSE | 0.86 | 0.8% | 0.9% | 0.86 | 0.7203 | 98% |
| 8 | DIG_BROWSE → DIG_ENGAGE → DIG_BROWSE | 0.75 | 12.9% | 17.2% | 0.71 | 0.0004 | 100% |
| 9 | DIG_BROWSE → CAMPAIGN → DIG_BROWSE | 0.69 | 2.3% | 3.3% | 0.68 | 0.0683 | 100% |
| 10 | CAMPAIGN → DIG_BROWSE → DIG_BROWSE | 0.68 | 1.2% | 1.8% | 0.68 | 0.1744 | 100% |

**重要な所見**:

- Liftが全て1.0〜1.5の範囲に収まり、**統計的に有意なパスがほぼない**（p値 > 0.05が大半）
- **#8は逆方向に有意**: `DIG_BROWSE → DIG_ENGAGE → DIG_BROWSE`（サイト回遊ループ）は非成果群で17.2%、成果群で12.9%。購買に至らないブラウジングループの存在を示唆
- CAMPAIGN絡みのパス（#1〜#3）がわずかに成果群で多いが、これは合成データの「CP参加には購買実績が必要」という因果構造の反映

### 3.4 PURCHASE除外モードの遷移確率

| 遷移 | 成果群 | 非成果群 | 差分 | 備考 |
|------|--------|---------|------|------|
| AD_CVR → CAMPAIGN | 78.9% | 50.0% | +28.9% | 少数サンプル |
| LINE_OPEN → LINE_CLK | 24.1% | 13.6% | +10.5% | LINEクリック率の差 |
| DIG_ENGAGE → CAMPAIGN | 18.8% | 12.3% | +6.5% | エンゲージ→CP参加 |
| LINE_ADD → DIG_BROWSE | 72.4% | 64.3% | +8.1% | LINE追加→サイト回遊 |

### 3.5 初回タッチポイントの分布

| 初回タッチポイント | 成果群 | 非成果群 |
|-------------------|--------|---------|
| PURCHASE | **57.9%** | 16.4% |
| DIG_BROWSE | 32.2% | **65.2%** |
| DIG_ENGAGE | 4.8% | 10.1% |
| CAMPAIGN | 3.3% | 3.3% |

**所見**: 成果群の過半数が「最初のタッチポイントが購買」。非成果群は「最初のタッチポイントがサイト閲覧」が6割超。観察窓の初期行動が強い予測因子になっている。

### 3.6 安定性評価

Bootstrap（100回・80%サンプル）による安定性:

| モード | 粒度 | 安定性 < 70% のパス割合 |
|--------|------|----------------------|
| FULL | 日次 | 72%（大半が不安定） |
| FULL | 週次 | 68% |
| NO_PURCHASE | 日次 | 56% |
| NO_PURCHASE | 週次 | 50% |

Top 10に入るパスは安定性100%が多いが、**ロングテールのパスは再現性が低い**。レポートではTop 10に絞って解釈することが妥当。

---

## 4. 総合解釈

### 4.1 合成データの構造が結果を支配している

本データは `engagement_score`（0-100）がハブ変数として全行動を駆動する設計:

```
engagement_score → 購買頻度
                 → デジタル行動量
                 → LINE反応率
                 → CP参加率
                 → 広告反応率
```

このため:
- **購買込みのパス分析**: 「よく買う人はよく買う」というトートロジーが支配（lift 20超）
- **購買除外のパス分析**: 非購買行動間の差分が微小（lift 1.0〜1.5、有意性なし）
- **遷移確率**: 全タッチポイント→購買の遷移確率が一様に成果群で高い

### 4.2 それでも得られた知見

| 知見 | 根拠 | 施策示唆 |
|------|------|---------|
| **能動的タッチポイントほど購買転換差が大きい** | AD_CLK(+42%), LINE_ADD(+41%) > DIG_BROWSE(+30%) | リーチ数より**エンゲージメント品質**を追う |
| **サイト回遊ループは非成果群の特徴** | DIG_BROWSE→DIG_ENGAGE→DIG_BROWSE が非成果群で有意に多い | 回遊が長い顧客にはCTAを早めに提示 |
| **初回タッチポイントが購買の人は成果群入りしやすい** | 成果群の58%が初回PURCHASE | 初回購買促進（初回クーポン等）の重要性 |
| **LINEクリック率に群間差** | LINE_OPEN→LINE_CLK: 24% vs 14% | LINEメッセージの個別最適化（クリックさせる設計） |
| **CP参加がパスに介在すると成果群でわずかに多い** | lift 1.35〜1.45（有意ではない） | CP設計の改善余地（ただし要実データ検証） |

### 4.3 手法の妥当性

| 評価項目 | 結果 |
|----------|------|
| 時系列リーク防止 | ウィンドウ分離により確保 |
| 日次 vs 週次の一致度 | 90%一致、粒度ロバスト |
| Bootstrap安定性 | Top 10は安定、ロングテールは不安定 |
| 統計検定 | Fisher正確検定・95%CIで有意性判定が機能 |
| PURCHASE除外による検証 | ハブ変数構造の影響を定量的に確認できた |

---

## 5. 制約と推奨事項

### 合成データ固有の制約

1. **engagement_score のハブ構造**: 全行動間の相関が人工的に高く、非自明なパターンが検出しにくい
2. **因果関係の不在**: 相関パターンは検出できるが、「Xをしたから買った」という因果推論はできない
3. **CP参加の購買依存**: 合成データではCP参加に購買実績が必要なため、CAMPAIGN絡みのパスにバイアスがある

### 実データ適用時の推奨事項

1. **同じスクリプトをそのまま適用可能** — Supabase接続情報を変えるだけで実行できる
2. **PURCHASE除外モードで初めて非自明なパターンが出る可能性** — 実データでは個人差が大きいため
3. **ウィンドウ長の調整** — 実際の購買サイクルに合わせてOBSERVATION_DAYS/OUTCOME_DAYSを調整
4. **タッチポイント粒度の細分化** — DIG_BROWSEをpage_view/search/recipe_viewに分割して再分析
5. **傾向スコアマッチングの追加** — エンゲージメントスコアで層別化し、同スコア内でのパス比較

---

## 6. 出力ファイル一覧

```
output/
├── full/                           # 全タッチポイント（PURCHASE含む）
│   ├── golden_paths_summary.json
│   ├── path_comparison_daily.csv
│   ├── path_comparison_weekly.csv
│   ├── customer_journeys.csv
│   ├── transition_matrix_daily.csv
│   ├── transition_matrix_weekly.csv
│   └── granularity_comparison.csv
└── no_purchase/                    # PURCHASE除外（ナーチャリング導線分析）
    ├── golden_paths_summary.json
    ├── path_comparison_daily.csv
    ├── path_comparison_weekly.csv
    ├── customer_journeys.csv
    ├── transition_matrix_daily.csv
    ├── transition_matrix_weekly.csv
    └── granularity_comparison.csv
```

---

## 付録: パラメータ設定

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| OBSERVATION_DAYS | 60 | 観察窓 |
| OUTCOME_DAYS | 90 | 成果判定窓 |
| MIN_PATH_LENGTH | 3 | 最低パス長 |
| MIN_SUPPORT | 21 | 最低サポート閾値（max(10, 顧客数×0.5%)） |
| BOOTSTRAP_ITER | 100 | Bootstrap反復回数 |
| BOOTSTRAP_SAMPLE_RATIO | 0.8 | Bootstrapサンプル比率 |
| SUPPRESS_AD_IMP | True | AD_IMPをパスから除外 |
| SUPPRESS_PURCHASE_IN_PATH | True | PURCHASE除外モード有効 |
| DATA_END_DATE | 2025-03-10 | データ終端日 |
| ELIGIBILITY_CUTOFF | 2024-10-11 | 適格条件（first_known_date上限） |
