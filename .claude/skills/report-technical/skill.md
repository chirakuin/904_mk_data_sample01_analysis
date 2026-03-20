---
name: report-technical
description: 技術レポート生成。A03分析結果のoutput/配下のJSON/CSVを読み取り、分析手法・パラメータ・結果を網羅した技術レポート（REPORT.md）を生成する。
---

# 技術レポート生成

A03_CDP_analysis の分析結果から技術レポート（REPORT.md）を生成する。

## 入力

- 分析結果: `/Users/cc/Documents/Code/000_business/030_MK/A03_CDP_analysis/output/` 配下のJSON/CSVファイル
- 分析スクリプト: 手法の詳細を確認するため、該当するPythonスクリプトも参照

## 出力先

- `/Users/cc/Documents/Code/000_business/030_MK/A03_CDP_analysis/output/<analysis>/REPORT.md`
- 複数分析をまとめる場合: `/Users/cc/Documents/Code/000_business/030_MK/A03_CDP_analysis/other/REPORT.md`

## レポート構成

1. **分析概要**: 目的、対象データ、分析日
2. **分析設計**: 手法の選択理由、パラメータ設定、前処理
3. **結果**: 主要な数値・表・発見事項
4. **考察**: 結果の解釈、限界事項、合成データの制約
5. **次のアクション**: 推奨される追加分析や施策

## 表記ルール

- 業界略語は日本語で書く（RTD → 缶チューハイ類等）
- ペルソナ表記は「[嗜好]好きの[年齢帯][性別]」で統一
- 商品カテゴリには代表的な商品名を併記
- クラスタリングの特徴量に金額・エンゲージメント等の「結果指標」を入れない
- 統計量にはp値・信頼区間を付記

## 手順

1. ユーザーが対象の分析を指定（番号、キーワード、または「全体」）
2. `output/<analysis>/` 配下のJSON/CSVを全て読み取る
3. 該当するPythonスクリプトの冒頭コメントと手法部分を確認
4. 上記構成に従いREPORT.mdを生成
5. 既存のREPORT.mdがあれば内容を比較し、更新が必要か確認してから上書き
