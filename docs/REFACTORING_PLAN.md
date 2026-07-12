# 論文再現性を維持した保守的リファクタリング

> この文書は、論文の再現性と既存インターフェースを維持しながら、
> リポジトリ全体を保守的に整理した際の実施計画と検証方針を記録したものです。
>
> 対応コミット: `4b447c3140276a522d957e1211bac3cd4f2d54b9`

## Summary

- `arxiv/` の論文を仕様として参照し、フルモデルと3アブレーションの挙動を維持する。
- 上流由来のAudioLDM・CLAP・BeatDance実装は削減せず、安全と確認できる不要import・デバッグ処理・死んだコメントのみ整理する。
- `arxiv/` 自体は編集・追跡せず、既存の未追跡ファイルも保護する。

## Implementation Changes

- 現在削除状態の `keypoints.py` を復元し、公開関数のシグネチャを維持したまま、入力検証、命名、mutable default、重複したピーク検出計算を整理する。
- Dance-only BeatDanceエンコーダーを単一実装へ統合する。音楽側モジュールを生成後に削除する処理や、出力に使われないattention計算を除去し、既存チェックポイントのキー互換性と旧クラス名のaliasを残す。
- 全設定のBeatDance targetと相対パスを統一し、`sys.path`変更や個人環境の絶対パスを除去する。設定対応は以下で固定する。
  - Full: `audioldm_original_medium_stretch_pretrained_frozen.yaml`
  - w/o Contrastive Pretraining: `audioldm_original_medium.yaml`
  - w/o MotionBERT: `audioldm_original_medium_stretch_wo_mb.yaml`
  - AudioLDM default: `2025_11_08_dance_controlnet/audioldm_original_medium_stretch.yaml`
- 全Pythonコードを監査し、未使用・重複import、未使用変数、placeholderのないf-string、`raise NotImplemented`、実行中の`pdb`/`ipdb`、コメントアウトされたデバッグ処理を手動確認後に除去する。アルゴリズム説明、shape、出典、互換性上必要なコメントは残す。
- 既存の学習・評価シェルは、Full設定を既定値として `CONFIG`、`CHECKPOINT`、`CUDA_VISIBLE_DEVICES` で上書き可能にし、一時評価スクリプトの機能を正式入口へ統合する。
- TensorBoardイベント、実験ログ、`.origin`、`*_origin.py`、無効なコピー/空ファイル、scratch notebook、`tmp_*` を削除する。`.gitignore` を重複なく整理し、今後のログ、キャッシュ、`.DS_Store` を除外する。
- READMEを、環境構築、データ・チェックポイント配置、MotionBERT特徴抽出、BeatDance対照事前学習、ControlNet学習、評価、上記アブレーション対応表の順に刷新する。既存の補助ドキュメントは保持し、READMEから必要箇所へリンクする。
- Ruff設定は安全な検査ルールに限定して追加し、変更ファイルだけ整形する。約108ファイルに及ぶ全体フォーマットは行わない。

## Public Interfaces

- `keypoints.py` の既存関数名と引数は維持する。
- Dance-only encoderの設定targetを新しい単一実装へ統一するが、旧 `DanceOnlyBeatDanceWrapper` importは互換aliasで維持する。
- 学習・評価Python CLIは維持し、シェルラッパーに環境変数による設定切替を追加する。
- 評価CLIの既存音声評価と任意の `--motion` BAS評価を維持する。

## Test Plan

- keypoint読込、NaN補間、リサンプリング、COCO→H36M変換、スケーリング、beat feature shapeの単体テストを追加する。
- Dance-only encoderについて、出力shape、不要な音楽側パラメータがないこと、既存BeatDanceチェックポイントを`strict=False`で読み込めることを検証する。
- beat scoreの空ビート、長さ不一致、Tensor入力、BAS計算をテストする。
- 全YAML/JSON解析、設定target import、Python compile、選択したRuffルール、`bash -n`、絶対パス・debugger・conflict markerの残存検査を実行する。
- データセット、GPU、チェックポイントが利用可能な場合のみ、Fullと3アブレーションのモデル初期化・1 batch forwardを追加検証する。フル学習は検証範囲外とする。

## Assumptions

- 保守的整理のため、論文外の既存実験スクリプトと上流サブシステムは残す。
- 追跡済み生成物は削除するが、ローカルの未追跡 `arxiv/` と `.DS_Store` は直接削除しない。
- リファクタリングでは数値処理と既存チェックポイント互換性を優先し、論文のモデル定義を新たに解釈して変更しない。
