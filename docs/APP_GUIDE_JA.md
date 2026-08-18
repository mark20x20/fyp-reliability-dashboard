# アプリケーション操作ガイド（日本語）

対象: 開発者本人、指導教員デモ、UAT参加者  
作成日: 2026-08-18  
確認環境: Windows 11, conda 環境 `fyp`, Python 3.11

---

## 目次

1. [起動方法](#1-起動方法)
2. [画面構成の概要](#2-画面構成の概要)
3. [各ページの説明](#3-各ページの説明)
   - [Home（ホーム）](#31-home-ホーム)
   - [Run Analysis（分析実行）](#32-run-analysis-分析実行--engineerのみ)
   - [Metrics Table（メトリクス表）](#33-metrics-table-メトリクス表--engineerのみ)
   - [Statistics（統計グラフ）](#34-statistics-統計グラフ--engineerのみ)
   - [Risk Queue（リスクキュー）](#35-risk-queue-リスクキュー--reviewerのみ)
   - [Image Detail（画像詳細）](#36-image-detail-画像詳細--reviewerのみ)
   - [Upload & Analyse（アップロード分析）](#37-upload--analyse-アップロード分析--reviewerのみ)
   - [Export（エクスポート）](#38-export-エクスポート--両ロール)
4. [デモ用推奨シナリオ（5分版）](#4-デモ用推奨シナリオ5分版)
5. [デバッグガイド](#5-デバッグガイド)

---

## 1. 起動方法

### 手順

```powershell
# プロジェクトルートに移動
cd C:\APU\FYPpart2

# conda 環境をアクティベート
conda activate fyp

# アプリを起動
streamlit run app/streamlit_app.py
```

起動すると以下のように表示されます：

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

ブラウザが自動的に開かない場合は、`http://localhost:8501` を手動で開いてください。

### 停止方法

ターミナルで `Ctrl + C` を押してください。

### 起動しない場合の原因と対処

| 症状 | 原因 | 対処 |
|------|------|------|
| `ModuleNotFoundError: No module named 'streamlit'` | conda 環境が有効でない | `conda activate fyp` を先に実行 |
| `ModuleNotFoundError: No module named 'torch'` | 同上 | 同上 |
| `Error: cannot connect to database` | `db/reliability.db` が存在しない | `python scripts/init_db.py` を実行 |
| ポート 8501 が使用中 | 別のStreamlitが起動中 | `--server.port 8502` オプションを追加: `streamlit run app/streamlit_app.py --server.port 8502` |
| ブラウザが真っ白 | JavaScript読み込みエラー | F5 でリロード、または`Ctrl+Shift+I`でコンソール確認 |

---

## 2. 画面構成の概要

```
┌─────────────────────────────────────────────┐
│  サイドバー（全ページ共通）                        │
│  ┌──────────────┐                           │
│  │ Role: [engineer ▼]  ← ロール切り替え      │
│  │ Active run: [Run 11 ▼]  ← 分析対象選択   │
│  │ Model: resnet18 | N=30 | p=0.2           │
│  └──────────────┘                           │
│  ナビゲーションメニュー                          │
│  - Home                                     │
│  - [Engineer のみ] Run Analysis             │
│  - [Engineer のみ] Metrics Table            │
│  - [Engineer のみ] Statistics              │
│  - [Reviewer のみ] Risk Queue              │
│  - [Reviewer のみ] Image Detail            │
│  - [Reviewer のみ] Upload & Analyse        │
│  - Export（両ロール）                         │
└─────────────────────────────────────────────┘
```

**ロール切り替えの挙動**: サイドバーの `Role` ドロップダウンを変更すると、ナビゲーションメニューが即座に切り替わります。ページ間の移動は左のメニューをクリックします。

**現在のデータ**（2026-08-18時点）:
| run_id | データセット | 画像数 | 内容 |
|--------|------------|--------|------|
| 11 | imagenette | 3,925 | クリーン画像（in-distribution）|
| 12 | imagewoof  | 1,500 | クリーン画像（near-OOD）|
| 13 | dtd        | 1,500 | テクスチャ画像（far-OOD）|
| 14 | imagenette-c | 5,000 | 4種×5段階の画像劣化 |

---

## 3. 各ページの説明

### 3.1 Home（ホーム）

**何をする画面か**: 選択中の run の集計サマリーと、全 run 一覧を表示します。

**画面の要素**:
- **4つの指標タイル**: Total images（総画像数）、Accuracy（正答率）、Mean confidence（平均信頼度）、Mean cam_iou（平均説明一貫性）
- **All runs テーブル**: 全実行履歴。status が `completed` の run だけがサイドバーの Active run に現れます

**試すべき操作**:
1. サイドバーの `Active run` で `Run 11` を選択 → 上部タイルに run_id=11 の集計が表示される
2. `Run 14` に切り替え → corrupted run（imagenette-c）の集計に変わる

**正常な表示**: accuracy≈0.94（run_11）、mean cam_iou≈0.77

---

### 3.2 Run Analysis（分析実行）— Engineerのみ

**何をする画面か**: モデルとデータセットを指定して、MC-Dropout + Grad-CAM のバッチ分析を実行します。完了後、新しい run_id が生成されます。

**画面の要素**:
- **Model checkpoint**: DBに登録済みのモデル一覧。現在は `resnet18_dp0.2_acc95.1%` のみ
- **Dataset**: imagenette / imagewoof / dtd から選択。`imagenette-c`（劣化画像）はCLIのみ対応
- **Dataset type**: `id`（in-distribution）/ `near_ood` / `far_ood` — DBの `images.dataset_type` 列に書き込まれる値
- **Limit**: チェックを入れると最大画像数を指定できる（テスト用）
- **Current config**: `configs/config.yaml` から読み込んだパラメータ（変更不可、参照用）
- **プログレスバー**: 実行中に 1 画像ごとに更新される

**試すべき操作（テスト用、小規模）**:
1. `Limit number of images` にチェックを入れ、`Max images = 10`
2. Dataset = `imagenette`、Dataset type = `id`
3. `Run analysis` ボタンをクリック
4. プログレスバーが 10/10 になり、"Run XX completed" が表示されれば成功

**注意**: full run（3,925 枚）は GPU あり約 30 分かかります。デモでは実行しないことを推奨します。

---

### 3.3 Metrics Table（メトリクス表）— Engineerのみ

**何をする画面か**: 選択 run の全画像メトリクスを表形式で表示し、フィルタ・ソート・個別画像の詳細確認ができます。

#### フィルタの使い方（詳細）

サイドバーに以下のフィルタが表示されます：

| フィルタ | 意味 | 使い方の例 |
|----------|------|-----------|
| Dataset type | `id` / `near_ood` / `far_ood` / `corrupted` / `uploaded` | run_id=14なら`corrupted`を選択 |
| Correct | Correct only / Incorrect only / All | 誤分類画像だけ見たいときは `Incorrect only` |
| Risk group | hidden_risk など4種 | `hidden_risk`を選択して「高信頼・不安定」画像を絞る |
| Confidence | スライダー（0〜1） | 0.95以上だけ見たい場合は左端を0.95に動かす |
| cam_iou_mean | スライダー（0〜1） | 右端を0.60にすると説明不安定な画像が絞れる |
| Corruption type | gaussian_noise など（run_id=14のみ） | 劣化種別ごとに比較できる |
| Corruption severity | 1〜5（run_id=14のみ） | 劣化強度ごとに比較できる |
| Reset filters ボタン | 全フィルタを初期化 | フィルタが絡まったときに使う |

**Summary bar**: フィルタ後の件数・全件数・正答率・平均 cam_iou が上部に表示されます。

**行クリックで詳細確認**:
1. 表の行をクリックすると、その行が選択（ハイライト）される
2. 下部に `Image Detail` エキスパンダーが展開され、元画像・メトリクス・B3との差分・Grad-CAMのmean/variability mapが表示される

**CSV エクスポート**: 表の右下の `Download CSV` ボタンでフィルタ後のデータを取得できます。

---

### 3.4 Statistics（統計グラフ）— Engineerのみ

**何をする画面か**: 説明安定性の統計グラフと、ベースラインB1/B2/B3との比較パネルを表示します。研究の中心的な主張を視覚的に確認するためのページです。

#### ベースライン参照パネル（B1 / B2 / B3）の読み方

| ベースライン | 意味 | 解釈 |
|------------|------|------|
| **B1 (upper)** | ドロップアウトOFFでの同一画像の2回計測 | 理論上の最大値（cam_corr, IoU ともに 1.000）。実装確認用 |
| **B2 (lower)** | ランダムマップとの比較 | 「偶然より悪い」下限。cam_iou ≈ 0.11 |
| **B3 (cross_image)** | 同クラス別画像との比較 | **解釈の基準点**。cam_iou ≈ 0.36。本実験の値がB3より高ければ「説明が画像に固有」と言える |

#### 4つのタブの内容

**タブ1: Correlation heatmap（E2）**  
全メトリクス間のピアソン相関行列。確認すべき点:
- `entropy` と `cam_iou_mean` が負の相関 → 予測不確実性が高いほど説明も不安定
- `confidence` と `cam_iou_mean` が正の相関 → ただし **相関は弱い**（主張の前提）

**タブ2: Confidence vs IoU**  
横軸=confidence、縦軸=cam_iou_mean。色=risk_group。  
注目点: **左上の領域**（confidence高・cam_iou低）= hidden_risk。B3の水平線より下は「クロスイメージ基準を下回る不安定」。

**タブ3: E3 — Confidence bins（中心的な結果）**  
`pred_agreement==1.0`（30回全パスで同じクラスを予測）のサブセットを confidence bin で層別したボックスプロット。  
確認すべき点: **最高confidence（0.99〜1.00）のbin でも cam_iou_mean の分布が広い** → 予測安定性だけでは説明安定性を説明できない

**タブ4: E4 — Quadrant（中心的な結果）**  
横軸=entropy（予測不確実性）、縦軸=1-cam_iou_mean（説明不安定性）、破線=中央値。  
左下（低entropy・低instability） = stable。**左上（低entropy・高instability）= hidden_risk** — これが研究の核心。

**Descriptive statistics（記述統計）**: ページ末尾のエキスパンダーを開くと全メトリクスの min/max/mean/std が確認できます。

---

### 3.5 Risk Queue（リスクキュー）— Reviewerのみ

**何をする画面か**: 「高信頼度でありながら説明が不安定」な画像を列挙し、Reviewerが優先的に確認すべき画像を提示します。

#### TH_CONF と TH_IOU のスライダーの効果

| スライダー | 意味 | デフォルト | 動かすと |
|-----------|------|-----------|---------|
| **TH_CONF** | 表示する画像の最小信頼度 | 0.90 | 上げると件数が減る（より高信頼の画像のみ） |
| **TH_IOU** | 表示する画像の最大 cam_iou_mean | B3値（≈0.361） | 上げると件数が増える（より多くの「不安定」画像を含む） |

**表示ルール**: `confidence >= TH_CONF` かつ `cam_iou_mean <= TH_IOU` を満たす画像を、confidence 降順（最も「高信頼なのに不安定」な順）で表示します。

**run_id=11 での操作例**:
- デフォルト（TH_CONF=0.90, TH_IOU=B3=0.361）→ 2件（わずか）
- TH_IOU を 0.60 に上げる → 約60件表示（hidden_risk かつ cam_iou < 0.60）
- TH_IOU を 0.70 に上げる → さらに多くが表示

**行を選択して Image Detail へ**:
1. 表の行をクリック → 行がハイライト + 「Open Image Detail」ボタンが現れる
2. ボタンをクリック → Image Detail ページへ遷移（選択した image_id が自動で引き継がれる）

---

### 3.6 Image Detail（画像詳細）— Reviewerのみ

**何をする画面か**: 1枚の画像について、元画像・メトリクス・Grad-CAMマップ・解釈テキスト・レビュー決定フォームを一覧表示します。

#### image_id の指定方法

2通りあります:
1. **Risk Queue から遷移する**（推奨）: Risk Queue で行を選択 → Open Image Detail
2. **直接入力**: ページ上部の `Enter image_id` 欄に数値を入力して `Load` ボタン

#### 各指標の読み方

| 指標 | 高い値の意味 | 低い値の意味 |
|------|------------|------------|
| **Confidence** | モデルが確信を持って予測 | 予測に迷いがある |
| **Entropy** | 予測の不確実性が高い（30パスで分布が広い） | 不確実性が低い（分布が集中） |
| **pred_variance** | 予測確率の分散が大きい | 安定した予測 |
| **pred_agreement** | 1.0 = 30回全パスで同クラスを予測（完全一致） | 低いほどパスごとにクラスが変わる |
| **cam_iou_mean** | Grad-CAMマップが30パスで一貫して同じ領域を見ている | **低いほど説明が不安定**。B3（≈0.361）を下回ると「クロスイメージ基準以下」 |
| **cam_corr_mean** | Grad-CAMマップ間の相関が高い（形状が似ている） | マップの形状がパス間でバラバラ |

**B3との比較（メトリクスパネルの±値）**:
- `cam_iou_mean` の右に表示される `+0.166 vs B3` のような数値は、この画像の値とB3（クロスイメージ基準）との差分です
- 正の値 = B3より良い（説明が画像に固有）
- 負の値 = B3より悪い（説明がクロスイメージ程度以下）

**解釈テキスト**: ルールベースの自動解釈が表示されます。`REVIEW PRIORITY` は研究の「hidden risk」条件（confidence ≥ 0.90 かつ cam_iou ≤ B3）に合致する画像に表示されます。

**Grad-CAMマップ**:
- **6 representative maps**: 30パスから等間隔に選んだ6枚のヒートマップ。マップ間の見た目の違いが説明不安定性の直感的な確認になります
- **Mean map**: 30パスの平均Grad-CAM（どの領域をモデルが「重要」と判断しているか）
- **Variability map**: パス間のピクセル単位の標準偏差（明るいほど変動が大きい）

**レビュー決定フォーム**:
- `accept`: この画像の予測は信頼できると判断
- `needs_review`: 専門家の追加確認が必要
- `reject`: この予測は信頼すべきでない
- コメント欄は任意。Submit後はページが自動リロードされ、レビューが一覧に追加されます

**Clear ボタン**: 右上の `Clear` を押すと image_id がリセットされ、入力フォームに戻ります。

---

### 3.7 Upload & Analyse（アップロード分析）— Reviewerのみ

**何をする画面か**: 任意の画像ファイルをアップロードして、MC-Dropout + Grad-CAM の分析をリアルタイムで実行します。分析結果は `dataset_type='uploaded'` としてDBに保存されます。

**操作手順**:
1. モデルを選択（現在は resnet18 のみ）
2. `Choose an image file` でファイルを選択（対応形式: JPG, PNG, BMP, GIF, TIFF, WEBP）
3. プレビューと画像情報が表示される
4. `Analyse image` ボタンをクリック
5. スピナーが表示される（N=30 のGPU推論、約 5〜15秒）
6. 分析結果（メトリクス・Grad-CAMマップ）が表示される
7. `Open in Image Detail` ボタンで詳細ページへ移動できる

**非画像ファイルを選択した場合**: `The uploaded file is not a valid image...` というエラーが赤く表示され、分析は実行されません（TC22 で検証済み）。

**注意**: アップロード画像はベースライン(B1/B2/B3)なしで分析されるため、cam_iou の絶対的な解釈が難しくなります。「正解クラス」も不明なため、correct列はNULLになります。

---

### 3.8 Export（エクスポート）— 両ロール

**何をする画面か**: 分析結果を CSV や ZIP ファイルとしてダウンロードします。

| セクション | 内容 | ファイル名 |
|-----------|------|----------|
| **Metrics CSV** | 選択 run の全メトリクス（1行=1画像） | `reliability_runXX.csv` |
| **Reviews CSV** | レビュー決定の記録（全 run または選択 run） | `reviews_runXX.csv` |
| **Figure bundle** | Statistics ページの PNG 図一式（ZIP） | `figures_runXX.zip` |

Figure bundle は Statistics ページを一度開いた後でないとファイルが生成されません。

---

## 4. デモ用推奨シナリオ（5分版）

### テーマ
**「高信頼でも説明が不安定なサンプルは存在し、それは信頼度だけでは検出できない」**

### 使用データ
- run_id = **11**（imagenette クリーン画像 3,925 枚、B3 = 0.361）
- 比較画像: image_id = **4446**（stable）vs. **4253**（hidden_risk）
  - 両方ともゴルフボール（n03445777）、confidence = 1.000、正解
  - 4446: cam_iou_mean = **0.986**（非常に一貫した説明）
  - 4253: cam_iou_mean = **0.527**（比較的不安定な説明）

### 画面遷移と説明文

**ステップ 1（30秒）: ホーム → run_id=11 の概要**
```
サイドバー: Role=engineer, Active run=Run 11
```
- 「run_id=11 は Imagenette の全 3,925 枚を分析した結果です。
  Accuracy ≈ 94%、Mean cam_iou ≈ 0.77 で、説明の一貫性はB3（0.36）を大きく上回っています。」

**ステップ 2（90秒）: Statistics ページ → E3, E4 タブ**
```
左メニュー: Statistics
タブ: E4 — Quadrant
```
- 「このグラフの縦軸は説明不安定性（1-cam_iou_mean）、横軸は予測不確実性（entropy）です。」
- 「破線は中央値。**左上の領域**（低 entropy・高 instability）が hidden_risk グループです。」
- 「この象限の画像は、モデルが確信を持って予測しているにもかかわらず、
  Grad-CAM マップがパスごとに異なる領域を指しています。」

```
タブ: E3 — Confidence bins
```
- 「pred_agreement = 1.0（全30パスで同クラス）のサブセットに絞っています。」
- 「**最高信頼度のビン（0.99〜1.00）でも cam_iou_mean の分布が広い**ことが分かります。」
- 「信頼度だけでは説明の一貫性を予測できないことを示しています。」

**ステップ 3（60秒）: Role を Reviewer に切り替え → Risk Queue**
```
サイドバー: Role=reviewer
左メニュー: Risk Queue
TH_IOU スライダーを 0.60 に上げる
```
- 「TH_IOU を B3（0.36）から 0.60 に上げると、
  confidence ≥ 0.90 かつ cam_iou ≤ 0.60 の画像が約60件表示されます。」
- 「image_id=4253 を探してクリックし、Open Image Detail をクリックします。」

**ステップ 4（90秒）: Image Detail → 2枚の比較**

image_id=4253（hidden_risk）を表示:
- 「confidence = 1.0000（完全確信）、pred_agreement = 1.0（全パス一致）。
  モデルの予測は完全に安定しています。」
- 「しかし cam_iou_mean = 0.527。6枚の Grad-CAM マップを見ると、
  パスごとにハイライト領域が異なっています。」
- 「これが hidden_risk の具体例です。正解かつ自信満々だが、
  根拠（どこを見ているか）は変動している。」

次に `Clear` → image_id = 4446 を入力:
- 「こちらも同じゴルフボール、confidence = 1.000 で正解。」
- 「しかし cam_iou_mean = 0.986。6枚のマップはほぼ同一で、
  一貫してゴルフボールの輪郭を捉えています。」
- 「この差が本研究の測定対象です。信頼度では区別できない。」

**ステップ 5（30秒）: Export → 図のダウンロード**
```
左メニュー: Export
```
- 「分析結果の CSV と Statistics の図を ZIP でダウンロードできます。
  レポートの図や数値はここから取得します。」

---

## 5. デバッグガイド

### 5.1 エラーが出たとき、どこを見るか

**優先順位**:
1. **Streamlit の赤いエラーボックス（画面上）** — 最も直接的。スタックトレースが表示される
2. **起動したターミナル** — Python の print 出力と logging の WARNING/ERROR が流れる
3. **ブラウザのコンソール（F12 → Console）** — JavaScript エラーや HTTP エラーが出るが、通常は Python 側が原因
4. **`logs/` ディレクトリ** — `scripts/run_logged.ps1` 経由で実行した場合のみログが保存される

### 5.2 Streamlit 特有の詰まりどころ

#### キャッシュが古い
症状: データを更新したはずなのに画面に反映されない

対処:
```
方法1: ブラウザ右上の「⋮」メニュー → 「Clear cache」
方法2: 画面右上の「…」(ハンバーガー) → Settings → Clear cache
方法3: ターミナルで Ctrl+C → 再起動
```

コード側でのキャッシュクリア（レビュー保存後など）は既に実装済み（`_load_detail.clear()` が呼ばれます）。

#### session_state が残る
症状: ロール切り替えや run 切り替えが画面に反映されない

対処: ブラウザをリロード（F5）。これで session_state がリセットされます。
または「⋮」メニュー → `Rerun` を選択。

#### rerun のタイミング問題
症状: ボタンを押しても反応しない、または2回押さないと動かない

原因: `st.form_submit_button` や `on_select` イベントの後、コードが `st.rerun()` を呼ぶまでに別の widget 操作が挟まると、rerun が二重に発生することがあります。

対処: ページを一度リロードしてから操作し直す。

### 5.3 よくある画面症状と確認順序

#### 「画面が真っ白」
1. ターミナルを確認 → SyntaxError や ImportError が出ていないか
2. ブラウザを F5 でリロード
3. `streamlit run app/streamlit_app.py` を再実行

#### 「データが表示されない」
1. サイドバーの `Active run` が選択されているか確認（「No completed runs yet」と出ていないか）
2. DB に対象の run が存在するか確認（下記のSQLコマンドを参照）
3. Metrics Table なら「Reset filters」ボタンを押してフィルタをリセット

#### 「図が出ない（Statisticsページ）」
1. `outputs/figures/run_{run_id}/` ディレクトリに PNG があるか確認:
   ```powershell
   ls C:\APU\FYPpart2\outputs\figures\run_11\
   ```
2. なければ Statistics ページを開くと自動生成される（初回は5〜10秒かかる）
3. それでも出ない場合は、Statistics ページ内で Plotly のインタラクティブ版がフォールバックとして表示されます

#### 「保存したレビューが消える」
レビューはページリロードで消えません。以下を確認:
1. Submit ボタンを押した後に「Review saved (review_id=X)」の緑メッセージが出たか
2. 出た場合は DB に保存されています。以下のSQLで確認:
   ```powershell
   cd C:\APU\FYPpart2
   "C:\Users\Masaki\miniconda3\envs\fyp\python.exe" -c "
   import sqlite3
   conn = sqlite3.connect('db/reliability.db')
   for row in conn.execute('SELECT * FROM reviews ORDER BY created_at DESC LIMIT 10'):
       print(dict(row))
   "
   ```

### 5.4 データベースの中身を直接確認するコマンド

以下はコピペで実行できます。`cd C:\APU\FYPpart2` を先に実行してください。

```powershell
# 全 run の一覧と画像数
"C:\Users\Masaki\miniconda3\envs\fyp\python.exe" -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('db/reliability.db')
print(pd.read_sql_query('''
    SELECT r.run_id, r.dataset_name, r.status,
           COUNT(i.image_id) as n_images, r.started_at
    FROM runs r
    LEFT JOIN images i ON i.run_id = r.run_id
    GROUP BY r.run_id ORDER BY r.run_id
''', conn).to_string(index=False))
"
```

```powershell
# run_id=11 の risk_group 分布
"C:\Users\Masaki\miniconda3\envs\fyp\python.exe" -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('db/reliability.db')
print(pd.read_sql_query('''
    SELECT r.risk_group, COUNT(*) as n,
           ROUND(AVG(p.correct),3) as accuracy,
           ROUND(AVG(p.confidence),4) as mean_conf,
           ROUND(AVG(e.cam_iou_mean),4) as mean_iou
    FROM risk_flags r
    JOIN images i ON i.image_id = r.image_id
    JOIN predictions p ON p.image_id = r.image_id
    JOIN explanations e ON e.image_id = r.image_id
    WHERE i.run_id = 11
    GROUP BY r.risk_group
''', conn).to_string(index=False))
"
```

```powershell
# 特定の image_id の全情報
"C:\Users\Masaki\miniconda3\envs\fyp\python.exe" -c "
import sqlite3
conn = sqlite3.connect('db/reliability.db')
conn.row_factory = sqlite3.Row
IMAGE_ID = 4253
for tbl, col in [('images','image_id'), ('predictions','image_id'), ('explanations','image_id'), ('risk_flags','image_id')]:
    row = conn.execute(f'SELECT * FROM {tbl} WHERE {col}=?', (IMAGE_ID,)).fetchone()
    if row: print(f'[{tbl}]', dict(row))
"
```

```powershell
# レビュー一覧
"C:\Users\Masaki\miniconda3\envs\fyp\python.exe" -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('db/reliability.db')
print(pd.read_sql_query('''
    SELECT rv.review_id, rv.image_id, rv.decision, rv.comment, rv.created_at
    FROM reviews rv ORDER BY rv.created_at DESC LIMIT 20
''', conn).to_string(index=False))
"
```

```powershell
# baselines (B1/B2/B3) の確認
"C:\Users\Masaki\miniconda3\envs\fyp\python.exe" -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('db/reliability.db')
print(pd.read_sql_query('SELECT * FROM baselines WHERE run_id=11', conn).to_string(index=False))
"
```

### 5.5 キャッシュを消して初期状態に戻す

#### Streamlit キャッシュのみリセット（DB は残る）
```
ブラウザ → 右上メニュー → Settings → Clear cache
```
または:
```powershell
# Streamlit の内部キャッシュディレクトリを削除
Remove-Item -Recurse -Force "C:\APU\FYPpart2\.streamlit\cache" -ErrorAction SilentlyContinue
```

#### テストデータだけ確認したい（本番DBに触らない）
```powershell
# 別のDB_PATHを指定して起動
$env:DATABASE_PATH = "db/test_check.db"
streamlit run app/streamlit_app.py
```
（`configs/config.yaml` を変更せずに別DBで動作を確認できます）

#### 全テストを再実行して動作を確認
```powershell
cd C:\APU\FYPpart2
conda activate fyp
python -m pytest tests/ -v 2>&1 | tail -30
```
27 passed が確認できれば、コアロジックに問題はありません。

### 5.6 自動検証スクリプト

本番DB・全メソッドの動作確認を一括で行うスクリプトが `scripts/verify_app.py` にあります:

```powershell
cd C:\APU\FYPpart2
"C:\Users\Masaki\miniconda3\envs\fyp\python.exe" scripts/verify_app.py
```

35項目のチェックが完了し「All checks passed.」が表示されれば正常です。
