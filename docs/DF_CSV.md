# df.csv の内容

`data/df.csv` は `Simulation/create_FVM.py` が生成する特徴量付きCSVです。
データが存在しないため **スクリプトの処理内容から列と加工内容を分析** しています。

## 生成条件
- 既定では補正モード（`FVM_USE_CORRECTION=1`）で生成されます。
- `FVM_USE_CORRECTION=0` の場合、補正列と一部の補正処理が入らない点に注意してください。

## 行の粒度とインデックス
- **1行 = 1レース内の1選手（1艇）** です。
- index は `race_id`（同一レース内で6行程度が並ぶ想定）。
- `to_csv()` で **index を保存** しているため、CSVの先頭列に `race_id` が入ります（ヘッダ名は空欄）。

## 列一覧（df.csv に最終的に残る列）

### 1) 入力由来の列（型変換のみ or そのまま）
`last_info_df_update_v2.pickle` と `shutuba_table_df_update_v2.pickle` から来る列です。
※ `racer_lank` は後段で削除されるため **最終出力には残りません**。

- レース/環境
  - `ET`, `tilt`, `EST`, `ESC`, `date`, `place`, `race_no`, `weather`
  - `air_t`, `wind_d`, `wind_v`, `water_t`, `wave_h`
- 枠・選手情報
  - `枠`, `F`, `L`, `age`, `weight`, `racer_no`, `sibu`, `born`
- 勝率・成績系
  - `zwin_r_1`, `zwin_r_2`, `zwin_r_3`
  - `twin_r_1`, `twin_r_2`, `twin_r_3`
- 機体・部品系
  - `moter_no`, `mwin_r_2`, `mwin_r_3`
  - `boat_no`, `bwin_r_2`, `bwin_r_3`
  - `piston`, `ring`, `electric`, `carburetor`, `cylinder`, `shafts`, `gears`, `carrier`, `propera`

### 1.5) レースマスタ由来の列
`race_class_df_update.pickle` から `race_id` で付与されます。

- `race_distance_m`
  - `RaceDistance` を数値化した距離（m）。
  - 欠損時は `RaceClass` / `RaceType` / `Steady_board` / `RaceType_bi` の組み合わせ、次いで `RaceType` 単位の最頻値で補完。
  - `5` や `50` などの異常値は欠損扱い。

### 2) レース結果由来の列
`race_results_df_update_v3.pickle`（補正モードの場合は `race_results_df_fix` を通したもの）から来る列です。

- `着`（全角「１〜６」は int 化、それ以外は 7）
- `TSC`（`race_results_df_fix` により実質的に再割当されたコース情報）
- `決まり手`
- `PST`
- `レースタイム`（文字列のまま保持）

### 3) 補正モードのみ追加される列
`FVM_USE_CORRECTION=1` のときのみ追加されます。

- `レースタイム_refix`
  - `レースタイム` から `'` と `"` を除去して数値化した列（float）。
  - 5/6着タイムが欠損の場合、レース内の平均差分から補完されます。

### 4) 追加特徴量（`add_features` で生成）
- `weather`
  - 雨/雪なら 1、それ以外は 0 に置換（元の文字列は上書き）
- `racer_lank_int`
  - `racer_lank` を数値化（A1=3, A2=2, B1=1, B2=0）
- `lank_counts`
  - 同一 `race_id` 内の A級人数（`racer_lank` の A/B 判定を集計）
- 標準化特徴量（z-score、レース内平均との差/標準偏差）
  - `ET_std`, `zwin_r_1_std`, `zwin_r_2_std`, `zwin_r_3_std`
  - `twin_r_1_std`, `mwin_r_2_std`, `bwin_r_2_std`
  - レース内で標準偏差が 0 の場合は `NaN` になります。

### 5) 補正モードのみ行われる値の置換
- `twin_r_1` が 0 の場合、`zwin_r_1` を代入します（`FVM_USE_CORRECTION=1` のみ）。

## 型変換・注意点
- `ESC` は `int`
- `place`, `race_no` は `str`
- `mwin_r_2`, `mwin_r_3`, `bwin_r_2`, `bwin_r_3` は `float`
- `age` は `int`, `weight` は `float`
- `レースタイム_refix` は `float`（補正モードのみ）

## 参照先
- 入力データの列一覧と詳細は `docs/DATA_COLUMNS.md` を参照してください。
