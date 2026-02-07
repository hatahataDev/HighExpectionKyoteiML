# create_FVM.py で参照する列（入力データ別）

このドキュメントは `Simulation/create_FVM.py` 内で **実際に参照される列** を整理したものです。
列の由来（どのファイルに入っているか）はデータ生成側の仕様に依存するため、
`last_info_df_update_v2` と `shutuba_table_df_update_v2` は **結合後に参照される列** として記載しています。

## 共通前提（インデックス）
- `last_info_df_update_v2` / `shutuba_table_df_update_v2` / `race_results_df_update_v3` は **index が race_id** である前提。
- `last_info_df_update_v2` と `shutuba_table_df_update_v2` は `race_id + merge_id`（各レース内の並び順）で結合。

---

## race_class_df_update.pickle
`create_FVM.py` では `race_id`（index）で参照し、`RaceDistance` から `race_distance_m` を生成します。

| 列名 | 使われ方 | 補足 |
|---|---|---|
| `RaceDistance` | `race_distance_m` の元データ | 文字列混在を数値化し、異常値（例: 5/50）は欠損扱い |
| `RaceClass` | 距離欠損時の補完キー | `RaceType` 等と組み合わせて最頻値補完 |
| `RaceType` | 距離欠損時の補完キー | 上記で埋まらない場合のフォールバックにも使用 |
| `Steady_board` | 距離欠損時の補完キー | 補完用の文脈情報 |
| `RaceType_bi` | 距離欠損時の補完キー | 補完用の文脈情報 |

---

## last_info_df_update_v2.pickle + shutuba_table_df_update_v2.pickle
`preprocessing` / `correction_RT_preprocessing` 内で結合後に参照される列。

| 列名 | 使われ方 | 軽い説明 |
|---|---|---|
| `ET` | 特徴量生成でそのまま利用 | 展示タイム（想定） |
| `tilt` | 特徴量生成でそのまま利用 | チルト角（想定） |
| `EST` | 特徴量生成でそのまま利用 | 展示スタートタイム（想定） |
| `ESC` | `int` 変換して利用 | 展示コース（想定） |
| `date` | そのまま利用 | レース日付（YYYY-MM-DD想定） |
| `place` | `str` 変換して利用 | 競艇場コード（想定） |
| `race_no` | `str` 変換して利用 | レース番号（想定） |
| `weather` | 雨/雪を 1、その他 0 に変換 | 天候 |
| `air_t` | そのまま利用 | 気温（想定） |
| `wind_d` | そのまま利用 | 風向（想定） |
| `wind_v` | そのまま利用 | 風速（想定） |
| `water_t` | そのまま利用 | 水温（想定） |
| `wave_h` | そのまま利用 | 波高（想定） |
| `枠` | `race_results` と結合キーに使用 | 枠番 |
| `F` | そのまま利用 | フライング回数（想定） |
| `L` | そのまま利用 | 遅れ（L）回数（想定） |
| `age` | `int` 変換して利用 | 年齢 |
| `weight` | `float` 変換して利用 | 体重 |
| `racer_no` | そのまま利用 | 選手番号 |
| `racer_lank` | ランク数値化・A/B判定に使用 | 選手級別（A1/A2/B1/B2） |
| `sibu` | そのまま利用 | 支部（想定） |
| `born` | そのまま利用 | 出身地/出生地（想定） |
| `zwin_r_1` | 標準化・他特徴量に使用 | 全国1着率（想定） |
| `zwin_r_2` | 標準化に使用 | 全国2着率（想定） |
| `zwin_r_3` | 標準化に使用 | 全国3着率（想定） |
| `twin_r_1` | 0 のとき `zwin_r_1` で補正 | 当地1着率（想定） |
| `twin_r_2` | そのまま利用 | 当地2着率（想定） |
| `twin_r_3` | そのまま利用 | 当地3着率（想定） |
| `moter_no` | そのまま利用 | モーター番号（想定） |
| `mwin_r_2` | `float` 変換して利用 | モーター2連率（想定） |
| `mwin_r_3` | `float` 変換して利用 | モーター3連率（想定） |
| `boat_no` | そのまま利用 | ボート番号（想定） |
| `bwin_r_2` | `float` 変換して利用 | ボート2連率（想定） |
| `bwin_r_3` | `float` 変換して利用 | ボート3連率（想定） |
| `piston` | そのまま利用 | 部品情報（ピストン） |
| `ring` | そのまま利用 | 部品情報（リング） |
| `electric` | そのまま利用 | 部品情報（電気） |
| `carburetor` | そのまま利用 | 部品情報（キャブ） |
| `cylinder` | そのまま利用 | 部品情報（シリンダー） |
| `shafts` | そのまま利用 | 部品情報（シャフト） |
| `gears` | そのまま利用 | 部品情報（ギア） |
| `carrier` | そのまま利用 | 部品情報（キャリア） |
| `propera` | そのまま利用 | 部品情報（プロペラ） |

補足:
- `racer_lank` は A/B のカテゴリに変換し、レース内の A 級人数カウントに使われます。
- `ET`, `zwin_r_*`, `twin_r_1`, `mwin_r_2`, `bwin_r_2` はレース内で標準化特徴量を生成します。

---

## race_results_df_update_v3.pickle
`race_results_df_fix` / `preprocessing` / `correction_RT_preprocessing` / `calc_recetime_statistics` 内で参照されます。

| 列名 | 使われ方 | 軽い説明 |
|---|---|---|
| `着` | `1〜6` 以外は 7 に変換 | 着順（全角数字） |
| `枠` | `str` 変換し結合キーに使用 | 枠番 |
| `TSC` | 枠再割当・統計量計算に使用 | 侵入コース/枠情報（想定） |
| `決まり手` | そのまま保持 | 決まり手 |
| `PST` | `race_results_df_fix` で保持 | 進入スタート関連（想定） |
| `レースタイム` | 補正・統計量計算に使用 | レースタイム（文字列） |

補足:
- `race_results_df_fix` で `TSC` を枠扱いにして再計算し、`PST` を保持したままマージします。
- `レースタイム` は `'` や `"` を除去して `float` 化します。
