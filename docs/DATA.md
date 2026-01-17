# 入力データの役割

`Simulation/create_FVM.py` は以下の pickle を読み込みます。

## 必須ファイル
- `race_class_df_update.pickle`
  - レース区分やグレードなどのマスタ的情報の想定。
  - ※現行の `create_FVM.py` では読み込みのみで直接利用していません。

- `last_info_df_update_v2.pickle`
  - レース直前情報（気象・展示・選手情報など）。
  - `preprocessing` / `correction_RT_preprocessing` で使用されます。

- `race_results_df_update_v3.pickle`
  - レース結果（着順、枠、タイムなど）。
  - レースタイムの補正やマージに使用されます。

- `shutuba_table_df_update_v2.pickle`
  - 出走表（枠や選手、機体情報など）。
  - `last_info_df_update_v2` と結合して特徴量の土台を作ります。

## 参照される主な列
スクリプト内で参照される列は以下の通りです（不足するとエラーになります）。

- `last_info_df_update_v2` / `shutuba_table_df_update_v2` 由来
  - `ET`, `tilt`, `EST`, `ESC`, `date`, `place`, `race_no`, `weather`,
    `air_t`, `wind_d`, `wind_v`, `water_t`, `wave_h`, `枠`, `F`, `L`, `age`,
    `weight`, `racer_no`, `racer_lank`, `sibu`, `born`,
    `zwin_r_1`, `zwin_r_2`, `zwin_r_3`, `twin_r_1`, `twin_r_2`, `twin_r_3`,
    `moter_no`, `mwin_r_2`, `mwin_r_3`, `boat_no`, `bwin_r_2`, `bwin_r_3`,
    `piston`, `ring`, `electric`, `carburetor`, `cylinder`, `shafts`,
    `gears`, `carrier`, `propera`

- `race_results_df_update_v3` 由来
  - `着`, `枠`, `TSC`, `決まり手`, `PST`, `レースタイム`

## 置き場所
既定では `Data/ml_data` を参照します。別パスに置く場合は
`FVM_DATASET_DIR` で明示してください。

