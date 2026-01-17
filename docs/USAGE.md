# 使用方法

## 1. 依存関係のインストール
```
python -m venv .venv
source .venv/bin/activate  # Windowsの場合は .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. 入力データの配置
既定の参照先は `Data/ml_data` です。下記のファイルが必要です。

- `race_class_df_update.pickle`
- `last_info_df_update_v2.pickle`
- `race_results_df_update_v3.pickle`
- `shutuba_table_df_update_v2.pickle`

別の場所に置きたい場合は環境変数で指定できます。

```
export FVM_DATASET_DIR=/absolute/path/to/Data/ml_data
```

`test_artifacts` 配下のプロファイルを使う場合は次の通りです。

```
export FVM_ARTIFACT_ROOT=/absolute/path/to/Data/test_artifacts
export FVM_PROFILE=FVM
```

## 3. 実行
```
python Simulation/create_FVM.py
```

成功すると `data/df.csv` が生成されます。

## 4. 補正の切り替え
既定では補正付きレースタイム（`correction_RT_preprocessing`）を使います。
補正なしに切り替える場合は環境変数を指定してください。

```
export FVM_USE_CORRECTION=0
```

## 5. 出力
- `data/df.csv`: 特徴量付与済みのCSV
