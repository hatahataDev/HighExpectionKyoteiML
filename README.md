# HighExpectionKyoteiML / create_FVM

create_FVM.py をこのリポジトリ直下で動かせるようにした最小構成です。
`Simulation/create_FVM.py` がデータを読み込み、特徴量を生成して `data/df.csv` を出力します。

## ディレクトリ構成
```
HighExpectionKyoteiML/
  Simulation/
    create_FVM.py
  Data/
    ml_data/            # 入力データ（Gitには載せない）
    test_artifacts/     # プロファイル指定時の入力データ（任意）
  data/                 # 出力先（df.csv）
  docs/
    README.md
    USAGE.md
    DATA.md
  requirements.txt
```

## セットアップ
```
python -m venv .venv
source .venv/bin/activate  # Windowsの場合は .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 入力データの配置
既定では `Data/ml_data` を参照します。最低限、以下のファイルが必要です。

- `race_class_df_update.pickle`
- `last_info_df_update_v2.pickle`
- `race_results_df_update_v3.pickle`
- `shutuba_table_df_update_v2.pickle`

別場所に置きたい場合は環境変数で指定できます。

```
# 例: Data/ml_data を明示
export FVM_DATASET_DIR=/path/to/Data/ml_data

# 例: test_artifacts のプロファイルを使う場合
export FVM_ARTIFACT_ROOT=/path/to/Data/test_artifacts
export FVM_PROFILE=FVM
```

## 実行
```
python Simulation/create_FVM.py
```
成功すると `data/df.csv` が作成されます。

### df 生成の切り替え
既定では補正付きレースタイム（`correction_RT_preprocessing`）を使います。
補正なしに切り替える場合は環境変数を指定してください。

```
export FVM_USE_CORRECTION=0
```

## 補足
- データ容量が大きいため `Data/` と `data/` は Git 管理から除外する運用を想定しています。

詳細は `docs/` を参照してください。
