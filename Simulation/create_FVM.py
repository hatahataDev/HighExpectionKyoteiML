import numpy as np
import pandas as pd
import random
import datetime
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _artifact_root() -> Path:
    root_env = os.environ.get("FVM_ARTIFACT_ROOT")
    if root_env:
        root_path = Path(root_env).expanduser().resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"FVM_ARTIFACT_ROOT で指定されたディレクトリが存在しません: {root_path}")
        return root_path
    return (BASE_DIR / "Data" / "test_artifacts").resolve()


def _resolve_profile_dir(profile: str) -> Path:
    profile = profile.strip().strip("/\\")
    if not profile:
        raise ValueError("FVM_PROFILE に空文字列は指定できません。")
    profile_path = _artifact_root() / Path(profile)
    if not profile_path.exists():
        raise FileNotFoundError(f"FVM_PROFILE で指定されたディレクトリが存在しません: {profile_path}")
    return profile_path.resolve()


def resolve_dataset_dir() -> Path:
    dataset_dir_env = os.environ.get("FVM_DATASET_DIR")
    if dataset_dir_env:
        dataset_dir = Path(dataset_dir_env).expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"FVM_DATASET_DIR で指定されたディレクトリが存在しません: {dataset_dir}")
        return dataset_dir

    profile = os.environ.get("FVM_PROFILE")
    if profile:
        return _resolve_profile_dir(profile)

    default_dir = (BASE_DIR / "Data" / "ml_data").resolve()
    if not default_dir.exists():
        raise FileNotFoundError(f"既定データディレクトリが存在しません: {default_dir}")
    return default_dir


DATA_DIR = resolve_dataset_dir()
dir_name = str(DATA_DIR) + "/"
print("create_FVM.py の入力データディレクトリ: {}".format(DATA_DIR))

race_class_df = pd.read_pickle(dir_name + "race_class_df_update.pickle") 
last_info_df = pd.read_pickle(dir_name + 'last_info_df_update_v2.pickle')#欠場レースのデータなし．
race_results_df = pd.read_pickle(dir_name+'race_results_df_update_v3.pickle')

shutuba_tables_df = pd.read_pickle(dir_name+'shutuba_table_df_update_v2.pickle')#欠場レース有のレースもデータ有り
#last_info_dfとshutuba_tables_dfとrace_results_dfの共通レースは，1752031(2022/09/07)


def race_results_df_fix(race_results_df_v2):#入力にはrace_results_df_v2.pickle
    race_results_df_v2 = race_results_df_v2[(race_results_df_v2["TSC"] == "1") |
                        (race_results_df_v2["TSC"] == "2") |
                        (race_results_df_v2["TSC"] == "3") |
                        (race_results_df_v2["TSC"] == "4") |
                        (race_results_df_v2["TSC"] == "5") |
                        (race_results_df_v2["TSC"] == "6") ]#謎のやつを除去する
    race_results_df_v2["race_id"] = race_results_df_v2.index
    race_corse_results_df = race_results_df_v2[["TSC", "race_id", "PST"]] #PSTの追加(23/1/2)
    race_corse_results_df = race_corse_results_df.rename(columns = {"TSC":"TSC枠"}) #TSCを枠と見なして、下のcomcount()+1を真のコースとしてマージする
    race_corse_results_df["TSC"] = race_corse_results_df.groupby(level=0).cumcount()+1 #TSCの枠を作る
    race_results_df_v2 = race_results_df_v2.drop(columns = ["TSC", "PST"]) #PSTの追加(23/1/2)
    race_corse_results_df["TSC枠"] = race_corse_results_df["TSC枠"].astype(int)
    race_corse_results_df = race_corse_results_df.rename(columns = {"TSC枠" : "枠"})
    return_df = pd.merge(race_corse_results_df, race_results_df_v2, on=['race_id','枠']) 
    return_df.index  = return_df["race_id"]
    return_df = return_df.rename_axis(None) #index名を削除する
    return return_df


def preprocessing(last_info_df, shutuba_tables_df, race_results_df):#
    '''
    water_t_statistic_values = np.array(water_t_statistic_values)
    place_statistic_values = np.array(place_statistic_values)
    TSC_statistic_values = np.array(TSC_statistic_values)
    これらの統計量を出す際に使用．
    これらの統計量は，5,6着補正をしない状態で算出されている．
    ->correction_RT_preprocessingは5,6着補正入り．
    '''
    #学習データのレースIDを取得
    lace_id_list = last_info_df.index.unique()

    #レースIDを持つ出馬表データを取得
    shutuba_tables_df_ = shutuba_tables_df[shutuba_tables_df.index.isin(lace_id_list)]
    last_info_df_ = last_info_df[last_info_df.index.isin(shutuba_tables_df_.index)]
    race_results_df_ = race_results_df[['着','枠', "TSC", "決まり手", "PST", "レースタイム"]]
    race_results_df_ = race_results_df_[race_results_df_.index.isin(lace_id_list)]
#(以下変更2022/11/22)
#     df = pd.concat([last_info_df_.reset_index(drop = True), shutuba_tables_df_.reset_index()], axis = 1)
#     df.index = df["index"]
#     df = df.drop(columns = "index")
#     df = df.rename_axis(None)

    shutuba_tables_df_["race_id"] = shutuba_tables_df_.index
    shutuba_tables_df_["merge_id"] = shutuba_tables_df_.groupby(level=0).cumcount()+1
    last_info_df_["race_id"] = last_info_df_.index
    last_info_df_["merge_id"] = last_info_df_.groupby(level=0).cumcount()+1
    df = pd.merge(last_info_df_, shutuba_tables_df_, how = "left",on=["race_id", "merge_id"])
    df.index = df["race_id"]
    df = df.drop(columns = ["race_id", "merge_id"])

    df = df[['ET', 'tilt', 'EST', 'ESC', 'date', 'place', 'race_no', 'weather',
       'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','枠','F', 'L', 'age', 'weight',
       'racer_no', 'racer_lank', 'sibu', 'born', 'zwin_r_1', 'zwin_r_2',
       'zwin_r_3', 'twin_r_1', 'twin_r_2', 'twin_r_3', 'moter_no', 'mwin_r_2',
       'mwin_r_3', 'boat_no', 'bwin_r_2', 'bwin_r_3', 'piston', 'ring',
       'electric', 'carburetor', 'cylinder', 'shafts', 'gears', 'carrier',
       'propera']]
    #E:エキシビジョン，T:タイム，C：コース，win：，wave_h，place：競艇場の場所，
    df['ESC'] = df['ESC'].astype(int)
    df['place'] = df['place'].astype(str)
    df['race_no'] = df['race_no'].astype(str)
    df['mwin_r_2'] = df['mwin_r_2'].astype(float)
    df['mwin_r_3'] = df['mwin_r_3'].astype(float)
    df['bwin_r_2'] = df['bwin_r_2'].astype(float)
    df['bwin_r_3'] = df['bwin_r_3'].astype(float)
    df['age'] = df['age'].astype(int)
    df['weight'] = df['weight'].astype(float)
    
    #df['着'] = df['着'].astype(int)
    
    race_results_df_['race_id'] = race_results_df_.index
    race_results_df_['枠'] = race_results_df_['枠'].astype(str)
    df_v2 = pd.merge(df, race_results_df_, on=['race_id','枠']) #心配ポイント1
    df_v2['着'] = df_v2['着'].map(lambda x: int(x) if x in ['１','２','３','４','５','６'] else 7)
    df_v2.index = df_v2['race_id']
    df_v2.drop(columns = "race_id", inplace = True)
    
    return df_v2

def estimate_fifthsixth_racetime(X):#2023/6/24追加
    mean_race_time_delta = X["レースタイム_refix"].dropna().astype(int).sort_values().diff().dropna().mean()#レースタイムの着間差分
    forth_racetime = X["レースタイム_refix"].sort_values().dropna().astype(int).max()#nanを除いた選手のレースタイム最大値
    
    if X["レースタイム_refix"].isna().sum() == 0:#全着のレースタイムを計測している場合もあるので，その場合の例外処理
        return X
    elif X["レースタイム_refix"].isna().sum() == 1:
        X.loc[(X["着"] == 6) | (X["着"] == 7), "レースタイム_refix"] = str(forth_racetime + (2*mean_race_time_delta))
        return X
    else:
        X.loc[X["着"] == 5, "レースタイム_refix"] = str(forth_racetime + mean_race_time_delta)
        X.loc[(X["着"] == 6) | (X["着"] == 7), "レースタイム_refix"] = str(forth_racetime + (2*mean_race_time_delta))
        return X
    
def correction_RT_preprocessing(last_info_df, shutuba_tables_df, race_results_df):
    
    #学習データのレースIDを取得
    lace_id_list = last_info_df.index.unique()

    #レースIDを持つ出馬表データを取得
    shutuba_tables_df_ = shutuba_tables_df[shutuba_tables_df.index.isin(lace_id_list)]
    last_info_df_ = last_info_df[last_info_df.index.isin(shutuba_tables_df_.index)]
    race_results_df_ = race_results_df[['着','枠', "TSC", "決まり手", "PST", "レースタイム"]]
    race_results_df_ = race_results_df_[race_results_df_.index.isin(lace_id_list)]
#(以下変更2022/11/22)
#     df = pd.concat([last_info_df_.reset_index(drop = True), shutuba_tables_df_.reset_index()], axis = 1)
#     df.index = df["index"]
#     df = df.drop(columns = "index")
#     df = df.rename_axis(None)

    shutuba_tables_df_["race_id"] = shutuba_tables_df_.index
    shutuba_tables_df_["merge_id"] = shutuba_tables_df_.groupby(level=0).cumcount()+1
    last_info_df_["race_id"] = last_info_df_.index
    last_info_df_["merge_id"] = last_info_df_.groupby(level=0).cumcount()+1
    df = pd.merge(last_info_df_, shutuba_tables_df_, how = "left",on=["race_id", "merge_id"])
    df.index = df["race_id"]
    df = df.drop(columns = ["race_id", "merge_id"])

    df = df[['ET', 'tilt', 'EST', 'ESC', 'date', 'place', 'race_no', 'weather',
       'air_t', 'wind_d', 'wind_v', 'water_t', 'wave_h','枠','F', 'L', 'age', 'weight',
       'racer_no', 'racer_lank', 'sibu', 'born', 'zwin_r_1', 'zwin_r_2',
       'zwin_r_3', 'twin_r_1', 'twin_r_2', 'twin_r_3', 'moter_no', 'mwin_r_2',
       'mwin_r_3', 'boat_no', 'bwin_r_2', 'bwin_r_3', 'piston', 'ring',
       'electric', 'carburetor', 'cylinder', 'shafts', 'gears', 'carrier',
       'propera']]
    #E:エキシビジョン，T:タイム，C：コース，win：，wave_h，place：競艇場の場所，
    df['ESC'] = df['ESC'].astype(int)
    df['place'] = df['place'].astype(str)
    df['race_no'] = df['race_no'].astype(str)
    df['mwin_r_2'] = df['mwin_r_2'].astype(float)
    df['mwin_r_3'] = df['mwin_r_3'].astype(float)
    df['bwin_r_2'] = df['bwin_r_2'].astype(float)
    df['bwin_r_3'] = df['bwin_r_3'].astype(float)
    df['age'] = df['age'].astype(int)
    df['weight'] = df['weight'].astype(float)
    
    #df['着'] = df['着'].astype(int)
    
    race_results_df_['race_id'] = race_results_df_.index
    race_results_df_['枠'] = race_results_df_['枠'].astype(str)
    df_v2 = pd.merge(df, race_results_df_, on=['race_id','枠']) #心配ポイント1
    df_v2['着'] = df_v2['着'].map(lambda x: int(x) if x in ['１','２','３','４','５','６'] else 7)
    df_v2.index = df_v2['race_id']
    df_v2.drop(columns = "race_id", inplace = True)
    
    #2023/6/24追加
    df_v2["レースタイム_refix"] = df_v2["レースタイム"].str.replace("'", "")
    df_v2["レースタイム_refix"] = df_v2["レースタイム_refix"].str.replace("\"", "")

    df_v2 = df_v2.groupby(level = 0).apply(estimate_fifthsixth_racetime)
    df_v2["レースタイム_refix"] = df_v2["レースタイム_refix"].astype(float)
    
    return df_v2


#特徴量の追加
def add_features(data):
    print("add_features: 開始")
    print("weather特徴量の作成 開始")
    data["weather"] = ((data["weather"]=="雨") | (data["weather"]=="雪"))*1 #雨と雪は1，それ以外は0
    print("weather特徴量の作成 終了")
    
    print("racer_lank_int特徴量の作成 開始")
    data["racer_lank_int"] = data.replace({'racer_lank' : {"A1": 3,"A2": 2, "B1": 1,"B2": 0 }})["racer_lank"]#レースランクを数値化
    print("racer_lank_int特徴量の作成 終了")
    
    print("レース内での階級のカウント特徴量の作成 開始")
    import time
    start_time1 = time.time()
    data['racer_lank_'] = data['racer_lank'].map(lambda x: x[:1])
    end_time1 = time.time()
    print(f"racer_lank_作成 所要時間: {end_time1 - start_time1:.2f}秒")

    start_time2 = time.time()
    data['racer_lank_'] = data['racer_lank_'].map(lambda x: 1 if x=='A' else 0)
    end_time2 = time.time()
    print(f"racer_lank_数値化 所要時間: {end_time2 - start_time2:.2f}秒")

    start_time3 = time.time()
    data['lank_counts'] = data.groupby(level=0)['racer_lank_'].transform(lambda x: sum(x))
    end_time3 = time.time()
    print(f"lank_counts作成 所要時間: {end_time3 - start_time3:.2f}秒")
    data = data.drop(['racer_lank_','racer_lank'],axis=1)
    print("レース内での階級のカウント特徴量の作成 終了")
    
    print("標準化特徴量の作成 開始")
  # 標準化する列をまとめる
    std_cols = ['ET', 'zwin_r_1', 'zwin_r_2', 'zwin_r_3', 'twin_r_1', 'mwin_r_2', 'bwin_r_2']

    # 事前に index を整える（GroupByの前処理として有効なことが多い）
    # data = data.sort_index()  # 既に整っていれば不要

    # 一括で groupby → mean / std を取得（この transform は C 実装で速い）
    g = data.groupby(level=0)[std_cols]
    means = g.transform('mean')
    stds = g.transform('std').replace(0, np.nan)  # 全要素同値のグループを NaN に

    # z-score をまとめて計算して列名を付与
    z = (data[std_cols] - means) / stds
    z.columns = [f'{c}_std' for c in z.columns]

    # 元データへ結合
    data[z.columns] = z
    print("標準化特徴量の作成 終了")
    #data["MST_std"] = data.groupby(level=0)['MST'].transform(standard_scaler)
    
    # data['1着率_std'] = data.groupby(level=0)['1着率'].transform(standard_scaler)
    # data['2着率_std'] = data.groupby(level=0)['2着率'].transform(standard_scaler)
    # data['3着率_std'] = data.groupby(level=0)['3着率'].transform(standard_scaler)
    # data['4着率_std'] = data.groupby(level=0)['4着率'].transform(standard_scaler)
    # data['5着率_std'] = data.groupby(level=0)['5着率'].transform(standard_scaler)
    # data['6着率_std'] = data.groupby(level=0)['6着率'].transform(standard_scaler)
    print("add_features: 終了")
    return data

def _twin_fix(X): #当地勝率が0のところに，全国勝率を代入
    # ここは処理が軽いのでprintは不要
    if X["twin_r_1"] == 0: 
        return X["zwin_r_1"]
    else: 
        return X["twin_r_1"]

#レータイムの統計量を算出

import swifter
def calc_recetime_statistics(race_results_df, last_info_df, shutuba_tables_df, race_results_df_refix):
    race_results_df_refix = race_results_df_fix(race_results_df)
    df= preprocessing(last_info_df, shutuba_tables_df, race_results_df_refix)

    df["レースタイム_refix"] = df["レースタイム"].str.replace("'", "")
    df["レースタイム_refix"] = df["レースタイム_refix"].str.replace("\"", "")

    speed_statistic_values = df.dropna(subset=['レースタイム_refix'])
    speed_statistic_values["レースタイム_refix"] = speed_statistic_values["レースタイム_refix"].astype(int)


    speed_statistic_values = speed_statistic_values[["レースタイム_refix", "water_t", "place", "TSC"]]
    water_t_statistic_values = []
    place_statistic_values = []
    TSC_statistic_values = []
    water_t_delta = 5
    for tsc in np.sort(speed_statistic_values["TSC"].unique()):
        TSC_statistic_values.append([tsc
                                    ,speed_statistic_values[speed_statistic_values["TSC"] == tsc]["レースタイム_refix"].mean()
                                    ,speed_statistic_values[speed_statistic_values["TSC"] == tsc]["レースタイム_refix"].std()])

    for place in np.sort(speed_statistic_values["place"].unique().astype(int)):
        place_statistic_values.append([place
                                    ,speed_statistic_values[speed_statistic_values["place"] == str(place)]["レースタイム_refix"].mean()
                                    ,speed_statistic_values[speed_statistic_values["place"] == str(place)]["レースタイム_refix"].std()])

    for water_t in range(-5, 40 , water_t_delta):
        water_t_statistic_values.append([water_t
                                    ,speed_statistic_values[(speed_statistic_values["water_t"] >= water_t) & (speed_statistic_values["water_t"] < (water_t+water_t_delta))]["レースタイム_refix"].mean()
                                    ,speed_statistic_values[(speed_statistic_values["water_t"] >= water_t) & (speed_statistic_values["water_t"] < (water_t+water_t_delta))]["レースタイム_refix"].std()])
    water_t_statistic_values[0] = [-5, 1505, 90] #-5 ~ 0度はサンプル数が少なすぎる．よって，主観的に補正

    print("water_t_statistic_values->")
    print(water_t_statistic_values)
    print("place_statistic_values->")
    print(place_statistic_values)
    print("TSC_statistic_values->")
    print(TSC_statistic_values)
    return np.array(water_t_statistic_values) , np.array(place_statistic_values), np.array(TSC_statistic_values)


#race_results_df.index.name = None #PST追加からこの行を追加．
#レースタイムの統計値を計算
# 計算時間(50.2[s])
#water_t_statistic_values , place_statistic_values, TSC_statistic_values = calc_recetime_statistics(race_results_df, last_info_df, shutuba_tables_df, race_results_df)
#water_t_delta = 5

# 計算時間(2.7[s])
# race_results_df_refix = race_results_df_fix(race_results_df)

# 計算時間(18.7[m])
use_correction = os.environ.get("FVM_USE_CORRECTION", "1") != "0"
if use_correction:
    race_results_df_refix = race_results_df_fix(race_results_df)
    df = correction_RT_preprocessing(last_info_df, shutuba_tables_df, race_results_df_refix)
    df["twin_r_1"] = df.swifter.apply(_twin_fix, axis=1)
else:
    df = preprocessing(last_info_df, shutuba_tables_df, race_results_df)
df = add_features(df)


df = df.droplevel(list(range(1, df.index.nlevels)))#マルチインデックスを削除
df.to_csv("data/df.csv")
