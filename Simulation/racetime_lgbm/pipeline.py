from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("lightgbm is required. Install with: pip install lightgbm") from exc

try:
    from sklearn.metrics import mean_absolute_error, r2_score
except ImportError as exc:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from exc

from .constants import CATEGORICAL_COLS, DATE_COL, DROP_COLS, RACE_ID_COL, TARGET_COL, TARGET_SEC_COL
from .metrics import compute_rmse, convert_racetime_to_seconds


@dataclass
class TrainSplitFrames:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def ensure_category(df: pd.DataFrame, cols: list) -> list:
    present = [c for c in cols if c in df.columns]
    for col in present:
        df[col] = df[col].astype("category")
    return present


def build_feature_columns(df: pd.DataFrame, add_date_features: bool) -> list:
    exclude = set(DROP_COLS + [TARGET_SEC_COL, "split"])
    if not add_date_features:
        exclude.add(DATE_COL)
    features = [col for col in df.columns if col not in exclude]
    if DATE_COL in features:
        features.remove(DATE_COL)
    return features


def load_raw_dataframe(data_path: str, log) -> pd.DataFrame:
    log("INFO", "data", "loading df.csv")
    df = pd.read_csv(data_path)
    log("INFO", "data", "loaded df.csv", {"rows": int(len(df)), "columns": int(df.shape[1])})
    return df


def sample_by_race_id(df: pd.DataFrame, sample_frac: float, sample_seed: int, log) -> pd.DataFrame:
    if not sample_frac or sample_frac >= 1.0:
        return df
    rng = np.random.default_rng(sample_seed)
    race_ids = df[RACE_ID_COL].unique()
    keep_size = int(len(race_ids) * sample_frac)
    keep_ids = set(rng.choice(race_ids, size=keep_size, replace=False).tolist())
    before = len(df)
    sampled = df[df[RACE_ID_COL].isin(keep_ids)].copy()
    log(
        "INFO",
        "data",
        "applied sample_frac",
        {"sample_frac": sample_frac, "rows": int(len(sampled)), "dropped": int(before - len(sampled))},
    )
    return sampled


def prepare_training_dataframe(df: pd.DataFrame, args, log):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_SEC_COL] = convert_racetime_to_seconds(df[TARGET_COL])

    before = len(df)
    df = df.dropna(subset=[TARGET_SEC_COL, DATE_COL])
    dropped = before - len(df)
    log("INFO", "data", "dropped rows with missing target/date", {"dropped": int(dropped)})

    if args.add_date_features:
        df["year"] = df[DATE_COL].dt.year
        df["month"] = df[DATE_COL].dt.month
        df["dayofweek"] = df[DATE_COL].dt.dayofweek

    df = sample_by_race_id(df, args.sample_frac, args.sample_seed, log)
    cat_cols_present = ensure_category(df, CATEGORICAL_COLS)
    log("INFO", "data", "categorical columns", {"categorical": cat_cols_present})
    return df, cat_cols_present


def assign_split_labels(df: pd.DataFrame, train_end: str, valid_end: str, log) -> pd.DataFrame:
    train_end_dt = pd.to_datetime(train_end)
    valid_end_dt = pd.to_datetime(valid_end)

    race_date = df.groupby(RACE_ID_COL, sort=False)[DATE_COL].min()
    race_split = pd.Series(index=race_date.index, dtype="object")
    race_split.loc[race_date <= train_end_dt] = "train"
    race_split.loc[(race_date > train_end_dt) & (race_date <= valid_end_dt)] = "valid"
    race_split.loc[race_date > valid_end_dt] = "test"

    df = df.copy()
    df["split"] = df[RACE_ID_COL].map(race_split)
    if df["split"].isna().any():
        missing_split = int(df["split"].isna().sum())
        log("ERROR", "split", "rows without split assignment", {"missing": missing_split})
        raise ValueError("Some rows are missing split assignment.")
    return df


def split_frames(df: pd.DataFrame) -> TrainSplitFrames:
    return TrainSplitFrames(
        train=df[df["split"] == "train"],
        valid=df[df["split"] == "valid"],
        test=df[df["split"] == "test"],
    )


def build_model_matrices(splits: TrainSplitFrames, features: list):
    x_train = splits.train[features]
    y_train = splits.train[TARGET_SEC_COL]
    x_valid = splits.valid[features]
    y_valid = splits.valid[TARGET_SEC_COL]
    x_test = splits.test[features]
    y_test = splits.test[TARGET_SEC_COL]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def build_lgbm_params(args) -> dict:
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
    }


def train_lgbm_model(
    params: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: list,
    early_stopping_rounds: int,
    log_every_n: int,
    log,
):
    log("INFO", "train", "start", {"params": params})
    model = lgb.LGBMRegressor(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="rmse",
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(log_every_n),
        ],
    )
    return model


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series, log):
    log("INFO", "eval", "evaluating")
    preds = model.predict(x_test, num_iteration=model.best_iteration_)
    metrics = {
        "rmse": float(compute_rmse(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "best_iteration": int(model.best_iteration_ or 0),
    }
    return preds, metrics
