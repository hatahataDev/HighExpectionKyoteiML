import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("lightgbm is required. Install with: pip install lightgbm") from exc

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError as exc:
    raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from exc


TARGET_COL = "レースタイム_refix"
RAW_TARGET_COL = "レースタイム"
TARGET_SEC_COL = "target_sec"
DATE_COL = "date"
RACE_ID_COL = "race_id"

RESULT_COLS = ["着", "決まり手", "PST", "TSC"]
IDENTIFIER_COLS = [RACE_ID_COL]
DROP_COLS = RESULT_COLS + IDENTIFIER_COLS + [RAW_TARGET_COL, TARGET_COL]

CATEGORICAL_COLS = [
    "place",
    "race_no",
    "ESC",
    "枠",
    "racer_no",
    "moter_no",
    "boat_no",
    "sibu",
    "born",
    "piston",
    "ring",
    "electric",
    "carburetor",
    "cylinder",
    "shafts",
    "gears",
    "carrier",
    "propera",
]
PRED_SAMPLE_ROWS_DEFAULT = 100000
PLOT_SAMPLE_ROWS_DEFAULT = 200000
ERROR_TOPN_PLACE_DEFAULT = 20


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_writer(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = log_path.open("w", encoding="utf-8")

    def _log(level: str, event: str, message: str, data=None):
        record = {
            "ts": _now_ts(),
            "level": level,
            "event": event,
            "message": message,
        }
        if data is not None:
            record["data"] = data
        line = json.dumps(record, ensure_ascii=False)
        print(line)
        fp.write(line + "\n")
        fp.flush()

    return _log, fp


def convert_racetime_to_seconds(rt_series: pd.Series) -> pd.Series:
    rt = rt_series.astype(float)
    minutes = np.floor(rt / 1000)
    seconds = (rt % 1000) / 10
    return minutes * 60 + seconds


def ensure_category(df: pd.DataFrame, cols: list) -> list:
    present = [c for c in cols if c in df.columns]
    for c in present:
        df[c] = df[c].astype("category")
    return present


def build_feature_columns(df: pd.DataFrame, add_date_features: bool) -> list:
    exclude = set(DROP_COLS + [TARGET_SEC_COL, "split"])
    if not add_date_features:
        exclude.add(DATE_COL)
    features = [c for c in df.columns if c not in exclude]
    # date 자체는 학習 특징量に使わない（派生特徴量のみ使用）
    if DATE_COL in features:
        features.remove(DATE_COL)
    return features


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_intermediate_artifacts(df: pd.DataFrame, out_dir: Path, log):
    inter_dir = out_dir / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)

    # Dataset overview
    overview = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "date_min": str(df[DATE_COL].min().date()),
        "date_max": str(df[DATE_COL].max().date()),
        "target_missing": int(df[TARGET_SEC_COL].isna().sum()),
    }
    write_json(inter_dir / "dataset_overview.json", overview)
    log("INFO", "intermediate", "wrote dataset_overview.json", {"path": str(inter_dir / "dataset_overview.json")})

    # Missing summary
    missing = df.isna().sum().to_frame("missing")
    missing["missing_rate"] = missing["missing"] / len(df)
    missing.to_csv(inter_dir / "missing_summary.csv", index=True)
    log("INFO", "intermediate", "wrote missing_summary.csv", {"path": str(inter_dir / "missing_summary.csv")})

    # Categorical cardinality
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cat_cols:
        cardinality = pd.DataFrame({"column": cat_cols, "unique": [df[c].nunique(dropna=True) for c in cat_cols]})
        cardinality.to_csv(inter_dir / "categorical_cardinality.csv", index=False)
        log("INFO", "intermediate", "wrote categorical_cardinality.csv", {"path": str(inter_dir / "categorical_cardinality.csv")})

    # Feature summary (numeric)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        summary = df[numeric_cols].describe(percentiles=[0.05, 0.5, 0.95]).T
        summary = summary.rename(columns={
            "5%": "p5",
            "50%": "p50",
            "95%": "p95",
        })
        summary.to_csv(inter_dir / "feature_summary.csv", index=True)
        log("INFO", "intermediate", "wrote feature_summary.csv", {"path": str(inter_dir / "feature_summary.csv")})

    # Target histogram
    target = df[TARGET_SEC_COL].dropna()
    if not target.empty:
        bins = 100
        hist, edges = np.histogram(target, bins=bins)
        hist_df = pd.DataFrame({
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": hist,
        })
        hist_df.to_csv(inter_dir / "target_hist.csv", index=False)
        log("INFO", "intermediate", "wrote target_hist.csv", {"path": str(inter_dir / "target_hist.csv")})

    # Target by month
    month_key = df[DATE_COL].dt.to_period("M").astype(str)
    target_by_month = (
        df.assign(month=month_key)
        .groupby("month")[TARGET_SEC_COL]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    target_by_month.to_csv(inter_dir / "target_by_month.csv", index=False)
    log("INFO", "intermediate", "wrote target_by_month.csv", {"path": str(inter_dir / "target_by_month.csv")})

    # Optional plots (if matplotlib installed)
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig = plt.figure(figsize=(8, 4))
        plt.hist(target, bins=100)
        plt.title("target_sec histogram")
        plt.tight_layout()
        fig.savefig(inter_dir / "target_hist.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        plt.plot(target_by_month["month"], target_by_month["mean"])
        plt.xticks(rotation=90)
        plt.title("target_sec mean by month")
        plt.tight_layout()
        fig.savefig(inter_dir / "target_by_month.png")
        plt.close(fig)

        log("INFO", "intermediate", "wrote target plots", {"path": str(inter_dir)})
    except Exception as exc:  # matplotlib not installed or other error
        log("WARNING", "intermediate", "plot generation skipped", {"error": str(exc)})


def save_split_artifacts(df: pd.DataFrame, out_dir: Path, log):
    inter_dir = out_dir / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)

    split_summary = (
        df.groupby("split")
        .agg(rows=(RACE_ID_COL, "size"), unique_race_id=(RACE_ID_COL, "nunique"))
        .reset_index()
    )
    split_summary.to_csv(inter_dir / "split_summary.csv", index=False)
    log("INFO", "intermediate", "wrote split_summary.csv", {"path": str(inter_dir / "split_summary.csv")})

    split_ranges = (
        df.groupby("split")[DATE_COL]
        .agg(date_min="min", date_max="max")
        .reset_index()
    )
    ranges_payload = {
        row["split"]: {
            "date_min": str(row["date_min"].date()),
            "date_max": str(row["date_max"].date()),
        }
        for _, row in split_ranges.iterrows()
    }
    write_json(inter_dir / "split_date_ranges.json", ranges_payload)
    log("INFO", "intermediate", "wrote split_date_ranges.json", {"path": str(inter_dir / "split_date_ranges.json")})

    # overlap check
    train_ids = set(df.loc[df["split"] == "train", RACE_ID_COL].unique())
    valid_ids = set(df.loc[df["split"] == "valid", RACE_ID_COL].unique())
    test_ids = set(df.loc[df["split"] == "test", RACE_ID_COL].unique())
    overlap = {
        "train_valid": len(train_ids & valid_ids),
        "train_test": len(train_ids & test_ids),
        "valid_test": len(valid_ids & test_ids),
    }
    (inter_dir / "split_overlap_check.txt").write_text(
        "\n".join([f"{k}: {v}" for k, v in overlap.items()]) + "\n",
        encoding="utf-8",
    )
    log("INFO", "intermediate", "wrote split_overlap_check.txt", {"path": str(inter_dir / "split_overlap_check.txt"), "overlap": overlap})


def sample_rows(df: pd.DataFrame, out_dir: Path, seed: int, n_rows: int, log):
    inter_dir = out_dir / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_rows, len(df))
    sample = df.sample(n=n, random_state=seed)
    sample.to_csv(inter_dir / "sample_rows.csv", index=False)
    log("INFO", "intermediate", "wrote sample_rows.csv", {"path": str(inter_dir / "sample_rows.csv"), "rows": int(n)})


def save_final_artifacts(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    out_dir: Path,
    log,
    seed: int,
    pred_sample_rows: int,
    plot_sample_rows: int,
    error_topn_place: int,
    evals_result=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    result_df = test_df[[RACE_ID_COL, DATE_COL, "place"]].copy()
    result_df[TARGET_SEC_COL] = test_df[TARGET_SEC_COL].values
    result_df["pred_sec"] = preds
    result_df["abs_error"] = (result_df["pred_sec"] - result_df[TARGET_SEC_COL]).abs()
    result_df["residual"] = result_df["pred_sec"] - result_df[TARGET_SEC_COL]

    # predictions sample
    sample_n = min(pred_sample_rows, len(result_df))
    pred_sample = result_df.sample(n=sample_n, random_state=seed)
    pred_sample_out = pred_sample[[RACE_ID_COL, DATE_COL, TARGET_SEC_COL, "pred_sec", "abs_error"]]
    pred_sample_out.to_csv(out_dir / "predictions_sample.csv", index=False)
    log("INFO", "final", "wrote predictions_sample.csv", {"path": str(out_dir / "predictions_sample.csv"), "rows": int(sample_n)})

    # residual aggregates
    month_key = result_df[DATE_COL].dt.to_period("M").astype(str)
    residuals_by_month = (
        result_df.assign(month=month_key)
        .groupby("month")
        .apply(
            lambda g: pd.Series(
                {
                    "count": len(g),
                    "mean": g["residual"].mean(),
                    "median": g["residual"].median(),
                    "std": g["residual"].std(),
                    "rmse": np.sqrt(np.mean(g["residual"] ** 2)),
                    "mae": g["abs_error"].mean(),
                }
            )
        )
        .reset_index()
    )
    residuals_by_month.to_csv(out_dir / "residuals_by_month.csv", index=False)
    log("INFO", "final", "wrote residuals_by_month.csv", {"path": str(out_dir / "residuals_by_month.csv")})

    residuals_by_place = (
        result_df.groupby("place")
        .apply(
            lambda g: pd.Series(
                {
                    "count": len(g),
                    "mean": g["residual"].mean(),
                    "median": g["residual"].median(),
                    "std": g["residual"].std(),
                    "rmse": np.sqrt(np.mean(g["residual"] ** 2)),
                    "mae": g["abs_error"].mean(),
                }
            )
        )
        .reset_index()
    )
    residuals_by_place.to_csv(out_dir / "residuals_by_place.csv", index=False)
    log("INFO", "final", "wrote residuals_by_place.csv", {"path": str(out_dir / "residuals_by_place.csv")})

    # error quantiles
    quantiles = result_df["abs_error"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    quantiles = {f"p{int(k*100)}": float(v) for k, v in quantiles.items()}
    write_json(out_dir / "error_quantiles.json", quantiles)
    log("INFO", "final", "wrote error_quantiles.json", {"path": str(out_dir / "error_quantiles.json")})

    # plots
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        # learning curve
        if evals_result and "valid_0" in evals_result:
            metric_name = next(iter(evals_result["valid_0"].keys()))
            values = evals_result["valid_0"][metric_name]
            fig = plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(values) + 1), values)
            plt.xlabel("Iteration")
            plt.ylabel(metric_name)
            plt.title("Learning Curve (valid)")
            plt.tight_layout()
            fig.savefig(plots_dir / "learning_curve.png")
            plt.close(fig)

        # sample for plots
        plot_n = min(plot_sample_rows, len(result_df))
        plot_sample = result_df.sample(n=plot_n, random_state=seed)

        # pred vs true scatter
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(plot_sample[TARGET_SEC_COL], plot_sample["pred_sec"], s=5, alpha=0.3)
        plt.xlabel("True (sec)")
        plt.ylabel("Pred (sec)")
        plt.title("Pred vs True (sample)")
        plt.tight_layout()
        fig.savefig(plots_dir / "pred_vs_true.png")
        plt.close(fig)

        # pred vs true hexbin
        fig = plt.figure(figsize=(6, 6))
        plt.hexbin(plot_sample[TARGET_SEC_COL], plot_sample["pred_sec"], gridsize=60, cmap="viridis")
        plt.xlabel("True (sec)")
        plt.ylabel("Pred (sec)")
        plt.title("Pred vs True Hexbin (sample)")
        plt.tight_layout()
        fig.savefig(plots_dir / "pred_vs_true_hexbin.png")
        plt.close(fig)

        # residual histogram
        fig = plt.figure(figsize=(8, 4))
        plt.hist(plot_sample["residual"], bins=100)
        plt.title("Residuals (pred - true)")
        plt.tight_layout()
        fig.savefig(plots_dir / "residual_hist.png")
        plt.close(fig)

        # absolute error histogram
        fig = plt.figure(figsize=(8, 4))
        plt.hist(plot_sample["abs_error"], bins=100)
        plt.title("Absolute Error")
        plt.tight_layout()
        fig.savefig(plots_dir / "abs_error_hist.png")
        plt.close(fig)

        # residual by month
        fig = plt.figure(figsize=(10, 4))
        plt.plot(residuals_by_month["month"], residuals_by_month["mean"])
        plt.xticks(rotation=90)
        plt.title("Residual Mean by Month")
        plt.tight_layout()
        fig.savefig(plots_dir / "residual_by_month.png")
        plt.close(fig)

        # mae/rmse by month
        fig = plt.figure(figsize=(10, 4))
        plt.plot(residuals_by_month["month"], residuals_by_month["mae"], label="MAE")
        plt.xticks(rotation=90)
        plt.legend()
        plt.title("MAE by Month")
        plt.tight_layout()
        fig.savefig(plots_dir / "mae_by_month.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        plt.plot(residuals_by_month["month"], residuals_by_month["rmse"], label="RMSE")
        plt.xticks(rotation=90)
        plt.legend()
        plt.title("RMSE by Month")
        plt.tight_layout()
        fig.savefig(plots_dir / "rmse_by_month.png")
        plt.close(fig)

        # error by place top N (RMSE)
        topn = residuals_by_place.sort_values("rmse", ascending=False).head(error_topn_place)
        fig = plt.figure(figsize=(10, 6))
        plt.bar(topn["place"].astype(str), topn["rmse"])
        plt.xticks(rotation=90)
        plt.title(f"RMSE by Place (Top {error_topn_place})")
        plt.tight_layout()
        fig.savefig(plots_dir / "error_by_place_topN.png")
        plt.close(fig)

        # feature importance plots
        fi = pd.read_csv(out_dir / "feature_importance.csv")
        fi_gain = fi.sort_values("importance_gain", ascending=False).head(50)
        fig = plt.figure(figsize=(8, 10))
        plt.barh(fi_gain["feature"][::-1], fi_gain["importance_gain"][::-1])
        plt.title("Feature Importance (gain) Top 50")
        plt.tight_layout()
        fig.savefig(plots_dir / "feature_importance_gain_top50.png")
        plt.close(fig)

        fi_split = fi.sort_values("importance_split", ascending=False).head(50)
        fig = plt.figure(figsize=(8, 10))
        plt.barh(fi_split["feature"][::-1], fi_split["importance_split"][::-1])
        plt.title("Feature Importance (split) Top 50")
        plt.tight_layout()
        fig.savefig(plots_dir / "feature_importance_split_top50.png")
        plt.close(fig)

        # calibration-like plot (pred bin -> mean true)
        bin_count = 50
        pred_bins = pd.qcut(plot_sample["pred_sec"], q=bin_count, duplicates="drop")
        calib = plot_sample.groupby(pred_bins)[TARGET_SEC_COL].mean()
        fig = plt.figure(figsize=(8, 4))
        plt.plot(range(len(calib)), calib.values)
        plt.title("Calibration-like Plot (Mean True by Pred Bin)")
        plt.tight_layout()
        fig.savefig(plots_dir / "calibration_like_plot.png")
        plt.close(fig)

        # error quantile curve
        q_vals = result_df["abs_error"].quantile([0.5, 0.9, 0.95, 0.99])
        fig = plt.figure(figsize=(6, 4))
        plt.plot([0.5, 0.9, 0.95, 0.99], q_vals.values, marker="o")
        plt.title("Absolute Error Quantiles")
        plt.tight_layout()
        fig.savefig(plots_dir / "error_quantile_curve.png")
        plt.close(fig)

        log("INFO", "final", "wrote plots", {"path": str(plots_dir)})
    except Exception as exc:
        log("WARNING", "final", "plot generation skipped", {"error": str(exc)})


def train(args):
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "logs" / f"run_{run_ts}.log"
    log, log_fp = _log_writer(log_path)

    start_time = time.perf_counter()
    log("INFO", "run", "start", {"data_path": args.data_path, "output_dir": str(out_dir)})

    # Load data
    log("INFO", "data", "loading df.csv")
    df = pd.read_csv(args.data_path)
    log("INFO", "data", "loaded df.csv", {"rows": int(len(df)), "columns": int(df.shape[1])})

    # date parsing
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # target conversion
    df[TARGET_SEC_COL] = convert_racetime_to_seconds(df[TARGET_COL])

    # drop rows with missing target or date
    before = len(df)
    df = df.dropna(subset=[TARGET_SEC_COL, DATE_COL])
    dropped = before - len(df)
    log("INFO", "data", "dropped rows with missing target/date", {"dropped": int(dropped)})

    # date features
    if args.add_date_features:
        df["year"] = df[DATE_COL].dt.year
        df["month"] = df[DATE_COL].dt.month
        df["dayofweek"] = df[DATE_COL].dt.dayofweek

    # sample by race_id if requested
    if args.sample_frac and args.sample_frac < 1.0:
        rng = np.random.default_rng(args.sample_seed)
        race_ids = df[RACE_ID_COL].unique()
        keep_size = int(len(race_ids) * args.sample_frac)
        keep_ids = set(rng.choice(race_ids, size=keep_size, replace=False).tolist())
        before = len(df)
        df = df[df[RACE_ID_COL].isin(keep_ids)].copy()
        log("INFO", "data", "applied sample_frac", {"sample_frac": args.sample_frac, "rows": int(len(df)), "dropped": int(before - len(df))})

    # categorical conversion
    cat_cols_present = ensure_category(df, CATEGORICAL_COLS)
    log("INFO", "data", "categorical columns", {"categorical": cat_cols_present})

    # intermediate artifacts before split
    save_intermediate_artifacts(df, out_dir, log)

    # split by race_id date
    train_end = pd.to_datetime(args.train_end)
    valid_end = pd.to_datetime(args.valid_end)

    race_date = df.groupby(RACE_ID_COL, sort=False)[DATE_COL].min()
    race_split = pd.Series(index=race_date.index, dtype="object")
    race_split.loc[race_date <= train_end] = "train"
    race_split.loc[(race_date > train_end) & (race_date <= valid_end)] = "valid"
    race_split.loc[race_date > valid_end] = "test"

    df["split"] = df[RACE_ID_COL].map(race_split)
    if df["split"].isna().any():
        missing_split = int(df["split"].isna().sum())
        log("ERROR", "split", "rows without split assignment", {"missing": missing_split})
        raise ValueError("Some rows are missing split assignment.")

    save_split_artifacts(df, out_dir, log)
    sample_rows(df, out_dir, args.sample_seed, args.sample_rows, log)

    # features
    features = build_feature_columns(df, args.add_date_features)
    log("INFO", "features", "feature columns prepared", {"count": len(features)})

    # datasets
    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[features]
    y_train = train_df[TARGET_SEC_COL]
    X_valid = valid_df[features]
    y_valid = valid_df[TARGET_SEC_COL]
    X_test = test_df[features]
    y_test = test_df[TARGET_SEC_COL]

    log("INFO", "split", "split sizes", {
        "train": int(len(train_df)),
        "valid": int(len(valid_df)),
        "test": int(len(test_df)),
    })

    params = {
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

    log("INFO", "train", "start", {"params": params})
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        categorical_feature=[c for c in cat_cols_present if c in features],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(args.log_every_n),
        ],
    )

    # evaluation
    log("INFO", "eval", "evaluating")
    preds = model.predict(X_test, num_iteration=model.best_iteration_)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "best_iteration": int(model.best_iteration_ or 0),
    }

    # outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.txt"
    model.booster_.save_model(str(model_path))

    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, metrics)

    # feature importances
    importance_split = model.booster_.feature_importance(importance_type="split")
    importance_gain = model.booster_.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame({
        "feature": features,
        "importance_split": importance_split,
        "importance_gain": importance_gain,
    }).sort_values("importance_gain", ascending=False)
    fi_df.to_csv(out_dir / "feature_importance.csv", index=False)
    fi_df.head(100).to_csv(out_dir / "feature_importance_gain_top100.csv", index=False)

    # config
    config = {
        "data_path": args.data_path,
        "output_dir": str(out_dir),
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "add_date_features": args.add_date_features,
        "sample_frac": args.sample_frac,
        "sample_seed": args.sample_seed,
        "pred_sample_rows": args.pred_sample_rows,
        "plot_sample_rows": args.plot_sample_rows,
        "error_topn_place": args.error_topn_place,
        "features": features,
        "categorical_features": [c for c in cat_cols_present if c in features],
        "params": params,
    }
    write_json(out_dir / "config.json", config)

    # final artifacts
    save_final_artifacts(
        test_df=test_df,
        preds=preds,
        out_dir=out_dir,
        log=log,
        seed=args.sample_seed,
        pred_sample_rows=args.pred_sample_rows,
        plot_sample_rows=args.plot_sample_rows,
        error_topn_place=args.error_topn_place,
        evals_result=getattr(model, "evals_result_", None),
    )

    elapsed = time.perf_counter() - start_time
    log("INFO", "run", "completed", {
        "elapsed_sec": round(elapsed, 2),
        "metrics": metrics,
        "outputs": {
            "model": str(model_path),
            "metrics": str(metrics_path),
            "feature_importance": str(out_dir / "feature_importance.csv"),
            "config": str(out_dir / "config.json"),
            "log": str(log_path),
        },
    })
    log_fp.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM regressor for race time prediction")
    parser.add_argument("--data-path", default="Data/df.csv", help="Path to df.csv")
    parser.add_argument("--output-dir", default="artifacts/lgbm_racetime", help="Output directory")
    parser.add_argument("--train-end", default="2022-12-31", help="Train end date (YYYY-MM-DD)")
    parser.add_argument("--valid-end", default="2023-12-31", help="Valid end date (YYYY-MM-DD)")
    parser.add_argument("--add-date-features", action="store_true", default=True, help="Add year/month/dayofweek features")
    parser.add_argument("--no-date-features", dest="add_date_features", action="store_false", help="Disable date features")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Fraction of race_id to sample")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--sample-rows", type=int, default=10000, help="Rows for sample_rows.csv")
    parser.add_argument("--pred-sample-rows", type=int, default=PRED_SAMPLE_ROWS_DEFAULT, help="Rows for predictions_sample.csv")
    parser.add_argument("--plot-sample-rows", type=int, default=PLOT_SAMPLE_ROWS_DEFAULT, help="Rows used in plots")
    parser.add_argument("--error-topn-place", type=int, default=ERROR_TOPN_PLACE_DEFAULT, help="Top-N places for error_by_place plot")

    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--log-every-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
