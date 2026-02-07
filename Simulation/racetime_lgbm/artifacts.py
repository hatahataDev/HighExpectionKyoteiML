import json
from pathlib import Path

import numpy as np
import pandas as pd

from .betting import evaluate_trifecta_betting
from .constants import CATEGORICAL_COLS, DATE_COL, RACE_ID_COL, TARGET_SEC_COL


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_intermediate_artifacts(df: pd.DataFrame, out_dir: Path, log):
    inter_dir = out_dir / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)

    overview = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "date_min": str(df[DATE_COL].min().date()),
        "date_max": str(df[DATE_COL].max().date()),
        "target_missing": int(df[TARGET_SEC_COL].isna().sum()),
    }
    write_json(inter_dir / "dataset_overview.json", overview)
    log("INFO", "intermediate", "wrote dataset_overview.json", {"path": str(inter_dir / "dataset_overview.json")})

    missing = df.isna().sum().to_frame("missing")
    missing["missing_rate"] = missing["missing"] / len(df)
    missing.to_csv(inter_dir / "missing_summary.csv", index=True)
    log("INFO", "intermediate", "wrote missing_summary.csv", {"path": str(inter_dir / "missing_summary.csv")})

    cat_cols = [col for col in CATEGORICAL_COLS if col in df.columns]
    if cat_cols:
        cardinality = pd.DataFrame({"column": cat_cols, "unique": [df[col].nunique(dropna=True) for col in cat_cols]})
        cardinality.to_csv(inter_dir / "categorical_cardinality.csv", index=False)
        log(
            "INFO",
            "intermediate",
            "wrote categorical_cardinality.csv",
            {"path": str(inter_dir / "categorical_cardinality.csv")},
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        summary = df[numeric_cols].describe(percentiles=[0.05, 0.5, 0.95]).T
        summary = summary.rename(columns={"5%": "p5", "50%": "p50", "95%": "p95"})
        summary.to_csv(inter_dir / "feature_summary.csv", index=True)
        log("INFO", "intermediate", "wrote feature_summary.csv", {"path": str(inter_dir / "feature_summary.csv")})

    target = df[TARGET_SEC_COL].dropna()
    if not target.empty:
        hist, edges = np.histogram(target, bins=100)
        hist_df = pd.DataFrame(
            {"bin_left": edges[:-1], "bin_right": edges[1:], "count": hist}
        )
        hist_df.to_csv(inter_dir / "target_hist.csv", index=False)
        log("INFO", "intermediate", "wrote target_hist.csv", {"path": str(inter_dir / "target_hist.csv")})

    month_key = df[DATE_COL].dt.to_period("M").astype(str)
    target_by_month = (
        df.assign(month=month_key)
        .groupby("month")[TARGET_SEC_COL]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    target_by_month.to_csv(inter_dir / "target_by_month.csv", index=False)
    log("INFO", "intermediate", "wrote target_by_month.csv", {"path": str(inter_dir / "target_by_month.csv")})

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
    except Exception as exc:
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

    split_ranges = df.groupby("split")[DATE_COL].agg(date_min="min", date_max="max").reset_index()
    ranges_payload = {
        row["split"]: {"date_min": str(row["date_min"].date()), "date_max": str(row["date_max"].date())}
        for _, row in split_ranges.iterrows()
    }
    write_json(inter_dir / "split_date_ranges.json", ranges_payload)
    log(
        "INFO",
        "intermediate",
        "wrote split_date_ranges.json",
        {"path": str(inter_dir / "split_date_ranges.json")},
    )

    train_ids = set(df.loc[df["split"] == "train", RACE_ID_COL].unique())
    valid_ids = set(df.loc[df["split"] == "valid", RACE_ID_COL].unique())
    test_ids = set(df.loc[df["split"] == "test", RACE_ID_COL].unique())
    overlap = {
        "train_valid": len(train_ids & valid_ids),
        "train_test": len(train_ids & test_ids),
        "valid_test": len(valid_ids & test_ids),
    }
    (inter_dir / "split_overlap_check.txt").write_text(
        "\n".join([f"{key}: {value}" for key, value in overlap.items()]) + "\n",
        encoding="utf-8",
    )
    log(
        "INFO",
        "intermediate",
        "wrote split_overlap_check.txt",
        {"path": str(inter_dir / "split_overlap_check.txt"), "overlap": overlap},
    )


def sample_rows(df: pd.DataFrame, out_dir: Path, seed: int, n_rows: int, log):
    inter_dir = out_dir / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_rows, len(df))
    sample = df.sample(n=n, random_state=seed)
    sample.to_csv(inter_dir / "sample_rows.csv", index=False)
    log("INFO", "intermediate", "wrote sample_rows.csv", {"path": str(inter_dir / "sample_rows.csv"), "rows": int(n)})


def summarize_error_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    work = df[[group_col, "residual", "abs_error"]].copy()
    work["residual_sq"] = work["residual"] ** 2
    summary = (
        work.groupby(group_col, observed=False)
        .agg(
            count=("residual", "size"),
            mean=("residual", "mean"),
            median=("residual", "median"),
            std=("residual", "std"),
            mse=("residual_sq", "mean"),
            mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    return summary


def build_trifecta_exact_hit_table(test_df: pd.DataFrame, preds: np.ndarray):
    required_cols = {RACE_ID_COL, DATE_COL, "place", "着", "枠"}
    missing_cols = sorted([col for col in required_cols if col not in test_df.columns])
    if missing_cols:
        raise KeyError(f"3連単評価に必要な列が不足しています: {missing_cols}")

    race_df = test_df[[RACE_ID_COL, DATE_COL, "place", "着", "枠"]].copy()
    race_df["pred_sec"] = preds
    race_df = race_df.dropna(subset=[RACE_ID_COL, "pred_sec", "着", "枠"]).copy()
    race_df["着_int"] = pd.to_numeric(race_df["着"], errors="coerce")
    race_df = race_df.dropna(subset=["着_int"]).copy()
    race_df["着_int"] = race_df["着_int"].astype(int)
    race_df["枠_str"] = race_df["枠"].astype(str)

    pred_top3 = (
        race_df.sort_values([RACE_ID_COL, "pred_sec", "枠_str"], kind="mergesort")
        .groupby(RACE_ID_COL, sort=False)
        .head(3)
        .copy()
    )
    pred_top3["rank"] = pred_top3.groupby(RACE_ID_COL, sort=False).cumcount() + 1
    pred_wide = pred_top3.pivot(index=RACE_ID_COL, columns="rank", values="枠_str")
    pred_wide = pred_wide.rename(columns={1: "pred_1", 2: "pred_2", 3: "pred_3"})
    pred_wide = pred_wide.reindex(columns=["pred_1", "pred_2", "pred_3"]).dropna()

    actual_top3 = race_df[race_df["着_int"].isin([1, 2, 3])].copy()
    actual_top3 = actual_top3.sort_values([RACE_ID_COL, "着_int", "枠_str"], kind="mergesort")
    actual_top3 = actual_top3.drop_duplicates(subset=[RACE_ID_COL, "着_int"], keep="first")
    valid_actual_ids = (
        actual_top3.groupby(RACE_ID_COL)["着_int"]
        .nunique()
        .loc[lambda s: s.eq(3)]
        .index
    )
    actual_top3 = actual_top3[actual_top3[RACE_ID_COL].isin(valid_actual_ids)]
    actual_wide = actual_top3.pivot(index=RACE_ID_COL, columns="着_int", values="枠_str")
    actual_wide = actual_wide.rename(columns={1: "actual_1", 2: "actual_2", 3: "actual_3"})
    actual_wide = actual_wide.reindex(columns=["actual_1", "actual_2", "actual_3"]).dropna()

    trifecta_df = pred_wide.merge(actual_wide, left_index=True, right_index=True, how="inner")
    race_meta = race_df.groupby(RACE_ID_COL, sort=False).agg(date=(DATE_COL, "min"), place=("place", "first"))
    trifecta_df = race_meta.join(trifecta_df, how="inner")

    trifecta_df["predicted_3rentan"] = (
        trifecta_df["pred_1"] + "-" + trifecta_df["pred_2"] + "-" + trifecta_df["pred_3"]
    )
    trifecta_df["actual_3rentan"] = (
        trifecta_df["actual_1"] + "-" + trifecta_df["actual_2"] + "-" + trifecta_df["actual_3"]
    )
    trifecta_df["hit"] = trifecta_df["predicted_3rentan"] == trifecta_df["actual_3rentan"]
    trifecta_df = trifecta_df.reset_index()

    races_evaluated = int(len(trifecta_df))
    hits = int(trifecta_df["hit"].sum()) if races_evaluated else 0
    hit_rate = float(hits / races_evaluated) if races_evaluated else None
    summary = {
        "races_evaluated": races_evaluated,
        "hits": hits,
        "hit_rate": hit_rate,
    }
    return trifecta_df, summary


def save_final_artifacts(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    out_dir: Path,
    log,
    seed: int,
    pred_sample_rows: int,
    plot_sample_rows: int,
    error_topn_place: int,
    odds_path: str | None = None,
    bet_stake_yen: int = 100,
    bet_min_odds: float | None = None,
    bet_max_odds: float | None = None,
    disable_betting_eval: bool = False,
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

    sample_n = min(pred_sample_rows, len(result_df))
    pred_sample = result_df.sample(n=sample_n, random_state=seed)
    pred_sample_out = pred_sample[[RACE_ID_COL, DATE_COL, TARGET_SEC_COL, "pred_sec", "abs_error"]]
    pred_sample_out.to_csv(out_dir / "predictions_sample.csv", index=False)
    log("INFO", "final", "wrote predictions_sample.csv", {"path": str(out_dir / "predictions_sample.csv"), "rows": int(sample_n)})

    month_key = result_df[DATE_COL].dt.to_period("M").astype(str)
    residuals_by_month = summarize_error_by_group(result_df.assign(month=month_key), "month")
    residuals_by_month.to_csv(out_dir / "residuals_by_month.csv", index=False)
    log("INFO", "final", "wrote residuals_by_month.csv", {"path": str(out_dir / "residuals_by_month.csv")})

    residuals_by_place = summarize_error_by_group(result_df, "place")
    residuals_by_place.to_csv(out_dir / "residuals_by_place.csv", index=False)
    log("INFO", "final", "wrote residuals_by_place.csv", {"path": str(out_dir / "residuals_by_place.csv")})

    quantiles = result_df["abs_error"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    quantiles = {f"p{int(k*100)}": float(v) for k, v in quantiles.items()}
    write_json(out_dir / "error_quantiles.json", quantiles)
    log("INFO", "final", "wrote error_quantiles.json", {"path": str(out_dir / "error_quantiles.json")})

    trifecta_summary = {
        "races_evaluated": 0,
        "hits": 0,
        "hit_rate": None,
        "skipped": True,
    }
    betting_summary = {
        "skipped": True,
        "reason": "trifecta_not_ready",
    }
    try:
        trifecta_df, trifecta_summary_base = build_trifecta_exact_hit_table(test_df, preds)
        trifecta_summary = {**trifecta_summary_base, "skipped": False}
        trifecta_df.to_csv(out_dir / "trifecta_predictions.csv", index=False)
        write_json(out_dir / "trifecta_hit_rate.json", trifecta_summary)
        log(
            "INFO",
            "final",
            "wrote trifecta predictions",
            {
                "path": str(out_dir / "trifecta_predictions.csv"),
                "races_evaluated": trifecta_summary["races_evaluated"],
            },
        )
        log(
            "INFO",
            "final",
            "wrote trifecta_hit_rate.json",
            {"path": str(out_dir / "trifecta_hit_rate.json"), "summary": trifecta_summary},
        )

        if disable_betting_eval:
            betting_summary = {
                "skipped": True,
                "reason": "disabled_by_flag",
            }
        elif odds_path is None:
            betting_summary = {
                "skipped": True,
                "reason": "odds_path_not_set",
            }
        else:
            try:
                summary = evaluate_trifecta_betting(
                    trifecta_df=trifecta_df,
                    out_dir=out_dir,
                    plots_dir=plots_dir,
                    log=log,
                    odds_path=odds_path,
                    stake_yen=bet_stake_yen,
                    min_odds=bet_min_odds,
                    max_odds=bet_max_odds,
                    enable_plot=True,
                )
                betting_summary = {**summary, "skipped": False}
            except Exception as exc:
                betting_summary = {
                    "skipped": True,
                    "reason": str(exc),
                }
                write_json(out_dir / "trifecta_recovery_summary.json", betting_summary)
                log("WARNING", "betting", "betting evaluation skipped", {"error": str(exc)})
    except Exception as exc:
        trifecta_summary = {**trifecta_summary, "reason": str(exc)}
        write_json(out_dir / "trifecta_hit_rate.json", trifecta_summary)
        log("WARNING", "final", "trifecta calculation skipped", {"error": str(exc)})

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

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

        plot_n = min(plot_sample_rows, len(result_df))
        plot_sample = result_df.sample(n=plot_n, random_state=seed)

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(plot_sample[TARGET_SEC_COL], plot_sample["pred_sec"], s=5, alpha=0.3)
        plt.xlabel("True (sec)")
        plt.ylabel("Pred (sec)")
        plt.title("Pred vs True (sample)")
        plt.tight_layout()
        fig.savefig(plots_dir / "pred_vs_true.png")
        plt.close(fig)

        fig = plt.figure(figsize=(6, 6))
        plt.hexbin(plot_sample[TARGET_SEC_COL], plot_sample["pred_sec"], gridsize=60, cmap="viridis")
        plt.xlabel("True (sec)")
        plt.ylabel("Pred (sec)")
        plt.title("Pred vs True Hexbin (sample)")
        plt.tight_layout()
        fig.savefig(plots_dir / "pred_vs_true_hexbin.png")
        plt.close(fig)

        fig = plt.figure(figsize=(8, 4))
        plt.hist(plot_sample["residual"], bins=100)
        plt.title("Residuals (pred - true)")
        plt.tight_layout()
        fig.savefig(plots_dir / "residual_hist.png")
        plt.close(fig)

        fig = plt.figure(figsize=(8, 4))
        plt.hist(plot_sample["abs_error"], bins=100)
        plt.title("Absolute Error")
        plt.tight_layout()
        fig.savefig(plots_dir / "abs_error_hist.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        plt.plot(residuals_by_month["month"], residuals_by_month["mean"])
        plt.xticks(rotation=90)
        plt.title("Residual Mean by Month")
        plt.tight_layout()
        fig.savefig(plots_dir / "residual_by_month.png")
        plt.close(fig)

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

        topn = residuals_by_place.sort_values("rmse", ascending=False).head(error_topn_place)
        fig = plt.figure(figsize=(10, 6))
        plt.bar(topn["place"].astype(str), topn["rmse"])
        plt.xticks(rotation=90)
        plt.title(f"RMSE by Place (Top {error_topn_place})")
        plt.tight_layout()
        fig.savefig(plots_dir / "error_by_place_topN.png")
        plt.close(fig)

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

        pred_bins = pd.qcut(plot_sample["pred_sec"], q=50, duplicates="drop")
        calib = plot_sample.groupby(pred_bins, observed=False)[TARGET_SEC_COL].mean()
        fig = plt.figure(figsize=(8, 4))
        plt.plot(range(len(calib)), calib.values)
        plt.title("Calibration-like Plot (Mean True by Pred Bin)")
        plt.tight_layout()
        fig.savefig(plots_dir / "calibration_like_plot.png")
        plt.close(fig)

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
    return {"trifecta": trifecta_summary, "trifecta_betting": betting_summary}


def save_core_artifacts(model, out_dir: Path, metrics: dict, features: list, config: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.txt"
    model.booster_.save_model(str(model_path))

    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, metrics)

    importance_split = model.booster_.feature_importance(importance_type="split")
    importance_gain = model.booster_.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame(
        {"feature": features, "importance_split": importance_split, "importance_gain": importance_gain}
    ).sort_values("importance_gain", ascending=False)
    fi_df.to_csv(out_dir / "feature_importance.csv", index=False)
    fi_df.head(100).to_csv(out_dir / "feature_importance_gain_top100.csv", index=False)

    config_path = out_dir / "config.json"
    write_json(config_path, config)
    return model_path, metrics_path, config_path


def build_run_config(args, out_dir: Path, features: list, categorical_features: list, params: dict) -> dict:
    return {
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
        "odds_path": args.odds_path,
        "bet_stake_yen": args.bet_stake_yen,
        "bet_min_odds": args.bet_min_odds,
        "bet_max_odds": args.bet_max_odds,
        "disable_betting_eval": args.disable_betting_eval,
        "features": features,
        "categorical_features": categorical_features,
        "params": params,
    }
