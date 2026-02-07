import argparse
import time
from datetime import datetime
from pathlib import Path

from racetime_lgbm.artifacts import (
    build_run_config,
    sample_rows,
    save_core_artifacts,
    save_final_artifacts,
    save_intermediate_artifacts,
    save_split_artifacts,
)
from racetime_lgbm.constants import (
    ERROR_TOPN_PLACE_DEFAULT,
    PLOT_SAMPLE_ROWS_DEFAULT,
    PRED_SAMPLE_ROWS_DEFAULT,
)
from racetime_lgbm.logging_utils import log_writer
from racetime_lgbm.pipeline import (
    assign_split_labels,
    build_feature_columns,
    build_lgbm_params,
    build_model_matrices,
    evaluate_model,
    load_raw_dataframe,
    prepare_training_dataframe,
    split_frames,
    train_lgbm_model,
)


def train(args):
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "logs" / f"run_{run_ts}.log"
    log, log_fp = log_writer(log_path)

    start_time = time.perf_counter()
    try:
        log("INFO", "run", "start", {"data_path": args.data_path, "output_dir": str(out_dir)})

        df = load_raw_dataframe(args.data_path, log)
        df, cat_cols_present = prepare_training_dataframe(df, args, log)
        save_intermediate_artifacts(df, out_dir, log)

        df = assign_split_labels(df, args.train_end, args.valid_end, log)
        save_split_artifacts(df, out_dir, log)
        sample_rows(df, out_dir, args.sample_seed, args.sample_rows, log)

        features = build_feature_columns(df, args.add_date_features)
        log("INFO", "features", "feature columns prepared", {"count": len(features)})

        splits = split_frames(df)
        x_train, y_train, x_valid, y_valid, x_test, y_test = build_model_matrices(splits, features)
        log("INFO", "split", "split sizes", {
            "train": int(len(splits.train)),
            "valid": int(len(splits.valid)),
            "test": int(len(splits.test)),
        })

        params = build_lgbm_params(args)
        categorical_features = [col for col in cat_cols_present if col in features]
        model = train_lgbm_model(
            params=params,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            categorical_features=categorical_features,
            early_stopping_rounds=args.early_stopping_rounds,
            log_every_n=args.log_every_n,
            log=log,
        )

        preds, metrics = evaluate_model(model, x_test, y_test, log)
        config = build_run_config(args, out_dir, features, categorical_features, params)
        model_path, metrics_path, config_path = save_core_artifacts(model, out_dir, metrics, features, config)

        final_summary = save_final_artifacts(
            test_df=splits.test,
            preds=preds,
            out_dir=out_dir,
            log=log,
            seed=args.sample_seed,
            pred_sample_rows=args.pred_sample_rows,
            plot_sample_rows=args.plot_sample_rows,
            error_topn_place=args.error_topn_place,
            odds_path=args.odds_path,
            bet_stake_yen=args.bet_stake_yen,
            bet_min_odds=args.bet_min_odds,
            bet_max_odds=args.bet_max_odds,
            disable_betting_eval=args.disable_betting_eval,
            evals_result=getattr(model, "evals_result_", None),
        )

        elapsed = time.perf_counter() - start_time
        log("INFO", "run", "completed", {
            "elapsed_sec": round(elapsed, 2),
            "metrics": metrics,
            "additional_metrics": final_summary,
            "outputs": {
                "model": str(model_path),
                "metrics": str(metrics_path),
                "feature_importance": str(out_dir / "feature_importance.csv"),
                "config": str(config_path),
                "log": str(log_path),
            },
        })
    finally:
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
    parser.add_argument("--odds-path", default="Data/ml_data/odds_tables_df_update.pickle", help="Path to 3連単オッズテーブル")
    parser.add_argument("--bet-stake-yen", type=int, default=100, help="Stake per 3連単 ticket (yen)")
    parser.add_argument("--bet-min-odds", type=float, default=None, help="Minimum odds filter for betting")
    parser.add_argument("--bet-max-odds", type=float, default=None, help="Maximum odds filter for betting")
    parser.add_argument("--disable-betting-eval", action="store_true", help="Disable 3連単 回収率/的中率 evaluation")

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
