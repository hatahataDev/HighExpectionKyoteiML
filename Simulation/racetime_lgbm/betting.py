import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DATE_COL, RACE_ID_COL


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_odds_table(odds_path: str, log) -> pd.DataFrame:
    path = Path(odds_path)
    if not path.exists():
        raise FileNotFoundError(f"odds file not found: {path}")
    log("INFO", "betting", "loading odds table", {"path": str(path)})
    odds_df = pd.read_pickle(path)
    if not isinstance(odds_df, pd.DataFrame):
        raise TypeError(f"odds data must be DataFrame, got {type(odds_df)}")
    odds_df.index = odds_df.index.astype(str)
    log("INFO", "betting", "loaded odds table", {"rows": int(len(odds_df)), "columns": int(len(odds_df.columns))})
    return odds_df


def attach_predicted_trifecta_odds(trifecta_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    df = trifecta_df.copy()
    df[RACE_ID_COL] = df[RACE_ID_COL].astype(str)
    df["predicted_3rentan"] = df["predicted_3rentan"].astype(str)

    odds_subset = odds_df.reindex(df[RACE_ID_COL])
    col_to_idx = {col: idx for idx, col in enumerate(odds_subset.columns)}
    col_idx = df["predicted_3rentan"].map(col_to_idx)

    odds_array = odds_subset.to_numpy(dtype=object, copy=False)
    selected_odds = np.full(len(df), np.nan, dtype=object)
    valid = col_idx.notna().to_numpy()
    if valid.any():
        row_idx = np.arange(len(df))[valid]
        selected_odds[valid] = odds_array[row_idx, col_idx[valid].astype(int).to_numpy()]

    df["odds_raw"] = selected_odds
    df["odds"] = pd.to_numeric(df["odds_raw"], errors="coerce")
    return df


def apply_trifecta_bet_rules(
    bet_df: pd.DataFrame,
    stake_yen: int,
    min_odds: float | None = None,
    max_odds: float | None = None,
) -> pd.DataFrame:
    df = bet_df.copy()
    df["skip_reason"] = ""
    df["is_bet"] = True

    invalid_odds = df["odds"].isna() | (df["odds"] <= 0)
    df.loc[invalid_odds, "is_bet"] = False
    df.loc[invalid_odds, "skip_reason"] = "invalid_odds"

    if min_odds is not None:
        min_mask = df["is_bet"] & df["odds"].lt(min_odds)
        df.loc[min_mask, "is_bet"] = False
        df.loc[min_mask, "skip_reason"] = "below_min_odds"

    if max_odds is not None:
        max_mask = df["is_bet"] & df["odds"].gt(max_odds)
        df.loc[max_mask, "is_bet"] = False
        df.loc[max_mask, "skip_reason"] = "above_max_odds"

    stake = float(stake_yen)
    df["stake_yen"] = np.where(df["is_bet"], stake, 0.0)
    df["payout_yen"] = np.where(df["is_bet"] & df["hit"], stake * df["odds"], 0.0)
    df["profit_yen"] = df["payout_yen"] - df["stake_yen"]
    return df


def summarize_betting_overall(bet_df: pd.DataFrame) -> dict:
    races_total = int(len(bet_df))
    bets = int(bet_df["is_bet"].sum())
    hits = int((bet_df["is_bet"] & bet_df["hit"]).sum())
    total_bet = float(bet_df["stake_yen"].sum())
    total_return = float(bet_df["payout_yen"].sum())
    profit = float(total_return - total_bet)
    hit_rate = float(hits / bets) if bets else None
    recovery_rate = float(total_return / total_bet) if total_bet > 0 else None
    return {
        "races_total": races_total,
        "bets": bets,
        "hits": hits,
        "hit_rate": hit_rate,
        "total_bet_yen": total_bet,
        "total_return_yen": total_return,
        "profit_yen": profit,
        "recovery_rate": recovery_rate,
    }


def summarize_betting_by_group(bet_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    use_df = bet_df[bet_df["is_bet"]].copy()
    if use_df.empty:
        return pd.DataFrame(columns=[group_col, "bets", "hits", "hit_rate", "total_bet_yen", "total_return_yen", "profit_yen", "recovery_rate"])

    grouped = (
        use_df.groupby(group_col, observed=False)
        .agg(
            bets=("is_bet", "size"),
            hits=("hit", "sum"),
            total_bet_yen=("stake_yen", "sum"),
            total_return_yen=("payout_yen", "sum"),
        )
        .reset_index()
    )
    grouped["profit_yen"] = grouped["total_return_yen"] - grouped["total_bet_yen"]
    grouped["hit_rate"] = grouped["hits"] / grouped["bets"]
    grouped["recovery_rate"] = grouped["total_return_yen"] / grouped["total_bet_yen"]
    return grouped[[group_col, "bets", "hits", "hit_rate", "total_bet_yen", "total_return_yen", "profit_yen", "recovery_rate"]]


def _save_betting_plots(bet_df: pd.DataFrame, by_month: pd.DataFrame, by_place: pd.DataFrame, plots_dir: Path, log):
    if by_month.empty:
        log("WARNING", "betting", "plot generation skipped", {"error": "no bets to plot"})
        return

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        month_df = by_month.copy()
        month_df["month_dt"] = pd.to_datetime(month_df["month"] + "-01", errors="coerce")
        month_df = month_df.sort_values("month_dt")

        fig = plt.figure(figsize=(10, 4))
        plt.plot(month_df["month"], month_df["recovery_rate"], marker="o")
        plt.xticks(rotation=90)
        plt.title("Trifecta Recovery Rate by Month")
        plt.tight_layout()
        fig.savefig(plots_dir / "trifecta_recovery_rate_by_month.png")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        plt.plot(month_df["month"], month_df["hit_rate"], marker="o")
        plt.xticks(rotation=90)
        plt.title("Trifecta Hit Rate by Month")
        plt.tight_layout()
        fig.savefig(plots_dir / "trifecta_hit_rate_by_month.png")
        plt.close(fig)

        cum = bet_df[bet_df["is_bet"]].copy()
        cum[DATE_COL] = pd.to_datetime(cum[DATE_COL], errors="coerce")
        cum = cum.sort_values(DATE_COL)
        cum["cum_profit_yen"] = cum["profit_yen"].cumsum()
        fig = plt.figure(figsize=(10, 4))
        plt.plot(cum[DATE_COL], cum["cum_profit_yen"])
        plt.title("Trifecta Cumulative Profit")
        plt.tight_layout()
        fig.savefig(plots_dir / "trifecta_cumulative_profit.png")
        plt.close(fig)

        if not by_place.empty:
            top_place = by_place.sort_values("bets", ascending=False).head(20)
            fig = plt.figure(figsize=(10, 6))
            plt.bar(top_place["place"].astype(str), top_place["recovery_rate"])
            plt.xticks(rotation=90)
            plt.title("Trifecta Recovery Rate by Place (Top 20 by Bets)")
            plt.tight_layout()
            fig.savefig(plots_dir / "trifecta_recovery_rate_by_place_top20.png")
            plt.close(fig)

        log("INFO", "betting", "wrote betting plots", {"path": str(plots_dir)})
    except Exception as exc:
        log("WARNING", "betting", "plot generation skipped", {"error": str(exc)})


def evaluate_trifecta_betting(
    trifecta_df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
    log,
    odds_path: str,
    stake_yen: int,
    min_odds: float | None = None,
    max_odds: float | None = None,
    enable_plot: bool = True,
) -> dict:
    odds_df = load_odds_table(odds_path, log)
    try:
        bet_df = attach_predicted_trifecta_odds(trifecta_df, odds_df)
    finally:
        del odds_df
        gc.collect()

    bet_df = apply_trifecta_bet_rules(
        bet_df=bet_df,
        stake_yen=stake_yen,
        min_odds=min_odds,
        max_odds=max_odds,
    )

    by_month_input = bet_df.copy()
    by_month_input["month"] = pd.to_datetime(by_month_input[DATE_COL], errors="coerce").dt.to_period("M").astype(str)
    by_month = summarize_betting_by_group(by_month_input, "month")
    by_place = summarize_betting_by_group(bet_df, "place")
    summary = summarize_betting_overall(bet_df)
    summary["odds_path"] = str(odds_path)
    summary["stake_yen"] = int(stake_yen)
    summary["min_odds"] = min_odds
    summary["max_odds"] = max_odds

    detail_cols = [
        RACE_ID_COL,
        DATE_COL,
        "place",
        "predicted_3rentan",
        "actual_3rentan",
        "hit",
        "odds_raw",
        "odds",
        "is_bet",
        "skip_reason",
        "stake_yen",
        "payout_yen",
        "profit_yen",
    ]
    bet_df[detail_cols].to_csv(out_dir / "trifecta_bet_detail.csv", index=False)
    _write_json(out_dir / "trifecta_recovery_summary.json", summary)
    by_month.to_csv(out_dir / "trifecta_recovery_by_month.csv", index=False)
    by_place.to_csv(out_dir / "trifecta_recovery_by_place.csv", index=False)
    log("INFO", "betting", "wrote trifecta_bet_detail.csv", {"path": str(out_dir / "trifecta_bet_detail.csv"), "rows": int(len(bet_df))})
    log("INFO", "betting", "wrote trifecta_recovery_summary.json", {"path": str(out_dir / "trifecta_recovery_summary.json"), "summary": summary})
    log("INFO", "betting", "wrote trifecta_recovery_by_month.csv", {"path": str(out_dir / "trifecta_recovery_by_month.csv")})
    log("INFO", "betting", "wrote trifecta_recovery_by_place.csv", {"path": str(out_dir / "trifecta_recovery_by_place.csv")})

    if enable_plot:
        _save_betting_plots(bet_df=bet_df, by_month=by_month, by_place=by_place, plots_dir=plots_dir, log=log)
    return summary
