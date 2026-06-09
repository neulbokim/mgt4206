from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "preprocessed" / "eda_gbg_master.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "time_series_tests"

TARGET = "visitors"
MAX_LAG = 7

WEIGHTS = {
    "spring": (2.58, 1.93, 3.04, 1.31, 1.14),
    "summer": (3.07, 1.90, 3.27, 0.90, 0.86),
    "autumn": (2.37, 2.18, 3.43, 1.04, 0.98),
    "winter": (2.27, 2.39, 3.07, 1.46, 0.81),
}
COMPONENTS = ["Cd", "Ca", "P", "W", "S"]
EVENT_COLS = ["event_day", "event_count", "night_event_count"]


PERIODS = {
    "2009-2025_full": ("2009-01-01", "2025-12-31"),
    "2015-2025_event_available": ("2015-01-01", "2025-12-31"),
    "2019-2025_recent_event": ("2019-01-01", "2025-12-31"),
}


BASE_CONTROLS = [
    "rain",
    "pm10",
    "covid_v3",
    "closed",
    "others_closed",
    "is_weekend",
    "d_holiday",
    "offdays_left",
    "long_break_3p",
    "pre_holiday",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["offday"] = ((out["is_weekend"] == 1) | (out["d_holiday"] == 1)).astype(int)
    out["next_day_off"] = out["offday"].shift(-1, fill_value=0).astype(int)
    out["pre_holiday"] = ((out["offday"] == 0) & (out["next_day_off"] == 1)).astype(int)

    group = out["offday"].ne(out["offday"].shift(fill_value=0)).cumsum()
    block_total = out.groupby(group)["offday"].transform("sum")
    position = out.groupby(group).cumcount() + 1
    out["offdays_left"] = np.where(
        out["offday"] == 1,
        block_total - position + 1,
        0,
    ).astype(int)
    out["long_break_3p"] = ((out["offday"] == 1) & (block_total >= 3)).astype(int)
    return out


def add_weighted_components(df: pd.DataFrame, season_col: str, suffix: str) -> pd.DataFrame:
    out = df.copy()
    weights = np.array([WEIGHTS[str(season)] for season in out[season_col]], dtype=float)
    for idx, component in enumerate(COMPONENTS):
        out[f"{component}_{suffix}"] = 2.0 * weights[:, idx] * out[component]
    out[f"ktci_{suffix}_sum"] = out[[f"{c}_{suffix}" for c in COMPONENTS]].sum(axis=1)
    return out


def load_model_frame() -> pd.DataFrame:
    df = pd.read_csv(MASTER_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index().asfreq("D")

    required = {
        TARGET,
        "tavg",
        "ktci",
        "ktci_a",
        "ktci_a_roll7",
        "season_month",
        "season_temp",
        *COMPONENTS,
        *EVENT_COLS,
        *BASE_CONTROLS[:7],
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        TARGET,
        "tavg",
        "ktci",
        "ktci_a",
        "ktci_a_roll7",
        *COMPONENTS,
        *EVENT_COLS,
        *BASE_CONTROLS[:7],
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    interpolate_cols = ["ktci", "ktci_a", "ktci_a_roll7", *COMPONENTS, "pm10"]
    df[interpolate_cols] = df[interpolate_cols].interpolate(method="time").ffill().bfill()
    fill_zero_cols = ["rain", "closed", "others_closed", "is_weekend", "d_holiday", "covid_v3", *EVENT_COLS]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)
    df["tavg"] = df["tavg"].interpolate(method="time").ffill().bfill()
    df[TARGET] = df[TARGET].fillna(0)

    df = add_calendar_features(df)
    df = add_weighted_components(df, "season_month", "month_w")
    df = add_weighted_components(df, "season_temp", "temp_w")
    return df


def model_specs() -> list[dict]:
    return [
        {
            "model": "Baseline_Temperature",
            "display": "Baseline (Temperature)",
            "predictors": ["tavg"],
            "controls": [*BASE_CONTROLS, *EVENT_COLS],
        },
        {
            "model": "KTCI",
            "display": "KTCI Model",
            "predictors": ["ktci"],
            "controls": [*BASE_CONTROLS, *EVENT_COLS],
        },
        {
            "model": "KTCI_a",
            "display": "KTCI-a Model",
            "predictors": ["ktci_a"],
            "controls": [*BASE_CONTROLS, *EVENT_COLS],
        },
        {
            "model": "KTCI_a_roll7",
            "display": "KTCI-a-roll7 Model",
            "predictors": ["ktci_a_roll7"],
            "controls": [*BASE_CONTROLS, *EVENT_COLS],
        },
        {
            "model": "Event_Block_given_KTCI_a_roll7",
            "display": "Event Block | given KTCI-a-roll7",
            "predictors": EVENT_COLS,
            "controls": [*BASE_CONTROLS, "ktci_a_roll7"],
        },
        {
            "model": "Decomposition_Reduced_Current",
            "display": "Decomposition Reduced | current notebook",
            "predictors": ["Cd", "Ca", "W", "S"],
            "controls": [*BASE_CONTROLS, *EVENT_COLS],
        },
        {
            "model": "Decomposition_Month_Weighted",
            "display": "Decomposition | month-season weighted",
            "predictors": [f"{c}_month_w" for c in COMPONENTS],
            "controls": [*BASE_CONTROLS, *EVENT_COLS],
        },
        {
            "model": "Decomposition_a_TempSeason_Weighted",
            "display": "Decomposition-a | temperature-season weighted",
            "predictors": [f"{c}_temp_w" for c in COMPONENTS],
            "controls": [*BASE_CONTROLS, *EVENT_COLS],
        },
    ]


def clean_columns(cols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def lagged_matrix(frame: pd.DataFrame, columns: list[str], lag: int, prefix: str) -> pd.DataFrame:
    data = {}
    for col in columns:
        for k in range(1, lag + 1):
            data[f"{prefix}{col}_lag{k}"] = frame[col].shift(k)
    return pd.DataFrame(data, index=frame.index)


def fit_sse(y: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    coef, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    return float(resid.T @ resid), int(rank)


def conditional_granger(frame: pd.DataFrame, predictors: list[str], controls: list[str], lag: int) -> dict:
    predictors = clean_columns(predictors)
    controls = [c for c in clean_columns(controls) if c not in predictors and c != TARGET]

    y = frame[[TARGET]]
    target_lags = lagged_matrix(frame, [TARGET], lag, "ar_")
    predictor_lags = lagged_matrix(frame, predictors, lag, "x_")
    control_now = frame[controls] if controls else pd.DataFrame(index=frame.index)

    restricted = pd.concat([y, target_lags, control_now], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    unrestricted = pd.concat([y, target_lags, control_now, predictor_lags], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    common_index = restricted.index.intersection(unrestricted.index)
    restricted = restricted.loc[common_index]
    unrestricted = unrestricted.loc[common_index]

    y_arr = unrestricted[TARGET].to_numpy(dtype=float)
    xr = np.column_stack([np.ones(len(restricted)), restricted.drop(columns=[TARGET]).to_numpy(dtype=float)])
    xu = np.column_stack([np.ones(len(unrestricted)), unrestricted.drop(columns=[TARGET]).to_numpy(dtype=float)])

    sse_r, rank_r = fit_sse(y_arr, xr)
    sse_u, rank_u = fit_sse(y_arr, xu)
    df_num = rank_u - rank_r
    df_den = len(y_arr) - rank_u
    if df_num <= 0 or df_den <= 0 or sse_u <= 0:
        f_stat = np.nan
        p_value = np.nan
    else:
        f_stat = max((sse_r - sse_u) / df_num, 0.0) / (sse_u / df_den)
        p_value = float(stats.f.sf(f_stat, df_num, df_den))

    return {
        "lag": lag,
        "n_obs": len(y_arr),
        "df_num": df_num,
        "df_den": df_den,
        "sse_restricted": sse_r,
        "sse_unrestricted": sse_u,
        "f_stat": f_stat,
        "p_value": p_value,
    }


def run_adf(frame: pd.DataFrame, period_name: str, columns: list[str]) -> list[dict]:
    rows = []
    for col in clean_columns(columns):
        series = frame[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.nunique() <= 1:
            rows.append({
                "period": period_name,
                "series": col,
                "n_obs": len(series),
                "adf_stat": np.nan,
                "p_value": np.nan,
                "used_lag": np.nan,
                "stationarity_call": "constant",
            })
            continue
        stat, p_value, used_lag, nobs, crit, _ = adfuller(series, autolag="AIC")
        rows.append({
            "period": period_name,
            "series": col,
            "n_obs": nobs,
            "adf_stat": stat,
            "p_value": p_value,
            "used_lag": used_lag,
            "crit_1pct": crit["1%"],
            "crit_5pct": crit["5%"],
            "crit_10pct": crit["10%"],
            "stationarity_call": "stationary_I0" if p_value < 0.05 else "unit_root_not_rejected",
        })
    return rows


def run_cointegration(frame: pd.DataFrame, spec: dict, period_name: str) -> dict:
    cols = [TARGET, *spec["predictors"]]
    work = frame[cols].replace([np.inf, -np.inf], np.nan).dropna()
    nonconstant_predictors = [col for col in spec["predictors"] if work[col].nunique() > 1]
    if len(nonconstant_predictors) == 0 or work[TARGET].nunique() <= 1:
        return {
            "period": period_name,
            "model": spec["model"],
            "display": spec["display"],
            "predictors": ", ".join(spec["predictors"]),
            "n_obs": len(work),
            "coint_t": np.nan,
            "p_value": np.nan,
            "crit_1pct": np.nan,
            "crit_5pct": np.nan,
            "crit_10pct": np.nan,
            "interpretation_guardrail": "not_applicable_constant_series",
        }

    y0 = work[TARGET].to_numpy(dtype=float)
    y1 = work[nonconstant_predictors].to_numpy(dtype=float)
    try:
        coint_t, p_value, crit = coint(y0, y1, trend="c", autolag="AIC")
    except Exception as exc:
        return {
            "period": period_name,
            "model": spec["model"],
            "display": spec["display"],
            "predictors": ", ".join(nonconstant_predictors),
            "n_obs": len(work),
            "coint_t": np.nan,
            "p_value": np.nan,
            "crit_1pct": np.nan,
            "crit_5pct": np.nan,
            "crit_10pct": np.nan,
            "interpretation_guardrail": f"failed: {exc}",
        }

    return {
        "period": period_name,
        "model": spec["model"],
        "display": spec["display"],
        "predictors": ", ".join(nonconstant_predictors),
        "n_obs": len(work),
        "coint_t": coint_t,
        "p_value": p_value,
        "crit_1pct": crit[0],
        "crit_5pct": crit[1],
        "crit_10pct": crit[2],
        "interpretation_guardrail": "cointegration_is_secondary_if_level_series_are_I0",
    }


def format_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_model_frame()
    specs = model_specs()

    all_model_cols = [TARGET]
    for spec in specs:
        all_model_cols.extend(spec["predictors"])
        all_model_cols.extend(spec["controls"])

    adf_rows = []
    granger_rows = []
    coint_rows = []

    for period_name, (start, end) in PERIODS.items():
        period_df = df.loc[start:end].copy()
        adf_rows.extend(run_adf(period_df, period_name, all_model_cols))

        for spec in specs:
            for lag in range(1, MAX_LAG + 1):
                row = conditional_granger(period_df, spec["predictors"], spec["controls"], lag)
                row.update({
                    "period": period_name,
                    "model": spec["model"],
                    "display": spec["display"],
                    "predictors": ", ".join(spec["predictors"]),
                    "controls": ", ".join(clean_columns(spec["controls"])),
                })
                granger_rows.append(row)
            coint_rows.append(run_cointegration(period_df, spec, period_name))

    adf_df = pd.DataFrame(adf_rows)
    granger_lag_df = pd.DataFrame(granger_rows)
    coint_df = pd.DataFrame(coint_rows)

    summary_rows = []
    for keys, group in granger_lag_df.groupby(["period", "model", "display", "predictors"], sort=False):
        best = group.loc[group["p_value"].idxmin()]
        lag7 = group[group["lag"] == MAX_LAG].iloc[0]
        sig_lags = group.loc[group["p_value"] < 0.05, "lag"].astype(str).tolist()
        summary_rows.append({
            "period": keys[0],
            "model": keys[1],
            "display": keys[2],
            "predictors": keys[3],
            "best_lag": int(best["lag"]),
            "best_f_stat": best["f_stat"],
            "best_p_value": best["p_value"],
            "lag7_f_stat": lag7["f_stat"],
            "lag7_p_value": lag7["p_value"],
            "significant_lags_p05": ", ".join(sig_lags) if sig_lags else "",
            "n_obs_at_best_lag": int(best["n_obs"]),
            "decision_best_p05": "reject_no_granger" if best["p_value"] < 0.05 else "not_reject",
        })
    granger_summary_df = pd.DataFrame(summary_rows)

    granger_summary_df["best_q_fdr"] = multipletests(
        granger_summary_df["best_p_value"].fillna(1.0),
        alpha=0.05,
        method="fdr_bh",
    )[1]
    granger_summary_df["decision_best_q05"] = np.where(
        granger_summary_df["best_q_fdr"] < 0.05,
        "reject_no_granger_after_fdr",
        "not_reject_after_fdr",
    )

    coint_df["q_fdr"] = multipletests(
        coint_df["p_value"].fillna(1.0),
        alpha=0.05,
        method="fdr_bh",
    )[1]
    coint_df["decision_p05"] = np.where(coint_df["p_value"] < 0.05, "reject_no_cointegration", "not_reject")
    coint_df["decision_q05"] = np.where(coint_df["q_fdr"] < 0.05, "reject_no_cointegration_after_fdr", "not_reject_after_fdr")

    adf_df.to_csv(OUTPUT_DIR / "adf_results.csv", index=False, encoding="utf-8-sig")
    granger_lag_df.to_csv(OUTPUT_DIR / "granger_lag_results.csv", index=False, encoding="utf-8-sig")
    granger_summary_df.to_csv(OUTPUT_DIR / "granger_summary.csv", index=False, encoding="utf-8-sig")
    coint_df.to_csv(OUTPUT_DIR / "cointegration_results.csv", index=False, encoding="utf-8-sig")

    print("Saved outputs to", OUTPUT_DIR)
    print("\n=== Granger summary: best lag within 1-7 days ===")
    view = granger_summary_df[[
        "period",
        "display",
        "best_lag",
        "best_f_stat",
        "best_p_value",
        "best_q_fdr",
        "significant_lags_p05",
        "decision_best_q05",
    ]].copy()
    view["best_f_stat"] = view["best_f_stat"].round(3)
    view["best_p_value"] = view["best_p_value"].map(format_p)
    view["best_q_fdr"] = view["best_q_fdr"].map(format_p)
    print(view.to_string(index=False))

    print("\n=== Cointegration summary ===")
    cview = coint_df[[
        "period",
        "display",
        "coint_t",
        "p_value",
        "q_fdr",
        "decision_q05",
        "interpretation_guardrail",
    ]].copy()
    cview["coint_t"] = cview["coint_t"].round(3)
    cview["p_value"] = cview["p_value"].map(format_p)
    cview["q_fdr"] = cview["q_fdr"].map(format_p)
    print(cview.to_string(index=False))

    print("\n=== ADF stationarity count ===")
    print(adf_df.groupby(["period", "stationarity_call"]).size().to_string())


if __name__ == "__main__":
    main()
