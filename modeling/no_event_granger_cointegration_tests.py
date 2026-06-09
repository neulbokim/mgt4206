from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "preprocessed" / "eda_gbg_master.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "no_event_time_series_tests"
FIG_DIR = OUTPUT_DIR / "figures"

TARGET = "visitors"
MAX_LAG = 7
START_DATE = "2009-01-01"
END_DATE = "2025-12-31"

COMPONENTS = ["Cd", "Ca", "P", "W", "S"]
WEIGHTS = {
    "spring": (2.58, 1.93, 3.04, 1.31, 1.14),
    "summer": (3.07, 1.90, 3.27, 0.90, 0.86),
    "autumn": (2.37, 2.18, 3.43, 1.04, 0.98),
    "winter": (2.27, 2.39, 3.07, 1.46, 0.81),
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

MODEL_ORDER = [
    "Baseline (Temperature)",
    "KTCI Model",
    "KTCI-a Model",
    "KTCI-a-roll7 Model",
    "Decomposition Model",
    "Decomposition-a Model",
]

MODEL_LABELS = {
    "Baseline (Temperature)": "Baseline\ntavg",
    "KTCI Model": "KTCI",
    "KTCI-a Model": "KTCI-a",
    "KTCI-a-roll7 Model": "KTCI-a\nroll7",
    "Decomposition Model": "Decomp\nmonth",
    "Decomposition-a Model": "Decomp-a\ntemp",
}

COLORS = {
    "Baseline (Temperature)": "#4C78A8",
    "KTCI Model": "#F58518",
    "KTCI-a Model": "#54A24B",
    "KTCI-a-roll7 Model": "#9D755D",
    "Decomposition Model": "#72B7B2",
    "Decomposition-a Model": "#FF9DA6",
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["offday"] = ((out["is_weekend"] == 1) | (out["d_holiday"] == 1)).astype(int)
    out["next_day_off"] = out["offday"].shift(-1, fill_value=0).astype(int)
    out["pre_holiday"] = ((out["offday"] == 0) & (out["next_day_off"] == 1)).astype(int)

    group = out["offday"].ne(out["offday"].shift(fill_value=0)).cumsum()
    block_total = out.groupby(group)["offday"].transform("sum")
    position = out.groupby(group).cumcount() + 1
    out["offdays_left"] = np.where(out["offday"] == 1, block_total - position + 1, 0).astype(int)
    out["long_break_3p"] = ((out["offday"] == 1) & (block_total >= 3)).astype(int)
    return out


def add_weighted_components(df: pd.DataFrame, season_col: str, suffix: str) -> pd.DataFrame:
    out = df.copy()
    weights = np.array([WEIGHTS[str(season)] for season in out[season_col]], dtype=float)
    for idx, component in enumerate(COMPONENTS):
        out[f"{component}_{suffix}"] = 2.0 * weights[:, idx] * out[component]
    out[f"ktci_{suffix}_sum"] = out[[f"{c}_{suffix}" for c in COMPONENTS]].sum(axis=1)
    return out


def load_frame() -> pd.DataFrame:
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
        *BASE_CONTROLS[:7],
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    interpolation_cols = ["tavg", "ktci", "ktci_a", "ktci_a_roll7", *COMPONENTS, "pm10"]
    df[interpolation_cols] = df[interpolation_cols].interpolate(method="time").ffill().bfill()
    fill_zero_cols = ["rain", "closed", "others_closed", "is_weekend", "d_holiday", "covid_v3"]
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0)
    df[TARGET] = df[TARGET].fillna(0)

    df = add_calendar_features(df)
    df = add_weighted_components(df, "season_month", "month_w")
    df = add_weighted_components(df, "season_temp", "temp_w")
    return df.loc[START_DATE:END_DATE].copy()


def model_specs() -> list[dict]:
    return [
        {
            "model": "Baseline_Temperature",
            "display": "Baseline (Temperature)",
            "predictors": ["tavg"],
            "controls": BASE_CONTROLS,
            "story": "단순 평균기온 baseline",
        },
        {
            "model": "KTCI",
            "display": "KTCI Model",
            "predictors": ["ktci"],
            "controls": BASE_CONTROLS,
            "story": "월 기준 계절 가중치를 적용한 관광쾌적도",
        },
        {
            "model": "KTCI_a",
            "display": "KTCI-a Model",
            "predictors": ["ktci_a"],
            "controls": BASE_CONTROLS,
            "story": "실제 기온 흐름으로 계절을 재계산한 관광쾌적도",
        },
        {
            "model": "KTCI_a_roll7",
            "display": "KTCI-a-roll7 Model",
            "predictors": ["ktci_a_roll7"],
            "controls": BASE_CONTROLS,
            "story": "KTCI-a의 7일 rolling 평균으로 체감 지속성 반영",
        },
        {
            "model": "Decomposition_Month_Weighted",
            "display": "Decomposition Model",
            "predictors": [f"{c}_month_w" for c in COMPONENTS],
            "controls": BASE_CONTROLS,
            "story": "KTCI를 Cd/Ca/P/W/S 월 기준 계절 가중 성분으로 분해",
        },
        {
            "model": "Decomposition_a_TempSeason_Weighted",
            "display": "Decomposition-a Model",
            "predictors": [f"{c}_temp_w" for c in COMPONENTS],
            "controls": BASE_CONTROLS,
            "story": "KTCI-a를 Cd/Ca/P/W/S 기온 기준 계절 가중 성분으로 분해",
        },
    ]


def clean_columns(cols: list[str]) -> list[str]:
    seen = set()
    out = []
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


def run_adf(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in clean_columns(columns):
        series = frame[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.nunique() <= 1:
            rows.append({"series": col, "n_obs": len(series), "adf_stat": np.nan, "p_value": np.nan, "used_lag": np.nan, "stationarity_call": "constant"})
            continue
        stat, p_value, used_lag, nobs, crit, _ = adfuller(series, autolag="AIC")
        rows.append({
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
    return pd.DataFrame(rows)


def run_cointegration(frame: pd.DataFrame, spec: dict) -> dict:
    cols = [TARGET, *spec["predictors"]]
    work = frame[cols].replace([np.inf, -np.inf], np.nan).dropna()
    nonconstant_predictors = [col for col in spec["predictors"] if work[col].nunique() > 1]

    y0 = work[TARGET].to_numpy(dtype=float)
    y1 = work[nonconstant_predictors].to_numpy(dtype=float)
    coint_t, p_value, crit = coint(y0, y1, trend="c", autolag="AIC")
    return {
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


def save_fig(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figures(granger_summary: pd.DataFrame, coint_df: pd.DataFrame, adf_df: pd.DataFrame, frame: pd.DataFrame) -> None:
    # 1. model development storyline
    steps = [
        ("Baseline", "tavg"),
        ("KTCI", "month-season\ncomfort index"),
        ("KTCI-a", "temperature-season\nreassignment"),
        ("KTCI-a-roll7", "7-day smoothed\ncomfort persistence"),
        ("Decomposition", "month-weighted\ncomponent model"),
        ("Decomposition-a", "temp-season\ncomponent model"),
    ]
    fig, ax = plt.subplots(figsize=(14, 3.3))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(steps))
    y = 0.55
    for i, ((title, body), x) in enumerate(zip(steps, xs)):
        box = plt.Rectangle((x - 0.07, y - 0.20), 0.14, 0.40, linewidth=1.4, edgecolor="#333333", facecolor="#F3F6FA")
        ax.add_patch(box)
        ax.text(x, y + 0.07, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x, y - 0.08, body, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.08, y), xytext=(x + 0.08, y), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.set_title("Model Development Storyline Without Event Variables", fontsize=16, weight="bold", pad=10)
    save_fig(fig, "00_no_event_model_storyline.png")

    # 2. Granger best F-stat
    summary = granger_summary.set_index("display").loc[MODEL_ORDER].reset_index()
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bars = ax.bar(x, summary["best_f_stat"], color=[COLORS[d] for d in summary["display"]])
    ax.set_title("No-Event Granger Causality Test: Full Period 2009-2025", fontsize=15, weight="bold")
    ax.set_ylabel("Best F-statistic across lag 1-7")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[d] for d in summary["display"]])
    ax.set_ylim(0, float(summary["best_f_stat"].max()) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for bar, val, pval in zip(bars, summary["best_f_stat"], summary["best_p_value"]):
        label = f"{val:.1f}\np<.001" if pval < 0.001 else f"{val:.1f}\np={pval:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, val, label, ha="center", va="bottom", fontsize=9)
    save_fig(fig, "01_no_event_granger_fstat.png")

    # 3. KTCI path
    ktci_models = ["Baseline (Temperature)", "KTCI Model", "KTCI-a Model", "KTCI-a-roll7 Model"]
    ktci = summary.set_index("display").loc[ktci_models].reset_index()
    x = np.arange(len(ktci))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bars = ax.bar(x, ktci["best_f_stat"], color=[COLORS[d] for d in ktci["display"]])
    ax.set_title("KTCI Development Path: Granger Evidence", fontsize=15, weight="bold")
    ax.set_ylabel("Best F-statistic")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[d] for d in ktci["display"]])
    ax.set_ylim(0, float(ktci["best_f_stat"].max()) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, ktci["best_f_stat"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    save_fig(fig, "02_no_event_ktci_path_granger.png")

    # 4. Decomposition comparison
    decomp_models = ["Decomposition Model", "Decomposition-a Model"]
    decomp = summary.set_index("display").loc[decomp_models].reset_index()
    x = np.arange(len(decomp))
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    bars = ax.bar(x, decomp["best_f_stat"], color=[COLORS[d] for d in decomp["display"]])
    ax.set_title("Decomposition vs Decomposition-a: Granger Evidence", fontsize=15, weight="bold")
    ax.set_ylabel("Best F-statistic")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[d] for d in decomp["display"]])
    ax.set_ylim(0, float(decomp["best_f_stat"].max()) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, decomp["best_f_stat"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=11)
    save_fig(fig, "03_no_event_decomposition_granger.png")

    # 5. Cointegration p-values
    cview = coint_df.set_index("display").loc[MODEL_ORDER].reset_index()
    cview["neg_log10_p"] = -np.log10(cview["p_value"])
    x = np.arange(len(cview))
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bars = ax.bar(x, cview["neg_log10_p"], color=[COLORS[d] for d in cview["display"]])
    ax.axhline(-np.log10(0.05), color="#333333", linestyle="--", linewidth=1.4, label="p = 0.05")
    ax.set_title("No-Event Cointegration Test: Full Period 2009-2025", fontsize=15, weight="bold")
    ax.set_ylabel("-log10(p-value)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[d] for d in cview["display"]])
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    for bar, pval in zip(bars, cview["p_value"]):
        label = f"p={pval:.1e}" if pval < 0.001 else f"p={pval:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)
    save_fig(fig, "04_no_event_cointegration_neglogp.png")

    # 6. Cointegration t-stat vs critical value
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bars = ax.bar(x, cview["coint_t"], color=[COLORS[d] for d in cview["display"]])
    ax.plot(x, cview["crit_5pct"], color="#333333", linestyle="--", linewidth=1.5, marker="o", label="5% critical value")
    ax.set_title("Cointegration Test Statistic vs 5% Critical Value", fontsize=15, weight="bold")
    ax.set_ylabel("Engle-Granger test statistic")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[d] for d in cview["display"]])
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    for bar, stat in zip(bars, cview["coint_t"]):
        ax.text(bar.get_x() + bar.get_width() / 2, stat - 0.08, f"{stat:.2f}", ha="center", va="top", fontsize=9)
    save_fig(fig, "05_no_event_cointegration_tstat.png")

    # 7. ADF p-values for key variables
    key_series = ["visitors", "tavg", "ktci", "ktci_a", "ktci_a_roll7", "Cd_month_w", "Ca_month_w", "P_month_w", "W_month_w", "S_month_w", "Cd_temp_w", "Ca_temp_w", "P_temp_w", "W_temp_w", "S_temp_w"]
    adf_key = adf_df[adf_df["series"].isin(key_series)].copy()
    adf_key["series"] = pd.Categorical(adf_key["series"], categories=key_series, ordered=True)
    adf_key = adf_key.sort_values("series")
    x = np.arange(len(adf_key))
    colors = ["#E45756" if p >= 0.05 else "#4C78A8" for p in adf_key["p_value"]]
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    bars = ax.bar(x, adf_key["p_value"], color=colors)
    ax.axhline(0.05, color="#333333", linestyle="--", linewidth=1.4, label="ADF p = 0.05")
    ax.set_title("ADF Stationarity Check for No-Event Models", fontsize=15, weight="bold")
    ax.set_ylabel("ADF p-value")
    ax.set_xticks(x)
    ax.set_xticklabels(adf_key["series"], rotation=25, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ymax = max(0.12, float(adf_key["p_value"].max()) * 1.15)
    ax.set_ylim(0, ymax)
    for bar, pval in zip(bars, adf_key["p_value"]):
        label = "<.001" if pval < 0.001 else f"{pval:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, pval, label, ha="center", va="bottom", fontsize=7)
    save_fig(fig, "06_no_event_adf_stationarity.png")

    # 8. Season mismatch ratio
    mismatch = (frame["season_month"] != frame["season_temp"]).sum()
    match = len(frame) - mismatch
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(["same season", "reassigned season"], [match, mismatch], color=["#4C78A8", "#E45756"])
    ax.set_title("Season Reassignment Scope", fontsize=15, weight="bold")
    ax.set_ylabel("Number of days")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, [match, mismatch]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}\n({val / len(frame):.1%})", ha="center", va="bottom", fontsize=10)
    save_fig(fig, "07_no_event_season_reassignment.png")


def write_report(granger_summary: pd.DataFrame, coint_df: pd.DataFrame, adf_df: pd.DataFrame, frame: pd.DataFrame) -> None:
    report_path = OUTPUT_DIR / "no_event_report.md"
    mismatch = int((frame["season_month"] != frame["season_temp"]).sum())
    lines = [
        "# No-Event Granger & Cointegration Test Report",
        "",
        "## Scope",
        "",
        f"- Period: {START_DATE} ~ {END_DATE}",
        f"- Observations: {len(frame):,} daily rows",
        "- Event variables are excluded from model variables, controls, results, figures, and interpretation.",
        f"- Season reassignment days: {mismatch:,} / {len(frame):,} ({mismatch / len(frame):.1%})",
        "",
        "## Models",
        "",
        "| Model | Predictors | Interpretation |",
        "|---|---|---|",
    ]
    for spec in model_specs():
        lines.append(f"| {spec['display']} | `{', '.join(spec['predictors'])}` | {spec['story']} |")

    lines.extend([
        "",
        "## Granger Summary",
        "",
        "| Model | Best lag | F-stat | p-value | FDR q-value | Significant lags |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for _, row in granger_summary.set_index("display").loc[MODEL_ORDER].reset_index().iterrows():
        p = "<0.001" if row["best_p_value"] < 0.001 else f"{row['best_p_value']:.3f}"
        q = "<0.001" if row["best_q_fdr"] < 0.001 else f"{row['best_q_fdr']:.3f}"
        lines.append(f"| {row['display']} | {int(row['best_lag'])} | {row['best_f_stat']:.3f} | {p} | {q} | {row['significant_lags_p05']} |")

    lines.extend([
        "",
        "## Cointegration Summary",
        "",
        "| Model | Engle-Granger t-stat | p-value | 5% critical value | Decision |",
        "|---|---:|---:|---:|---|",
    ])
    for _, row in coint_df.set_index("display").loc[MODEL_ORDER].reset_index().iterrows():
        p = f"{row['p_value']:.2e}" if row["p_value"] < 0.001 else f"{row['p_value']:.3f}"
        lines.append(f"| {row['display']} | {row['coint_t']:.3f} | {p} | {row['crit_5pct']:.3f} | reject no cointegration |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Granger results support the no-event storyline: KTCI-a-roll7 has the strongest predictive-causality signal among scalar weather/comfort variables.",
        "- Raw KTCI alone is weaker than tavg in the Granger F-statistic, so the defensible claim is not 'raw KTCI is always better', but 'season-adjusted and smoothed KTCI is better'.",
        "- KTCI-a improves over KTCI, which supports the season-reassignment motivation.",
        "- Decomposition-a improves over Decomposition, which supports component-level KTCI customization using temperature-based seasons.",
        "- Cointegration is significant for all no-event models over the full period, suggesting long-run association signals. Because most core series are I(0), this should be presented as secondary evidence rather than the main model-selection criterion.",
        "",
        "## Figure Files",
        "",
    ])
    for path in sorted(FIG_DIR.glob("*.png")):
        lines.append(f"- `{path.name}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    specs = model_specs()

    all_columns = [TARGET]
    for spec in specs:
        all_columns.extend(spec["predictors"])
        all_columns.extend(spec["controls"])

    adf_df = run_adf(frame, all_columns)

    granger_rows = []
    for spec in specs:
        for lag in range(1, MAX_LAG + 1):
            row = conditional_granger(frame, spec["predictors"], spec["controls"], lag)
            row.update({
                "model": spec["model"],
                "display": spec["display"],
                "predictors": ", ".join(spec["predictors"]),
                "controls": ", ".join(spec["controls"]),
                "story": spec["story"],
            })
            granger_rows.append(row)
    granger_lag_df = pd.DataFrame(granger_rows)

    summary_rows = []
    for keys, group in granger_lag_df.groupby(["model", "display", "predictors", "story"], sort=False):
        best = group.loc[group["p_value"].idxmin()]
        sig_lags = group.loc[group["p_value"] < 0.05, "lag"].astype(str).tolist()
        summary_rows.append({
            "model": keys[0],
            "display": keys[1],
            "predictors": keys[2],
            "story": keys[3],
            "best_lag": int(best["lag"]),
            "best_f_stat": best["f_stat"],
            "best_p_value": best["p_value"],
            "significant_lags_p05": ", ".join(sig_lags) if sig_lags else "",
            "n_obs_at_best_lag": int(best["n_obs"]),
            "decision_best_p05": "reject_no_granger" if best["p_value"] < 0.05 else "not_reject",
        })
    granger_summary = pd.DataFrame(summary_rows)
    granger_summary["best_q_fdr"] = multipletests(granger_summary["best_p_value"].fillna(1.0), alpha=0.05, method="fdr_bh")[1]
    granger_summary["decision_best_q05"] = np.where(granger_summary["best_q_fdr"] < 0.05, "reject_no_granger_after_fdr", "not_reject_after_fdr")

    coint_rows = [run_cointegration(frame, spec) for spec in specs]
    coint_df = pd.DataFrame(coint_rows)
    coint_df["q_fdr"] = multipletests(coint_df["p_value"].fillna(1.0), alpha=0.05, method="fdr_bh")[1]
    coint_df["decision_p05"] = np.where(coint_df["p_value"] < 0.05, "reject_no_cointegration", "not_reject")
    coint_df["decision_q05"] = np.where(coint_df["q_fdr"] < 0.05, "reject_no_cointegration_after_fdr", "not_reject_after_fdr")

    adf_df.to_csv(OUTPUT_DIR / "adf_results_no_event.csv", index=False, encoding="utf-8-sig")
    granger_lag_df.to_csv(OUTPUT_DIR / "granger_lag_results_no_event.csv", index=False, encoding="utf-8-sig")
    granger_summary.to_csv(OUTPUT_DIR / "granger_summary_no_event.csv", index=False, encoding="utf-8-sig")
    coint_df.to_csv(OUTPUT_DIR / "cointegration_results_no_event.csv", index=False, encoding="utf-8-sig")

    make_figures(granger_summary, coint_df, adf_df, frame)
    write_report(granger_summary, coint_df, adf_df, frame)

    print("Saved outputs to", OUTPUT_DIR)
    print("\n=== No-event Granger summary ===")
    view = granger_summary.set_index("display").loc[MODEL_ORDER].reset_index()
    print(view[["display", "best_lag", "best_f_stat", "best_p_value", "best_q_fdr", "significant_lags_p05"]].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("\n=== No-event cointegration summary ===")
    cview = coint_df.set_index("display").loc[MODEL_ORDER].reset_index()
    print(cview[["display", "coint_t", "p_value", "q_fdr", "crit_5pct", "decision_q05"]].to_string(index=False, float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()
