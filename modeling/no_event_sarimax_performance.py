from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from no_event_granger_cointegration_tests import (
    BASE_CONTROLS,
    MODEL_LABELS,
    MODEL_ORDER,
    OUTPUT_DIR,
    TARGET,
    load_frame,
    model_specs,
)


TEST_SIZE = 365
SARIMAX_ORDER = (2, 0, 1)
SEASONAL_ORDER = (1, 0, 1, 7)
MAXITER = 200
FIG_DIR = OUTPUT_DIR / "figures"


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2))


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else float("nan")


def adjusted_r2_score(y_true: pd.Series, y_pred: pd.Series, n_predictors: int) -> float:
    n_obs = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n_obs <= n_predictors + 1:
        return float("nan")
    return float(1 - (1 - r2) * (n_obs - 1) / (n_obs - n_predictors - 1))


def prepare_sarimax_frame() -> pd.DataFrame:
    frame = load_frame().copy()

    continuous = [
        "rain",
        "tavg",
        "ktci",
        "ktci_a",
        "ktci_a_roll7",
        "Cd_month_w",
        "Ca_month_w",
        "P_month_w",
        "W_month_w",
        "S_month_w",
        "Cd_temp_w",
        "Ca_temp_w",
        "P_temp_w",
        "W_temp_w",
        "S_temp_w",
    ]
    for col in continuous:
        frame[f"{col}_c"] = frame[col] - frame[col].mean()

    frame["rain_c_x_weekend"] = frame["rain_c"] * frame["is_weekend"]
    frame["rain_c_x_offdays_left"] = frame["rain_c"] * frame["offdays_left"]

    for col in ["tavg", "ktci", "ktci_a", "ktci_a_roll7"]:
        frame[f"{col}_c_x_weekend"] = frame[f"{col}_c"] * frame["is_weekend"]
        frame[f"{col}_c_x_offdays_left"] = frame[f"{col}_c"] * frame["offdays_left"]

    return frame


def exog_columns(spec: dict) -> list[str]:
    predictors = list(spec["predictors"])
    common = [*BASE_CONTROLS, "rain_c_x_weekend", "rain_c_x_offdays_left"]
    if len(predictors) == 1:
        main = predictors[0]
        return [
            main,
            *common,
            f"{main}_c_x_weekend",
            f"{main}_c_x_offdays_left",
        ]
    return [*predictors, *common]


def fit_model(model_name: str, exog_cols: list[str], train: pd.DataFrame, test: pd.DataFrame) -> dict:
    y_train = train[TARGET]
    y_test = test[TARGET]
    x_train = train[exog_cols]
    x_test = test[exog_cols]

    model = SARIMAX(
        y_train,
        exog=x_train,
        order=SARIMAX_ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=MAXITER)
    pred = result.get_forecast(steps=len(y_test), exog=x_test).predicted_mean
    pred = pd.Series(pred, index=y_test.index)

    return {
        "display": model_name,
        "n_predictors": len(exog_cols),
        "AIC": result.aic,
        "BIC": result.bic,
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
        "Adj_R2": adjusted_r2_score(y_test, pred, len(exog_cols)),
    }


def make_figure(score_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ordered = score_df.set_index("display").loc[MODEL_ORDER].reset_index()
    x = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    bars = axes[0].bar(x, ordered["RMSE"], color="#4C78A8")
    axes[0].set_title("No-Event SARIMAX Test RMSE", fontsize=14, weight="bold")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODEL_LABELS[d] for d in ordered["display"]])
    axes[0].grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, ordered["RMSE"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=8)

    bars = axes[1].bar(x, ordered["Adj_R2"], color="#54A24B")
    axes[1].set_title("No-Event SARIMAX Adjusted R-squared", fontsize=14, weight="bold")
    axes[1].set_ylabel("Adjusted R-squared")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_LABELS[d] for d in ordered["display"]])
    axes[1].grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, ordered["Adj_R2"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_no_event_sarimax_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame = prepare_sarimax_frame()
    train = frame.iloc[:-TEST_SIZE].copy()
    test = frame.iloc[-TEST_SIZE:].copy()

    rows = []
    for spec in model_specs():
        cols = exog_columns(spec)
        print(f"Fitting {spec['display']} with {len(cols)} exog columns")
        rows.append(fit_model(spec["display"], cols, train, test))

    score_df = pd.DataFrame(rows).sort_values(["RMSE", "MAE", "Adj_R2"], ascending=[True, True, False])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(OUTPUT_DIR / "sarimax_scores_no_event.csv", index=False, encoding="utf-8-sig")
    make_figure(score_df)
    print(score_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()
