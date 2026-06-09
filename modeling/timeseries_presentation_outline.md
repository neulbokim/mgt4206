# Final Team Project Presentation Draft

## 1. Title
**Forecasting Daily Visitor Demand at Gyeongbokgung Palace with SARIMAX and Climate-Comfort Exogenous Variables**

## 2. Problem Motivation
- Daily visitor demand at Gyeongbokgung is strongly affected by weather, holidays, and seasonal transitions.
- A simple temperature-based baseline does not fully capture thermal comfort or the way Korean seasonal changes affect visiting behavior.
- Our main question is whether a tourism climate index and a season redefinition can improve daily forecasting performance.
- The core story is: `Baseline -> KTCI -> KTCI-a -> KTCI-a-roll7`.

## 3. Literature Review
| Paper | Why it matters for this project |
|---|---|
| Witt & Witt (1995), *Forecasting tourism demand: A review of empirical research* | Establishes tourism demand forecasting as a forecasting problem where no single method is always best. Supports the use of benchmark comparisons. |
| Song & Li (2008), *Tourism demand modelling and forecasting - A review of recent research* | Shows that seasonality and forecast combination matter, and that econometric/time-series models remain central. |
| Kim & Kim (2014), *Development of Korea Tourism Climate Index (KTCI)* | Provides the conceptual basis for using a Korea-specific climate comfort index instead of raw temperature alone. |
| Hwang, Kim, & Yu (2018), *The empirical test on the impact of climate volatility on tourism demand* | Shows that climate-related variability matters for tourism demand in Korea, which supports climate-based predictors. |
| Biagi & Pulina (2009), *Bivariate VAR models to test Granger causality between tourist demand and supply* | Supports using Granger causality as a supplementary lag-screening tool, separate from forecast accuracy evaluation. |
| Xu, Liu, & Jin (2023), *Forecasting daily tourism demand with multiple factors* | Confirms that daily tourism forecasting improves when multiple heterogeneous factors are used. |

## 4. Statement of Research Objectives
- Test whether `KTCI` improves on the temperature baseline.
- Test whether `KTCI-a` improves on `KTCI` by redefining the season structure for Korean climate transitions.
- Test whether `KTCI-a-roll7` further improves stability by smoothing short-term noise.
- Use Granger causality as a predictive screening tool, not as the final model-selection criterion.

## 5. Data & Methodology
### Data overview
- Unit of analysis: daily visitor count at Gyeongbokgung Palace.
- Sample period: 2009-01-01 to 2025-12-31.
- Observations: 6,209 daily rows.
- PM10 coverage: 5,926 rows.

### Summary statistics

| Variable | Mean | Median | Std. Dev. | Min | Max |
|---|---:|---:|---:|---:|---:|
| Visitors | 12079.03 | 10729.00 | 11504.60 | 0.00 | 154000.00 |
| Average temperature (`tavg`) | 13.28 | 14.60 | 10.75 | -14.90 | 33.70 |
| `KTCI` | 76.54 | 77.79 | 11.00 | 32.99 | 95.74 |
| `KTCI-a` | 76.50 | 77.70 | 10.98 | 32.99 | 96.30 |
| `KTCI-a-roll7` | 76.50 | 76.84 | 8.34 | 56.08 | 93.33 |
| Rainfall (`rain`) | 3.83 | 0.00 | 14.45 | 0.00 | 301.50 |
| PM10 | 44.39 | 38.73 | 30.37 | 4.00 | 613.00 |

### Methodology
- Model family: SARIMAX.
- Main exogenous variables:
  - `tavg` for baseline
  - `ktci`
  - `ktci_a`
  - `ktci_a_roll7`
- Common controls:
  - rainfall, PM10, holiday indicators, weekend indicators, closure indicators, and derived holiday variables.
- Diagnostics:
  - ADF for stationarity
  - VIF for collinearity
  - Granger causality for predictive precedence
  - Ljung-Box and residual ACF/PACF for model adequacy

### Stationarity and cointegration note
- The main presentation should not use cointegration as the central argument.
- Our core research question is short-term daily forecasting and predictive explanatory power, not long-run equilibrium.
- ADF results show that the core variables are mostly I(0), so a cointegration test is not the most appropriate main evidence.
- If needed, cointegration can be mentioned briefly in an appendix as a completeness check, but it should not drive the main conclusion.
- If the professor asks about stationarity, explain ADF first and keep cointegration as backup material only.

## 6. Analysis Result
### Slide message
- `Baseline` tests whether raw temperature is enough.
- `KTCI` tests whether a climate-comfort index is better than temperature.
- `KTCI-a` tests whether Korean season recalibration improves the fit.
- `KTCI-a-roll7` tests whether smoothing the season-aware index stabilizes short-term prediction.

### How to interpret Granger vs SARIMAX
- Granger causality answers: "Does X help predict Y at some lag?"
- SARIMAX answers: "Which specification gives the best out-of-sample forecast?"
- Therefore, the best Granger variable and the best SARIMAX model do not need to be the same.
- This difference is expected and should be explained explicitly in the presentation.

### Recommended figures
- Visitor time-series plot with major seasonal and COVID shifts.
- Baseline vs `KTCI` vs `KTCI-a` vs `KTCI-a-roll7` forecast comparison.
- Transition-period zoom-in plot where `season_month != season_temp`.
- Residual ACF/PACF and Ljung-Box table.
- Coefficient comparison table for the four models.

### Suggested result narration
- If `KTCI` improves over the baseline, we can say that climate comfort is more informative than temperature alone.
- If `KTCI-a` improves further, we can say that Korean seasonal transitions are not fully captured by the original season definition.
- If `KTCI-a-roll7` is the most stable, we can say that smoothing helps reduce short-term noise and improves practical forecasting stability.

## 7. Expected Original Contribution
- A Korea-specific climate-comfort predictor can outperform raw temperature in daily visitor forecasting.
- Recalculating seasonal structure around Korean climate transitions adds explanatory power.
- The project connects predictive modeling with seasonality design, not just with variable selection.
- The work is directly usable for visitor management and staffing at a heritage site.

## 8. Limitations and Future Work
- The analysis is based on one site only.
- The main horizon is daily, short-term forecasting.
- Exogenous variables are still limited to weather, holidays, closures, and climate indices.
- Future work:
  - extend the framework to other heritage sites and attractions,
  - test probabilistic forecasts,
  - compare against more flexible nonlinear or deep learning models,
  - examine whether the season redefinition generalizes across regions.

## 9. Reference List
- Biagi, B., & Pulina, M. (2009). *Bivariate VAR models to test Granger causality between tourist demand and supply: Implications for regional sustainable growth*. Papers in Regional Science, 88(1), 231-245.
- Hwang, Y. S., Kim, H. S. H., & Yu, C. (2018). *The empirical test on the impact of climate volatility on tourism demand: A case of Japanese tourists visiting Korea*. Sustainability, 10(10), 3569.
- Kim, N. J., & Kim, S. T. (2014). *Development of Korea Tourism Climate Index (KTCI)*. Journal of Tourism Sciences, 38(6), 253-275.
- Song, H., & Li, G. (2008). *Tourism demand modelling and forecasting - A review of recent research*. Tourism Management, 29(2), 203-220.
- Witt, S. F., & Witt, C. A. (1995). *Forecasting tourism demand: A review of empirical research*. International Journal of Forecasting, 11(3), 447-475.
- Xu, S., Liu, Y., & Jin, C. (2023). *Forecasting daily tourism demand with multiple factors*. Annals of Tourism Research, 103, 103675.

## Appendix Note
- The cointegration slide can be kept in backup material only.
- If the professor asks about stationarity, say that the ADF result supports the use of forecast-oriented diagnostics rather than long-run equilibrium testing as the main evidence.
