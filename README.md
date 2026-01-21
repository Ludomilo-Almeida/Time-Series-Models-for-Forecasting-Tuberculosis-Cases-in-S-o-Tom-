# Time-Series-Models-for-Forecasting-Tuberculosis-Cases-in-S-o-Tom-

# Tuberculosis Incidence Forecasting in São Tomé: A Time Series Approach

![Evolução dos Casos de Tuberculose em STP](grafico_tuberculose_final.png)
*(O gráfico vai aparecer aqui em grande)*

## 1. Project Overview & Impact
Tuberculosis (TB) remains a critical public health challenge in São Tomé and Príncipe. Effective resource allocation relies heavily on knowing how many cases to expect in the future. 

The goal of this project was to **model and forecast the incidence of Tuberculosis cases**, providing statistical evidence to support public health officials in decision-making regarding medication stock and hospital resources.

## 2. The Data
The dataset consists of historical records of reported TB cases in São Tomé.
* **Period:** [2011] to [2023] .
* **Frequency:** [MENSAL].
* **Source:** Ministry of Health of São Tomé and Príncipe / National TB Program.

## 3. Methodology (Time Series Analysis)
As a strict mathematical approach, the Box-Jenkins methodology was applied to identify the best fit for the data structure.

1.  **Exploratory Analysis:** Decomposition of the series to identify Trend, Seasonality, and Noise.
2.  **Stationarity Tests:** Applied Augmented Dickey-Fuller (ADF) test to check for unit roots.
    * *Result:* The series required 1 order differencing ($d=1$) to become stationary.
3.  **Model Selection:** Tested multiple ARIMA/SARIMA specifications based on ACF and PACF plots.
    * Model selection criteria: Lowest AIC (Akaike Information Criterion) and BIC.
4.  **Residual Analysis:** Verified "White Noise" assumption (Ljung-Box test) and normality of residuals.

## 4. Models & Results
The following models were evaluated for forecasting accuracy:

* **Model A:** [ARIMA(2,1,1)]


**Best Model Performance:**
The **[ARIMA(2,1,1)]** proved to be the most robust predictor.
* **MAPE (Mean Absolute Percentage Error):** [XX]%
* **RMSE (Root Mean Square Error):** [XX]

> **Key Finding:** The model predicted a [TENDÊNCIA: Aumento/Diminuição/Estabilidade] in cases for the subsequent period, suggesting the need for [AÇÃO: ex: increased screening campaigns].

## 5. Mathematical Framework
To model the stochastic component of the series, the general form of the ARIMA model used can be described as:

$$\phi(B) (1-B)^d X_t = \theta(B) Z_t$$

Where $B$ is the backshift operator, illustrating the dependency on past values and errors.

## 6. Tools & Technologies
* **R Language** (Packages: `forecast`, `tseries`, `ggplot2`)
* **Statistical Analysis:** Hypothesis Testing, Time Series Decomposition.

## 7. Author
**[O Seu Nome]**
* *B.Sc. in Mathematics | M.Sc. in Statistical Modeling*
* [Link para o seu LinkedIn]

