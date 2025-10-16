# What is Conformal Prediction?

**Type:** Model-Agnostic Uncertainty Wrapper

**Authors:** Schlembach et al. (2022), building on Barber et al. (2022) and Stankevičiūtė et al. (2021)

---

## Definition

**Conformal Prediction** is a technique that wraps any machine learning model to provide **statistically valid prediction intervals** with theoretical coverage guarantees.

It is **model-agnostic**, meaning it works with any point forecasting model (LSTM, Transformer, Random Forest, etc.) without modifying the underlying model architecture.

---

## Core Concept

Instead of just predicting a single value, Conformal Prediction provides:

```
Point Forecast ± Uncertainty Interval
```

With a **mathematical guarantee** that the true value falls within the interval a specified percentage of the time (e.g., 95% coverage).

---

## How It Works

1. **Train** any base forecasting model on training data
2. **Calibrate** using a held-out calibration set to measure prediction errors
3. **Compute quantiles** of these errors to determine interval width
4. **Wrap predictions** with intervals that provide guaranteed coverage

---

## Key Features

- ✅ **Model-agnostic**: Works with any forecasting model
- ✅ **Theoretical guarantees**: Mathematically proven coverage rates
- ✅ **Distribution-shift robust**: Maintains coverage even when data patterns change
- ✅ **Multistep-ahead**: Extends to forecasting multiple future time steps
- ✅ **Multivariate**: Handles multiple features/dimensions simultaneously
- ✅ **Efficient**: Minimal computational overhead (<1ms per prediction)

---

## Why It Matters

Most ML models only provide **point predictions** without uncertainty quantification. Conformal Prediction answers:

> "How confident should I be in this prediction?"

This is critical for:
- **Risk management** (knowing when predictions are uncertain)
- **Decision-making** (acting based on confidence levels)
- **Production systems** (providing reliable bounds on forecasts)

---

## Technical Innovation (This Paper)

The Schlembach et al. (2022) method specifically addresses:

1. **Non-exchangeable time series**: Uses weighted quantiles to handle temporal dependence
2. **Distribution shifts**: Adapts to changing data patterns over time
3. **Multistep multivariate**: Extends to complex forecasting scenarios with Bonferroni correction

---

## In Our System

Conformal Prediction wraps the **Dejavu + LSTM ensemble** to provide:
- Point forecast from ensemble (e.g., +12.5 points)
- 95% confidence interval (e.g., [+8.2, +16.8])
- Guaranteed coverage that holds even during unusual game situations

---

**Bottom Line:** Conformal Prediction transforms deterministic forecasts into probabilistic predictions with statistical guarantees.

