# What is Informer?

**Type:** Deep Learning Forecasting Model (Transformer Architecture)

**Authors:** Zhou et al. (2021) - Beihang University

---

## Definition

**Informer** is a Transformer-based deep learning model specifically designed for **long-sequence time series forecasting**.

It improves upon standard Transformers by addressing their computational inefficiency, making it practical to forecast far into the future with long historical context.

---

## Core Concept

```
Long Historical Sequence → Efficient Attention → Long Future Forecast
```

Informer can process hundreds of time steps efficiently and predict dozens of steps ahead in a single pass.

---

## The Problem It Solves

Standard Transformers have **quadratic complexity** O(L²) in attention computation:
- Long sequences (L=512+) become computationally prohibitive
- Memory requirements explode with sequence length
- Impractical for long-term forecasting

Informer reduces this to **O(L log L)** complexity.

---

## Key Innovations

### 1. **ProbSparse Self-Attention**
Instead of computing attention for all pairs of time steps, only focus on the most "informative" queries:
- Measures which queries contribute most to predictions
- Dramatically reduces computation while maintaining accuracy

### 2. **Self-Attention Distilling**
Progressively reduces sequence length through pooling layers:
- Input: 512 steps → 256 → 128 → 64
- Extracts hierarchical temporal features
- Reduces memory footprint

### 3. **Generative Decoder**
Predicts all future steps in one forward pass (not autoregressive):
- Faster inference
- Avoids error accumulation
- Better for long-term forecasts

---

## Architecture Overview

```
Input Sequence (e.g., 18 minutes)
    ↓
Embedding Layer (position + time encoding)
    ↓
Encoder (ProbSparse Attention + Distilling)
    ↓
Decoder (Generative attention)
    ↓
Output Sequence (e.g., next 6 minutes)
```

---

## Key Features

- ✅ **Long-sequence capable**: Handles 100+ time steps efficiently
- ✅ **Multistep-ahead**: Predicts entire future horizon at once
- ✅ **Multivariate**: Handles multiple features/dimensions
- ✅ **Efficient**: O(L log L) complexity vs O(L²) for standard Transformers
- ✅ **State-of-the-art**: Outperforms LSTM, GRU, and standard Transformers on long-term forecasting
- ✅ **Captures long-range dependencies**: Attention mechanism finds distant patterns

---

## Strengths

- **Powerful pattern recognition**: Learns complex temporal dependencies
- **Long-term forecasting**: Excels at predicting far into the future
- **Handles irregularities**: Robust to missing data and noise
- **Scalable**: Efficient enough for production deployment
- **SOTA performance**: Best accuracy on benchmark datasets

---

## Limitations

- **Black box**: Hard to interpret what the model learned
- **Requires training**: Needs large datasets and GPU resources
- **Hyperparameter tuning**: Sensitive to learning rate, layers, attention heads
- **Overfitting risk**: Can memorize training patterns
- **Inference cost**: More expensive than simple models (~10ms vs <1ms)

---

## Why "Informer"?

**Informed + Transformer** = Informer
- Uses information theory to select most informative attention queries
- "Informs" predictions with long historical context

---

## In Our System

In our NBA prediction system, we use **LSTM instead of Informer** because:
- Shorter sequences (18 minutes) don't require Informer's long-sequence optimizations
- LSTM is simpler and faster for our use case
- Informer excels at very long horizons (predicting hours/days ahead)

However, Informer principles inform our architecture:
- Attention to temporal patterns
- Efficient sequence processing
- Multi-step-ahead prediction capabilities

---

## Performance Benchmarks (from paper)

| Dataset | Horizon | Informer MAE | LSTM MAE | Speedup |
|---------|---------|--------------|----------|---------|
| ETTh1 | 168 steps | 0.413 | 0.522 | 3.2× |
| Weather | 720 steps | 0.266 | 0.319 | 5.1× |
| Electricity | 336 steps | 0.193 | 0.274 | 4.8× |

---

## Example Use Case

**Energy demand forecasting:**
- Input: 7 days of historical load (168 hours)
- Output: Next 3 days of predicted load (72 hours)
- Informer captures weekly patterns, daily cycles, and weather correlations efficiently

---

**Bottom Line:** Informer is a highly efficient Transformer for long-sequence time series forecasting that achieves state-of-the-art accuracy with practical computational costs.

