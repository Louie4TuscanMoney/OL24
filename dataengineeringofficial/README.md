# 📚 DATA ENGINEERING OFFICIAL - RESEARCH FOUNDATIONS

**The Three Paradigms: Dejavu, Conformal, and Informer**

---

## 🎯 **WHAT IS DATA ENGINEERING OFFICIAL?**

**Data Engineering Official** contains the **complete research foundations** for three cutting-edge time-series forecasting paradigms:

1. **Dejavu** - Data-centric pattern matching (model-free)
2. **Conformal** - Uncertainty quantification wrapper
3. **Informer** - Efficient Transformer for long sequences

These three papers form the **theoretical foundation** for NBA score prediction, each contributing unique strengths to the autonomous trading system.

---

## 🔥 **THE THREE PARADIGMS**

### **1. Dejavu: Data-Centric Forecasting**

**Paper:** Kang et al., "Déjà vu: A data-centric forecasting approach through time series cross-similarity", arXiv:1909.00221v3 [stat.ME], September 2020

**Authors:**
- Yanfei Kang (Beihang University)
- Evangelos Spiliotis (NTUA)
- Fotios Petropoulos (University of Bath)
- Nikolaos Athiniotis (NTUA)
- Feng Li (CUFE)
- Vassilios Assimakopoulos (NTUA)

**Key Innovation:**
Instead of training a model, Dejavu **finds similar historical patterns** and uses their outcomes to predict the future.

**Why It's Powerful for NBA:**
- ✅ **No training required** - instant deployment
- ✅ **Interpretable** - shows which historical games are similar
- ✅ **Best for limited data** - paper proves it excels with ≤6 years of data
- ✅ **Perfect scale** - 18-minute patterns ideal for pattern matching

**Performance (M1/M3 Competitions):**
- Yearly: MASE 2.783 (BEST vs. ETS/ARIMA/Theta/SHD)
- Monthly: MASE 0.932 (tied with ETS)
- ETS-Similarity combo: BEST across all frequencies

---

### **2. Conformal: Uncertainty Quantification**

**Paper:** Schlembach et al., "Conformal Multistep-Ahead Multivariate Time-Series Forecasting", Proceedings of Machine Learning Research 179:1–3, 2022

**Authors:**
- Filip Schlembach (Maastricht University)
- Evgueni Smirnov (Maastricht University)
- Irena Koprinska (University of Sydney)

**Key Innovation:**
**Wraps any forecasting model** to provide calibrated prediction intervals with guaranteed coverage, even under distribution shifts.

**Why It's Powerful for NBA:**
- ✅ **Guaranteed coverage** - 90% interval contains true value 90% of time
- ✅ **Handles momentum swings** - robust to distribution shifts
- ✅ **Model-agnostic** - works with Dejavu, LSTM, Mamba
- ✅ **Exponential weighting** - recent games weighted more heavily

**Performance (ELEC2 Dataset):**
- Dataset: t=192, h=12, 1650 examples
- Features: 3 multivariate (Nswdemand, Vicdemand, Transfer)
- Base model: RNN (5 random initializations averaged)
- Key finding: Exponential weighting best for 1-α > 0.5

---

### **3. Informer: Efficient Transformer**

**Paper:** Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021, pages 11106-11115

**Authors:**
- Haoyi Zhou (Beihang University)
- Shanghang Zhang (UC Berkeley)
- Jieqi Peng (Beihang University)
- Shuai Zhang (Amazon Web Services)
- Jianxin Li (Beihang University)
- Hui Xiong (Rutgers University)
- Wancai Zhang (JD Finance)

**Key Innovation:**
**ProbSparse Self-Attention** reduces Transformer complexity from O(L²) to O(L log L), enabling long-sequence forecasting (300+ timesteps).

**Why It's Powerful for Long Sequences:**
- ✅ **Handles 1440 timesteps** - 10× longer than vanilla Transformer
- ✅ **64.3% MSE reduction** vs. LSTM on long horizons
- ✅ **Single GPU training** - Nvidia V100 32GB
- ✅ **Generative decoder** - one-shot prediction (no autoregressive)

**Performance (ETTh1, Horizon 720):**
- Informer MSE: 0.540
- LSTM MSE: 1.511
- Improvement: 64.3% MSE reduction

**⚠️ NBA Limitation:**
- Designed for 300+ inputs, but NBA games only have ~18 minutes
- Better suited for **full season analysis** (82 games × 48 min = 3,936 datapoints)

---

## 📂 **PACKAGE STRUCTURE**

```
dataengineeringofficial/
├── dejavu/                        # Dejavu research & specs
│   ├── DEFINITION.md                    # What is Dejavu?
│   ├── DEJAVU_MODEL.md                  # Model architecture
│   ├── DEJAVU_IMPLEMENTATION_SPEC.md    # Implementation guide
│   ├── DATA_ENGINEERING_DEJAVU.md       # Data pipeline specs
│   ├── MATH_BREAKDOWN.txt               # Mathematical foundations
│   └── RESEARCH_BREAKDOWN.txt           # Research insights
│
├── conformal/                     # Conformal research & specs
│   ├── DEFINITION.md                    # What is Conformal?
│   ├── CONFORMAL_MODEL.markdown         # Model architecture
│   ├── CONFORMAL_IMPLEMENTATION_SPEC.md # Implementation guide
│   ├── DATA_ENGINEERING_CONFORMAL.md    # Data pipeline specs
│   ├── MATH_BREAKDOWN.txt               # Mathematical foundations
│   └── RESEARCH_BREAKDOWN.txt           # Research insights
│
├── informer/                      # Informer research & specs
│   ├── DEFINITION.md                    # What is Informer?
│   ├── INFORMER_MODEL.markdown          # Model architecture
│   ├── INFORMER_IMPLEMENTATION_SPEC.md  # Implementation guide
│   ├── DATA_ENGINEERING_INFORMER.md     # Data pipeline specs
│   ├── MATH_BREAKDOWN.txt               # Mathematical foundations
│   └── RESEARCH_BREAKDOWN.txt           # Research insights
│
├── papers/                        # Original research papers
│   ├── Dejavu_Paper.pdf                 # Original Dejavu paper
│   ├── Conformal_Paper.pdf              # Original Conformal paper
│   └── Informer_Paper.pdf               # Original Informer paper
│
├── applied/                       # Applied implementations
│   └── informer/
│       ├── config_matrix.csv            # Hyperparameter configs
│       ├── deployment_template.py       # Production deployment
│       ├── model_specs.py               # Model specifications
│       ├── QUICKSTART.py                # Quick start guide
│       ├── README.md                    # Applied model docs
│       ├── requirements.txt             # Dependencies
│       └── sql_data_pipeline.py         # SQL data integration
│
├── documentation/                 # Research analysis
│   ├── PAPER_METADATA_SUMMARY.md        # Complete paper citations
│   ├── PAPER_VERIFICATION_COMPLETE.md   # Verification report
│   ├── PAPER_VERIFIED_SUMMARY.md        # Executive summary
│   ├── FINAL_VERIFICATION_REPORT.md     # Final report
│   ├── SYNTHESIS_AND_STRATEGIC_ANALYSIS.md # Strategic analysis
│   └── REFLECTIONS_ON_THREE_PARADIGMS.md # Philosophical reflections
│
└── README.md                      # This file
```

---

## 🧠 **MODEL COMPARISON FOR NBA**

### **Dejavu vs. Conformal vs. Informer**

| **Aspect** | **Dejavu** | **Conformal** | **Informer** |
|------------|-----------|--------------|-------------|
| **Training** | None (model-free) | Wraps any model | Deep learning |
| **Data Required** | ≤6 years (optimal) | Any (wraps base model) | 300+ timesteps |
| **NBA Fit** | ✅ Perfect (18 inputs) | ✅ Perfect (any scale) | ⚠️ Overkill (needs 300+) |
| **Interpretability** | ✅ Shows similar games | ⚠️ Interval only | ❌ Black box |
| **Deployment** | ✅ Instant | ✅ After base model | ⚠️ Complex |
| **Uncertainty** | ❌ Point prediction | ✅ Calibrated intervals | ❌ Point prediction |
| **Best Use Case** | Pattern matching | Wrap Mamba/LSTM | Full season analysis |

---

## 🎯 **RECOMMENDED PRODUCTION STACK**

### **For NBA (Current System):**

```
PRIMARY: Mamba (LSTM-like) + OntoRisk
├─ Mamba: Trained on 5,529 games (18-minute patterns)
├─ OntoRisk: Probability calibration (Conformal-inspired)
└─ Kelly Criterion: Optimal bet sizing

ALTERNATIVE: Dejavu + Conformal
├─ Dejavu: Find similar historical games (instant)
├─ Conformal: Wrap for calibrated intervals
└─ Interpretability: Show which past games are similar

RESERVE INFORMER FOR:
├─ Full season predictions (82 games)
├─ Player career trajectories (hundreds of games)
└─ League-wide trend analysis
```

---

## 🚀 **QUICK START**

### **1. Dejavu Pattern Matching**

```python
from dataengineeringofficial.dejavu import DejavuForecaster

# Find similar historical games
dejavu = DejavuForecaster(reference_set='nba_games_2020_2025.parquet')

# Current game state
current_pattern = [2, -3, 5, 1, -2, 4, 3, -1, 2, 0, 1, -2, 3, 2, -1, 1, 0, 2]

# Find K=5 most similar games
similar_games = dejavu.find_similar(current_pattern, k=5)

# Predict final differential
prediction = dejavu.predict(current_pattern, k=5, method='weighted_mean')
print(f"Predicted final differential: {prediction:.1f}")
```

### **2. Conformal Uncertainty Quantification**

```python
from dataengineeringofficial.conformal import ConformalWrapper

# Wrap any base model (Mamba, LSTM, etc.)
conformal = ConformalWrapper(
    base_model=mamba_model,
    calibration_data=calibration_games,
    alpha=0.1  # 90% coverage
)

# Get prediction with calibrated interval
prediction, lower, upper = conformal.predict(current_features)
print(f"Prediction: {prediction:.1f}, Interval: [{lower:.1f}, {upper:.1f}]")
```

### **3. Informer Long-Sequence Forecasting**

```python
from dataengineeringofficial.applied.informer import InformerModel

# Full season prediction (82 games)
informer = InformerModel(
    seq_len=336,  # 336 timesteps input
    label_len=168,
    pred_len=96,  # Predict 96 timesteps ahead
    d_model=512,
    n_heads=8,
    e_layers=3,
    d_layers=1
)

# Train on full season data
informer.fit(season_data)

# Predict next 96 timesteps
future = informer.predict(last_336_timesteps)
```

---

## 📊 **PAPER PERFORMANCE COMPARISON**

### **Dejavu (M1/M3 Competitions)**
- Dataset: 3,830 series (M1/M3), 95,000 reference (M4)
- Performance:
  - Yearly: MASE 2.783 (BEST)
  - Quarterly: MASE 1.154 (competitive)
  - Monthly: MASE 0.932 (tied with ETS)
- Key insight: Best for ≤6 years of data

### **Conformal (ELEC2 Dataset)**
- Dataset: 1,650 examples (660/660/330 split)
- Input: t=192, Output: h=12
- Features: 3 multivariate (Nswdemand, Vicdemand, Transfer)
- Coverage: 90% intervals contain true value 90% of time
- Key insight: Exponential weighting best for recent data

### **Informer (ETT, ECL, Weather)**
- Dataset: ETTh1 (12/4/4 months), ECL (15/3/4 months), Weather (28/10/10 months)
- Input lengths: 336, 480, 720, 960, 1440 timesteps
- Horizons: 48, 96, 168, 336, 720, 960 steps
- Best performance: ETTh1-720 (Informer 0.540 vs LSTM 1.511)
- Platform: Single Nvidia V100 32GB GPU

---

## 🧠 **MATHEMATICAL FOUNDATIONS**

### **Dejavu: K-NN Pattern Matching**

**Similarity Measure:**
```
d(x, y) = sqrt(sum((x_i - y_i)^2))  # Euclidean distance
d(x, y) = DTW(x, y)                 # Dynamic Time Warping
d(x, y) = 1 - corr(x, y)            # Correlation distance
```

**Weighted Forecast:**
```
y_hat = sum(w_i * y_i) / sum(w_i)
where w_i = 1 / (d_i + epsilon)
```

### **Conformal: Weighted Quantiles**

**Conformity Score:**
```
r_i = |y_i - f(x_i)|  # Absolute error
```

**Prediction Interval:**
```
y_hat ± Q_{1-α}({(r_i, w_i)})
where w_i = exp(-λ * (n - i))  # Exponential weighting
```

**Bonferroni Correction (Multistep):**
```
α_corrected = α / h
where h = forecast horizon
```

### **Informer: ProbSparse Self-Attention**

**Query Sparsity Measure:**
```
M(q_i) = ln(sum(exp(q_i * k_j / sqrt(d)))) - 1/L * sum(q_i * k_j / sqrt(d))
```

**Top-u Queries:**
```
Q_sparse = top_u(M(q_i), u = c * ln L)
where c = sampling factor (5 optimal)
```

**Complexity:**
```
Vanilla Transformer: O(L^2)
Informer: O(L ln L)
```

---

## 🔧 **DATA ENGINEERING SPECIFICATIONS**

### **Dejavu Data Requirements**
- **Reference set:** 1,000+ historical games
- **Pattern length:** 18 minutes (Q1 + Q2 6:00)
- **Features:** Score differentials (minute-by-minute)
- **Storage:** Parquet format (efficient)
- **Similarity index:** K-D tree or Annoy for fast search

### **Conformal Data Requirements**
- **Calibration set:** 200+ games
- **Base model:** Any (Mamba, LSTM, Dejavu)
- **Weighting:** Exponential (λ=0.01 to 0.1)
- **Coverage target:** 90% (α=0.1)
- **Validation:** Split-conformal (separate calibration/test)

### **Informer Data Requirements**
- **Training set:** 10,000+ timesteps (300+ games minimum)
- **Input length:** 336+ timesteps
- **Prediction horizon:** 96+ timesteps
- **Features:** Multivariate (score, pace, efficiency)
- **Normalization:** StandardScaler or MinMaxScaler
- **GPU:** Nvidia V100 or equivalent (32GB+ VRAM)

---

## 📈 **WHEN TO USE EACH MODEL**

### **Use Dejavu When:**
- ✅ You have limited data (≤6 years)
- ✅ You need instant deployment (no training)
- ✅ You want interpretability (show similar games)
- ✅ Input length is short (18 timesteps)
- ✅ You're predicting familiar patterns

### **Use Conformal When:**
- ✅ You need calibrated uncertainty (betting odds)
- ✅ You have distribution shifts (momentum swings)
- ✅ You want guaranteed coverage (90% intervals)
- ✅ You're wrapping any base model (Mamba, LSTM)
- ✅ You're under regulatory scrutiny (explainable AI)

### **Use Informer When:**
- ✅ You have LONG sequences (300+ timesteps)
- ✅ You're predicting FAR ahead (96+ timesteps)
- ✅ You have abundant data (10,000+ timesteps)
- ✅ You have GPU resources (V100 or better)
- ✅ You're doing full season analysis (82 games)

---

## 🛠️ **DEPENDENCIES**

### **Dejavu**
```
numpy >= 1.21.0
pandas >= 1.3.0
scipy >= 1.7.0
scikit-learn >= 0.24.0  # For distance metrics
fastdtw >= 0.3.4        # For DTW
```

### **Conformal**
```
numpy >= 1.21.0
pandas >= 1.3.0
scipy >= 1.7.0
scikit-learn >= 0.24.0  # For base models
```

### **Informer**
```
torch >= 1.9.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 0.24.0
matplotlib >= 3.4.0     # For visualization
```

---

## 📚 **DOCUMENTATION FILES**

### **Dejavu Documentation**
- **DEFINITION.md** - What is Dejavu? High-level overview
- **DEJAVU_MODEL.md** - Model architecture and algorithms
- **DEJAVU_IMPLEMENTATION_SPEC.md** - Production implementation guide
- **DATA_ENGINEERING_DEJAVU.md** - Data pipeline specifications
- **MATH_BREAKDOWN.txt** - Mathematical foundations (similarity, K-NN, DTW)
- **RESEARCH_BREAKDOWN.txt** - Research insights and use cases

### **Conformal Documentation**
- **DEFINITION.md** - What is Conformal? High-level overview
- **CONFORMAL_MODEL.markdown** - Model architecture and algorithms
- **CONFORMAL_IMPLEMENTATION_SPEC.md** - Production implementation guide
- **DATA_ENGINEERING_CONFORMAL.md** - Data pipeline specifications
- **MATH_BREAKDOWN.txt** - Mathematical foundations (weighted quantiles, Bonferroni)
- **RESEARCH_BREAKDOWN.txt** - Research insights and ELEC2 experiments

### **Informer Documentation**
- **DEFINITION.md** - What is Informer? High-level overview
- **INFORMER_MODEL.markdown** - Model architecture and algorithms
- **INFORMER_IMPLEMENTATION_SPEC.md** - Production implementation guide
- **DATA_ENGINEERING_INFORMER.md** - Data pipeline specifications
- **MATH_BREAKDOWN.txt** - Mathematical foundations (ProbSparse, Distilling, Decoder)
- **RESEARCH_BREAKDOWN.txt** - Research insights and ETT/ECL/Weather experiments

### **Research Analysis Documentation**
- **PAPER_METADATA_SUMMARY.md** - Complete paper citations and metadata
- **PAPER_VERIFICATION_COMPLETE.md** - Detailed verification report
- **PAPER_VERIFIED_SUMMARY.md** - Executive summary of verification
- **FINAL_VERIFICATION_REPORT.md** - Final verification report
- **SYNTHESIS_AND_STRATEGIC_ANALYSIS.md** - Strategic analysis across papers
- **REFLECTIONS_ON_THREE_PARADIGMS.md** - Philosophical reflections

---

## 🎯 **QUICK REFERENCE**

```python
# Dejavu: Find similar games
from dejavu import DejavuForecaster
dejavu = DejavuForecaster(reference_set)
prediction = dejavu.predict(current_pattern, k=5)

# Conformal: Calibrated intervals
from conformal import ConformalWrapper
conformal = ConformalWrapper(base_model, calibration_data, alpha=0.1)
pred, lower, upper = conformal.predict(features)

# Informer: Long-sequence forecasting
from applied.informer import InformerModel
informer = InformerModel(seq_len=336, pred_len=96)
future = informer.predict(last_336_timesteps)
```

---

## 📄 **CITATIONS**

### **Dejavu**
```bibtex
@article{kang2020dejavu,
  title={D{\'e}j{\`a} vu: A data-centric forecasting approach through time series cross-similarity},
  author={Kang, Yanfei and Spiliotis, Evangelos and Petropoulos, Fotios and Athiniotis, Nikolaos and Li, Feng and Assimakopoulos, Vassilios},
  journal={arXiv preprint arXiv:1909.00221},
  year={2020}
}
```

### **Conformal**
```bibtex
@inproceedings{schlembach2022conformal,
  title={Conformal Multistep-Ahead Multivariate Time-Series Forecasting},
  author={Schlembach, Filip and Smirnov, Evgueni and Koprinska, Irena},
  booktitle={Proceedings of Machine Learning Research},
  volume={179},
  pages={1--3},
  year={2022}
}
```

### **Informer**
```bibtex
@inproceedings{zhou2021informer,
  title={Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={Proceedings of the 35th AAAI Conference on Artificial Intelligence},
  pages={11106--11115},
  year={2021}
}
```

---

**📚 Data Engineering Official: The research foundations that power intelligent NBA forecasting! 📚**

