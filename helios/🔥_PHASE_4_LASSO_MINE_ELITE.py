#!/usr/bin/env python3
"""
PHASE 4: LASSO MINE ELITE 30-50 FEATURES
Use LASSO to identify most predictive features
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler
from datetime import datetime

print("=" * 100)
print("🔥 PHASE 4: LASSO FEATURE MINING")
print("=" * 100)
print()

# Load extracted features
print("[1/4] Loading extracted features...")
with open("helios/data/HELIOS_6291_GAMES_720_FEATURES.pkl", 'rb') as f:
    df = pickle.load(f)

print(f"✅ Loaded {len(df)} games with {df.shape[1]-2} features")
print()

# Prepare data
print("[2/4] Preparing data for LASSO...")
feature_cols = [col for col in df.columns if col not in ['game_id', 'target']]
X = df[feature_cols].values
y = df['target'].values

print(f"   Features: {len(feature_cols)}")
print(f"   Samples: {len(X)}")
print()

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# LASSO feature selection
print("[3/4] Running LassoCV (this may take a few minutes)...")
print(f"Time: {datetime.now().strftime('%I:%M %p')}")
print()

alphas = np.logspace(-4, 1, 100)
lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000, n_jobs=-1, random_state=42)
lasso.fit(X_scaled, y)

print(f"✅ Best alpha: {lasso.alpha_:.6f}")
print()

# Get feature importances
feature_importance = np.abs(lasso.coef_)
feature_ranking = sorted(zip(feature_cols, feature_importance), key=lambda x: x[1], reverse=True)

# Select elite features (non-zero coefficients)
elite_features = [name for name, coef in feature_ranking if coef > 0]

print("[4/4] Elite features selected:")
print(f"   Total features: {len(feature_cols)}")
print(f"   Elite features: {len(elite_features)}")
print(f"   Reduction: {(1 - len(elite_features)/len(feature_cols))*100:.1f}%")
print()

# Show top 30
print("Top 30 features by importance:")
for i, (name, coef) in enumerate(feature_ranking[:30], 1):
    print(f"   {i:2d}. {name:30s} {coef:10.6f}")
print()

# Save results
print("Saving results...")
results = {
    'elite_features': elite_features,
    'all_features': feature_cols,
    'feature_importance': dict(feature_ranking),
    'lasso_model': lasso,
    'scaler': scaler
}

with open("helios/data/LASSO_ELITE_FEATURES.pkl", 'wb') as f:
    pickle.dump(results, f)

print("✅ Saved to: helios/data/LASSO_ELITE_FEATURES.pkl")
print()

print("=" * 100)
print("✅ PHASE 4 COMPLETE!")
print("=" * 100)
print()
print(f"Selected {len(elite_features)} elite features from {len(feature_cols)} total")
print(f"Ready for Phase 5: Model training")
print()

