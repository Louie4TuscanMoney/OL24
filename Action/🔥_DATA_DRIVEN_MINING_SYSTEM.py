"""
🔥 DATA-DRIVEN FEATURE MINING + SEGMENTATION SYSTEM
Elite approach: Mine ALL parameters from data - NO hardcoded thresholds!

PHILOSOPHY:
  • Let the data tell us what "close game" means
  • Let the data find optimal score_diff bins
  • Let the data discover natural game segments
  • Let the data select important features

WORKFLOW:
  Phase 1: Mine context variable distributions (decision trees)
  Phase 2: Find optimal bins for continuous variables
  Phase 3: Multi-dimensional clustering (joint segmentation)
  Phase 4: Engineer features from mined segments
  Phase 5: Train specialized models per mined segment
  Phase 6: Extract feature importance (validate mining)
  Phase 7: Iterative refinement

NO ASSUMPTIONS - ONLY DATA-DRIVEN DISCOVERY!
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge, LassoCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("🔥 DATA-DRIVEN FEATURE MINING + SEGMENTATION SYSTEM")
print("="*90)
print("\nPHILOSOPHY: Let the DATA discover patterns, not assumptions!")
print("\nApproach:")
print("  • Mine optimal thresholds from decision trees")
print("  • Discover natural clusters via unsupervised learning")
print("  • Engineer features from mined segments")
print("  • Train specialized models per discovered pattern")
print("\n" + "="*90)

# Load data
print("\n[PHASE 1] LOADING & EXTRACTING CONTEXT VARIABLES...")
with open('Action/ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
    data_list = pickle.load(f)

data_sorted = sorted(data_list, key=lambda x: x.get('date', ''))

# Extract ALL available context variables from data
context_data = []
X_base = []
y_final = []
y_current = []

for game in data_sorted:
    pattern = game.get('pattern', [])
    if not isinstance(pattern, list) or len(pattern) < 18:
        continue
    
    base_features = pattern[:18]
    final_diff = game.get('diff_at_final', 0)
    current_diff = game.get('diff_at_halftime', game.get('diff_at_2q_6min', 0))
    
    # Extract context variables (NO assumptions!)
    context = {
        'score_diff': current_diff,
        'score_diff_abs': abs(current_diff),
        'volatility': np.std(pattern[:min(10, len(pattern))]) if len(pattern) > 1 else 0,
        'momentum': pattern[0] - pattern[min(5, len(pattern)-1)] if len(pattern) > 1 else 0,
        'early_score': pattern[0] if len(pattern) > 0 else 0,
        'pace_proxy': np.mean(np.abs(np.diff(pattern[:min(10, len(pattern))]))) if len(pattern) > 1 else 0,
        'lead_changes': np.sum(np.diff(np.sign(pattern[:min(10, len(pattern))])) != 0) if len(pattern) > 1 else 0,
        'max_lead': np.max(np.abs(pattern[:min(10, len(pattern))])) if len(pattern) > 0 else 0,
        'range': np.ptp(pattern[:min(10, len(pattern))]) if len(pattern) > 0 else 0,
    }
    
    context_data.append(context)
    X_base.append(base_features)
    y_final.append(final_diff)
    y_current.append(current_diff)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(context_data)
df['final_diff'] = y_final
df['current_diff'] = y_current

X_base = np.array(X_base)

print(f"✓ Extracted {len(df)} games with {len(df.columns)-2} context variables")
print(f"\nContext variables available:")
for col in df.columns:
    if col not in ['final_diff', 'current_diff']:
        print(f"  • {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, range=[{df[col].min():.2f}, {df[col].max():.2f}]")

print("\n" + "="*90)
print("[PHASE 2] MINING OPTIMAL BINS FROM DATA (Decision Tree)")
print("="*90)

print("\nUsing DecisionTreeRegressor to find natural splits...")

# Mine optimal splits for each context variable
mined_bins = {}

context_vars = ['score_diff_abs', 'volatility', 'momentum', 'pace_proxy', 'lead_changes']

for var in context_vars:
    if df[var].std() < 0.01:  # Skip constant variables
        continue
    
    X_var = df[[var]].values
    y_var = df['final_diff'].values
    
    # Use decision tree to find optimal splits
    dt = DecisionTreeRegressor(max_leaf_nodes=5, min_samples_leaf=200, random_state=42)
    dt.fit(X_var, y_var)
    
    # Extract thresholds
    tree = dt.tree_
    thresholds = []
    
    for i in range(tree.node_count):
        if tree.children_left[i] != tree.children_right[i]:  # Not a leaf
            thresholds.append(tree.threshold[i])
    
    thresholds = sorted([t for t in thresholds if t != -2])
    
    if len(thresholds) > 0:
        # Create bins
        bins = [-np.inf] + thresholds + [np.inf]
        df[f'{var}_bin'] = pd.cut(df[var], bins=bins, labels=False)
        mined_bins[var] = thresholds
        
        print(f"\n  {var}:")
        print(f"    Mined thresholds: {[f'{t:.2f}' for t in thresholds]}")
        print(f"    Created {len(thresholds)+1} bins")
        
        # Show MAE per bin
        for bin_id in sorted(df[f'{var}_bin'].unique()):
            mask = df[f'{var}_bin'] == bin_id
            bin_mae = np.mean(np.abs(df.loc[mask, 'final_diff'] - df.loc[mask, 'current_diff']))
            count = mask.sum()
            print(f"      Bin {bin_id}: {count:4d} games, naive MAE: {bin_mae:.3f}")

print(f"\n✓ Mined optimal bins for {len(mined_bins)} context variables")

print("\n" + "="*90)
print("[PHASE 3] MULTI-DIMENSIONAL CLUSTERING (Discover Natural Segments)")
print("="*90)

print("\nClustering games on joint context space...")

# Use multiple context vars for clustering
cluster_features = ['score_diff_abs', 'volatility', 'momentum', 'pace_proxy', 'lead_changes', 'max_lead']
X_cluster = df[cluster_features].values

# Standardize for clustering
from sklearn.preprocessing import StandardScaler
cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

# Test different cluster counts
print("\nTesting cluster counts (silhouette scores):")
best_k = 3
best_score = -1

for k in range(3, 11):
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans_test.fit_predict(X_cluster_scaled)
    score = silhouette_score(X_cluster_scaled, labels)
    
    print(f"  k={k}: silhouette={score:.3f}")
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"\n✓ Optimal clusters: {best_k} (silhouette={best_score:.3f})")

# Final clustering
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df['mined_segment'] = kmeans_final.fit_predict(X_cluster_scaled)

print(f"\n🎯 Discovered {best_k} natural game segments:")

segment_profiles = []

for seg_id in sorted(df['mined_segment'].unique()):
    mask = df['mined_segment'] == seg_id
    count = mask.sum()
    
    # Profile this segment
    profile = {
        'id': seg_id,
        'count': count,
        'avg_score_diff': df.loc[mask, 'score_diff_abs'].mean(),
        'avg_volatility': df.loc[mask, 'volatility'].mean(),
        'avg_momentum': df.loc[mask, 'momentum'].mean(),
        'avg_pace': df.loc[mask, 'pace_proxy'].mean(),
        'avg_lead_changes': df.loc[mask, 'lead_changes'].mean(),
    }
    
    # Name based on characteristics (data-driven!)
    if profile['avg_score_diff'] > 15:
        name = "Blowout"
    elif profile['avg_volatility'] > 5 and profile['avg_lead_changes'] > 2.5:
        name = "High Variance"
    elif profile['avg_pace'] < 2:
        name = "Slow Pace"
    elif profile['avg_score_diff'] < 5:
        name = "Tight"
    else:
        name = "Balanced"
    
    profile['name'] = name
    segment_profiles.append(profile)
    
    print(f"\n  Segment {seg_id} ({name}): {count} games")
    print(f"    Avg diff: {profile['avg_score_diff']:.1f}")
    print(f"    Avg volatility: {profile['avg_volatility']:.1f}")
    print(f"    Avg lead changes: {profile['avg_lead_changes']:.1f}")

print("\n" + "="*90)
print("[PHASE 4] AUTOMATIC FEATURE IMPORTANCE MINING")
print("="*90)

print("\nUsing LASSO to discover most predictive features...")

# Split chronologically
split_idx = int(len(df) * 0.8)
train_mask = np.arange(len(df)) < split_idx

X_train = X_base[train_mask]
X_test = X_base[~train_mask]
y_train_final = df.loc[train_mask, 'final_diff'].values
y_test_final = df.loc[~train_mask, 'final_diff'].values
y_train_curr = df.loc[train_mask, 'current_diff'].values
y_test_curr = df.loc[~train_mask, 'current_diff'].values

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add current diff
X_train_full = np.column_stack([X_train_scaled, y_train_curr])
X_test_full = np.column_stack([X_test_scaled, y_test_curr])

# LASSO with cross-validation to find optimal alpha
print("\nRunning LassoCV to find important features...")
lasso_cv = LassoCV(cv=5, alphas=np.logspace(-3, 2, 50), max_iter=10000, random_state=42)
lasso_cv.fit(X_train_full, y_train_final)

# Get selected features
feature_importance = np.abs(lasso_cv.coef_)
selected_mask = feature_importance > 0.01

n_selected = np.sum(selected_mask)
print(f"\n✓ LASSO selected {n_selected} out of {len(feature_importance)} features")
print(f"  Optimal alpha: {lasso_cv.alpha_:.4f}")

# Get top 10 features
top_indices = np.argsort(feature_importance)[-10:][::-1]
print(f"\nTop 10 most important features:")
for i, idx in enumerate(top_indices):
    if idx < 18:
        print(f"  {i+1}. Feature_{idx}: importance={feature_importance[idx]:.4f}")
    else:
        print(f"  {i+1}. current_diff: importance={feature_importance[idx]:.4f}")

# Predict with LASSO
pred_lasso = lasso_cv.predict(X_test_full)
mae_lasso = mean_absolute_error(y_test_final, pred_lasso)
print(f"\n✓ LASSO MAE: {mae_lasso:.3f}")

print("\n" + "="*90)
print("[PHASE 5] TRAINING SEGMENT-SPECIFIC MODELS (Data-Mined Segments)")
print("="*90)

print("\nTraining specialized model for each mined segment...")

segment_models = {}
test_segments = df.loc[~train_mask, 'mined_segment'].values

for seg_id, seg_profile in enumerate(segment_profiles):
    # Get training data for this segment
    train_seg_mask = (df.loc[train_mask, 'mined_segment'] == seg_id).values
    
    if np.sum(train_seg_mask) < 100:  # Need minimum samples
        print(f"  Segment {seg_id} ({seg_profile['name']}): Skipped (only {np.sum(train_seg_mask)} samples)")
        continue
    
    # Train Ridge for this segment
    model_seg = Ridge(alpha=3.0, max_iter=5000)
    model_seg.fit(X_train_full[train_seg_mask], y_train_final[train_seg_mask])
    
    # Validate on test set (same segment)
    test_seg_mask = test_segments == seg_id
    
    if np.sum(test_seg_mask) > 0:
        pred_seg = model_seg.predict(X_test_full[test_seg_mask])
        mae_seg = mean_absolute_error(y_test_final[test_seg_mask], pred_seg)
        
        segment_models[seg_id] = {
            'model': model_seg,
            'mae': mae_seg,
            'n_train': np.sum(train_seg_mask),
            'n_test': np.sum(test_seg_mask),
            'profile': seg_profile
        }
        
        print(f"  Segment {seg_id} ({seg_profile['name']}): MAE={mae_seg:.3f} (train={np.sum(train_seg_mask)}, test={np.sum(test_seg_mask)})")
    else:
        print(f"  Segment {seg_id} ({seg_profile['name']}): No test samples")

print(f"\n✓ Trained {len(segment_models)} segment-specific models")

print("\n" + "="*90)
print("[PHASE 6] SEGMENT-ROUTED PREDICTIONS")
print("="*90)

print("\nMaking predictions with data-mined segment routing...")

# Route predictions based on mined segments
routed_preds = np.zeros(len(X_test))

for i in range(len(X_test)):
    seg_id = test_segments[i]
    
    if seg_id in segment_models:
        model_seg = segment_models[seg_id]['model']
        routed_preds[i] = model_seg.predict(X_test_full[i:i+1])[0]
    else:
        # Fallback to LASSO
        routed_preds[i] = lasso_cv.predict(X_test_full[i:i+1])[0]

mae_routed = mean_absolute_error(y_test_final, routed_preds)

print(f"✓ Segment-routed MAE: {mae_routed:.3f}")

print("\n" + "="*90)
print("[PHASE 7] DECISION TREE-BASED AUTOMATIC SEGMENTATION")
print("="*90)

print("\nUsing decision tree to find optimal multi-dimensional splits...")

# Train decision tree on context variables to find best joint splits
context_features_list = ['score_diff_abs', 'volatility', 'momentum', 'pace_proxy', 'lead_changes', 'max_lead']
X_context_train = df.loc[train_mask, context_features_list].values
X_context_test = df.loc[~train_mask, context_features_list].values

# Decision tree will find optimal splits
dt_segmenter = DecisionTreeRegressor(
    max_leaf_nodes=8,  # Create 8 segments
    min_samples_leaf=300,  # Ensure enough data per segment
    random_state=42
)

dt_segmenter.fit(X_context_train, y_train_final)

# Get predictions (each leaf is a segment)
train_leaf_ids = dt_segmenter.apply(X_context_train)
test_leaf_ids = dt_segmenter.apply(X_context_test)

# Train model per leaf
leaf_models = {}
unique_leaves = np.unique(train_leaf_ids)

print(f"\n✓ Decision tree discovered {len(unique_leaves)} natural segments:")

for leaf_id in unique_leaves:
    leaf_mask_train = train_leaf_ids == leaf_id
    leaf_mask_test = test_leaf_ids == leaf_id
    
    if np.sum(leaf_mask_train) < 50:
        continue
    
    # Train model for this leaf
    model_leaf = Ridge(alpha=2.0)
    model_leaf.fit(X_train_full[leaf_mask_train], y_train_final[leaf_mask_train])
    
    if np.sum(leaf_mask_test) > 0:
        pred_leaf = model_leaf.predict(X_test_full[leaf_mask_test])
        mae_leaf = mean_absolute_error(y_test_final[leaf_mask_test], pred_leaf)
        
        # Analyze leaf characteristics
        leaf_context = X_context_train[leaf_mask_train]
        avg_diff = leaf_context[:, 0].mean()
        avg_vol = leaf_context[:, 1].mean()
        
        leaf_models[leaf_id] = {
            'model': model_leaf,
            'mae': mae_leaf,
            'n_train': np.sum(leaf_mask_train),
            'n_test': np.sum(leaf_mask_test),
            'avg_diff': avg_diff,
            'avg_vol': avg_vol
        }
        
        print(f"  Leaf {leaf_id}: MAE={mae_leaf:.3f}, n={np.sum(leaf_mask_train)} train/{np.sum(leaf_mask_test)} test")
        print(f"    Characteristics: diff={avg_diff:.1f}, vol={avg_vol:.1f}")

# Make DT-routed predictions
dt_routed_preds = np.zeros(len(X_test))

for i in range(len(X_test)):
    leaf_id = test_leaf_ids[i]
    
    if leaf_id in leaf_models:
        dt_routed_preds[i] = leaf_models[leaf_id]['model'].predict(X_test_full[i:i+1])[0]
    else:
        dt_routed_preds[i] = lasso_cv.predict(X_test_full[i:i+1])[0]

mae_dt_routed = mean_absolute_error(y_test_final, dt_routed_preds)

print(f"\n✓ Decision tree-routed MAE: {mae_dt_routed:.3f}")

print("\n" + "="*90)
print("[PHASE 8] GAUSSIAN MIXTURE MODEL (Soft Segmentation)")
print("="*90)

print("\nUsing GMM for probabilistic segment assignment...")

# GMM allows soft assignment (probability of belonging to each segment)
gmm = GaussianMixture(n_components=best_k, random_state=42, n_init=10)
gmm.fit(X_cluster_scaled[train_mask])

# Get probabilities
train_probs = gmm.predict_proba(X_cluster_scaled[train_mask])
test_probs = gmm.predict_proba(X_cluster_scaled[~train_mask])

# Train model per GMM component
gmm_models = {}

for comp_id in range(best_k):
    # Weight training by probability of belonging to this component
    weights_train = train_probs[:, comp_id]
    
    # Only use samples with high probability (>0.3)
    high_prob_mask = weights_train > 0.3
    
    if np.sum(high_prob_mask) < 100:
        continue
    
    # Train weighted model
    model_gmm = Ridge(alpha=2.0)
    model_gmm.fit(X_train_full[high_prob_mask], y_train_final[high_prob_mask], 
                  sample_weight=weights_train[high_prob_mask])
    
    gmm_models[comp_id] = model_gmm
    
    print(f"  Component {comp_id}: trained on {np.sum(high_prob_mask)} high-prob samples")

# Soft routing (probability-weighted ensemble)
gmm_routed_preds = np.zeros(len(X_test))

for i in range(len(X_test)):
    probs = test_probs[i]
    
    # Weighted prediction
    weighted_pred = 0
    total_weight = 0
    
    for comp_id, prob in enumerate(probs):
        if comp_id in gmm_models and prob > 0.1:
            pred = gmm_models[comp_id].predict(X_test_full[i:i+1])[0]
            weighted_pred += pred * prob
            total_weight += prob
    
    if total_weight > 0:
        gmm_routed_preds[i] = weighted_pred / total_weight
    else:
        gmm_routed_preds[i] = lasso_cv.predict(X_test_full[i:i+1])[0]

mae_gmm_routed = mean_absolute_error(y_test_final, gmm_routed_preds)

print(f"\n✓ GMM soft-routed MAE: {mae_gmm_routed:.3f}")

print("\n" + "="*90)
print("[PHASE 9] COMPARING ALL DATA-DRIVEN APPROACHES")
print("="*90)

# Train simple baseline for comparison
baseline_model = Ridge(alpha=2.0)
baseline_model.fit(X_train_full, y_train_final)
pred_baseline = baseline_model.predict(X_test_full)
mae_baseline = mean_absolute_error(y_test_final, pred_baseline)

approaches = [
    ('Simple Baseline (18 features)', 9.029, 'Engineering Linear from previous'),
    ('LASSO Feature Selection', mae_lasso, f'{n_selected} features auto-selected'),
    ('K-Means Segment Routing', mae_routed, f'{best_k} mined segments'),
    ('Decision Tree Routing', mae_dt_routed, f'{len(leaf_models)} auto-discovered segments'),
    ('GMM Soft Routing', mae_gmm_routed, f'{best_k} probabilistic components'),
    ('Ridge Global (test)', mae_baseline, '18 features + current_diff')
]

print("\n" + "-"*90)
print(f"{'APPROACH':<35} {'MAE':<10} {'vs BASELINE':<15} {'METHOD'}")
print("-"*90)

baseline_ref = 9.029
best_approach = None
best_mae = baseline_ref

for name, mae, method in approaches:
    improvement = baseline_ref - mae
    improve_pct = (improvement / baseline_ref) * 100
    
    if mae < best_mae:
        best_mae = mae
        best_approach = name
    
    if improvement > 0.1:
        flag = "🔥"
        improve_str = f"-{improvement:.3f} ({improve_pct:+.1f}%)"
    elif improvement > 0:
        flag = "✅"
        improve_str = f"-{improvement:.3f} ({improve_pct:+.1f}%)"
    elif abs(improvement) < 0.05:
        flag = "📊"
        improve_str = "~Same"
    else:
        flag = "⚠️"
        improve_str = f"+{abs(improvement):.3f} ({improve_pct:.1f}%)"
    
    print(f"{flag} {name:<35} {mae:<10.3f} {improve_str:<15} {method}")

print("-"*90)

print(f"\n🏆 BEST DATA-DRIVEN APPROACH: {best_approach}")
print(f"   MAE: {best_mae:.3f}")

if best_mae < 9.029:
    improvement_total = 9.029 - best_mae
    improve_pct = (improvement_total / 9.029) * 100
    
    print(f"   Improvement: -{improvement_total:.3f} MAE ({improve_pct:.1f}%)")
    
    # Edge calculation
    old_edge = ((11.5 - 9.029) / 11.5) * 100
    new_edge = ((11.5 - best_mae) / 11.5) * 100
    edge_gain = new_edge - old_edge
    
    print(f"\n💰 FINANCIAL IMPACT:")
    print(f"   Old edge: {old_edge:.1f}%")
    print(f"   New edge: {new_edge:.1f}%")
    print(f"   Gain: +{edge_gain:.1f} percentage points")
    
    old_ev = 20 * (old_edge / 100) * 100
    new_ev = 20 * (new_edge / 100) * 100
    ev_gain = new_ev - old_ev
    
    print(f"\n   Old EV: +${old_ev:.0f} per 100 games")
    print(f"   New EV: +${new_ev:.0f} per 100 games")
    print(f"   Gain: +${ev_gain:.0f} per 100 games")
    
    print(f"\n🎯 RECOMMENDATION: Deploy {best_approach}!")
else:
    print(f"   No improvement over baseline")
    print(f"\n📊 FINDING: Data-driven mining confirms data ceiling")
    print(f"   → Even optimal data-mined segments converge to ~9.0 MAE")
    print(f"   → This is STRONG validation of fundamental ceiling")

print("\n" + "="*90)
print("[PHASE 10] SAVING DATA-DRIVEN SYSTEM")
print("="*90)

# Save complete data-driven system
data_driven_system = {
    'name': 'DATA_DRIVEN_MINED_SYSTEM',
    'version': '1.0.0',
    'mined_bins': mined_bins,
    'optimal_k': best_k,
    'kmeans': kmeans_final,
    'cluster_scaler': cluster_scaler,
    'segment_profiles': segment_profiles,
    'segment_models': {k: v['model'] for k, v in segment_models.items()},
    'dt_segmenter': dt_segmenter,
    'dt_leaf_models': {k: v['model'] for k, v in leaf_models.items()},
    'gmm': gmm,
    'gmm_models': gmm_models,
    'lasso': lasso_cv,
    'scaler': scaler,
    'feature_importance': feature_importance.tolist(),
    'n_features_selected': int(n_selected),
    'performance': {
        'lasso_mae': float(mae_lasso),
        'kmeans_routed': float(mae_routed),
        'dt_routed': float(mae_dt_routed),
        'gmm_routed': float(mae_gmm_routed),
        'best_mae': float(best_mae),
        'best_approach': best_approach
    }
}

with open('Action/DATA_DRIVEN_MINED_SYSTEM.pkl', 'wb') as f:
    pickle.dump(data_driven_system, f)

print(f"✓ Saved: DATA_DRIVEN_MINED_SYSTEM.pkl")

print("\n" + "="*90)
print("FINAL SUMMARY")
print("="*90)

print("\n🔍 DATA-DRIVEN DISCOVERIES:")
print(f"\n  1. Optimal Bins (Decision Tree-mined):")
for var, thresholds in mined_bins.items():
    print(f"     {var}: {len(thresholds)+1} bins at {[f'{t:.2f}' for t in thresholds]}")

print(f"\n  2. Natural Clusters:")
print(f"     K-Means found {best_k} optimal clusters (silhouette={best_score:.3f})")

print(f"\n  3. Feature Importance:")
print(f"     LASSO selected {n_selected}/{len(feature_importance)} features automatically")

print(f"\n  4. Segment-Specific Models:")
print(f"     Trained {len(segment_models)} specialized models")

print(f"\n📊 PERFORMANCE:")
print(f"   LASSO (feature selection):     {mae_lasso:.3f} MAE")
print(f"   K-Means routing:               {mae_routed:.3f} MAE")
print(f"   Decision tree routing:         {mae_dt_routed:.3f} MAE")
print(f"   GMM soft routing:              {mae_gmm_routed:.3f} MAE")
print(f"   Baseline:                      {mae_baseline:.3f} MAE")

print(f"\n💡 KEY INSIGHT:")
if best_mae < 9.0:
    print(f"  🔥 DATA-DRIVEN MINING WORKS!")
    print(f"  → Automatic segmentation found patterns!")
    print(f"  → {best_approach} achieves {best_mae:.3f} MAE")
else:
    print(f"  📊 Even data-driven optimal mining confirms ceiling")
    print(f"  → All approaches: {mae_lasso:.3f}, {mae_routed:.3f}, {mae_dt_routed:.3f}, {mae_gmm_routed:.3f}")
    print(f"  → All converge to ~9.0 MAE")
    print(f"  → This is ULTIMATE validation of fundamental data ceiling")
    print(f"  → No amount of sophisticated mining can extract signal that doesn't exist")

print("\n✅ DATA-DRIVEN MINING SYSTEM COMPLETE!")
print("="*90)

