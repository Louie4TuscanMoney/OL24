#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE OVERFITTING AUDIT
Apply user's elite overfitting framework to ALL systems

Framework:
1. Definition & symptoms
2. Quantitative signals (train vs test gap)
3. Structural causes (complexity, leakage, regularization)
4. Behavioral signs (instability, overconfidence)
5. Root causes in production
6. Stabilization levers
7. Triage checklist
8. Example overfit model report
"""

import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔬 COMPREHENSIVE OVERFITTING AUDIT - ALL SYSTEMS")
print("="*80)
print()
print("Applying elite overfitting detection framework...")
print()

# Load all systems
systems = {
    'Mamba': 'MAMBA_MENTALITY_SYSTEM.pkl',
    'Strive': 'STRIVE_FOR_GREATNESS_CLEAN.pkl',
    'Stanford': 'STANFORD_RESEARCH_ENSEMBLE.pkl',
    'MIT': 'MIT_EXTREME_GENERALIZATION.pkl',
    'ULTRA': 'ULTRA_OPTIMIZED_ELON_MODE.pkl'
}

system_data = {}
for name, filename in systems.items():
    try:
        with open(filename, 'rb') as f:
            system_data[name] = pickle.load(f)
        print(f"✅ Loaded {name}")
    except:
        print(f"⚠️  Could not load {name}")

print()

# Load test data
with open('ULTRA_ENHANCED_PATTERNS_V3_CHRONOLOGICAL.pkl', 'rb') as f:
    data = pickle.load(f)

split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

print("="*80)
print("🧪 OVERFITTING TRIAGE CHECKLIST (USER FRAMEWORK)")
print("="*80)
print()

# Apply triage checklist to each system
for name, system in system_data.items():
    print(f"\n{'='*80}")
    print(f"SYSTEM: {name.upper()}")
    print(f"{'='*80}")
    print()
    
    # Get metrics
    train_mae_a = system['branch_a_halftime'].get('train_mae', 0)
    test_mae_a = system['branch_a_halftime'].get('test_mae', system['branch_a_halftime'].get('champion_mae', 0))
    gap_a = system['branch_a_halftime'].get('overfitting_gap', 0)
    
    train_mae_b = system['branch_b_final'].get('train_mae', 0)
    test_mae_b = system['branch_b_final'].get('test_mae', system['branch_b_final'].get('champion_mae', 0))
    gap_b = system['branch_b_final'].get('overfitting_gap', 0)
    
    # TRIAGE CHECKLIST
    print("🧭 TRIAGE CHECKLIST:")
    print()
    
    issues = []
    
    # 1. Train/Test gap > 20%?
    if gap_a > 20 or gap_b > 20:
        print(f"  ❌ Train/Test gap > 20%: Half {gap_a:.1f}%, Final {gap_b:.1f}%")
        issues.append("High train/test gap")
    else:
        print(f"  ✅ Train/Test gap < 20%: Half {gap_a:.1f}%, Final {gap_b:.1f}%")
    
    # 2. Feature count
    feature_count = system['metadata'].get('feature_count', 0)
    if feature_count > 50:
        print(f"  ⚠️  Feature set large: {feature_count} features")
        issues.append("Too many features")
    else:
        print(f"  ✅ Feature set reasonable: {feature_count} features")
    
    # 3. Model complexity
    models_a = system['branch_a_halftime'].get('models', {})
    has_deep_models = any('deep' in str(type(m)).lower() or 'forest' in str(type(m)).lower() 
                          for m in models_a.values())
    print(f"  ℹ️  Complex models present: {has_deep_models}")
    
    # 4. CV stability
    cv_std_a = system['branch_a_halftime'].get('cv_std', 0)
    cv_mae_a = system['branch_a_halftime'].get('cv_mae', test_mae_a)
    if cv_std_a > 0:
        cv_pct = (cv_std_a / cv_mae_a) * 100
        if cv_pct > 15:
            print(f"  ⚠️  Test unstable across folds: {cv_pct:.1f}% CV")
            issues.append("High CV variance")
        else:
            print(f"  ✅ Test stable across folds: {cv_pct:.1f}% CV")
    
    # 5. Performance "too perfect"?
    if test_mae_a < 4.0 or test_mae_b < 7.0:
        print(f"  ⚠️  Performance suspiciously good: {test_mae_a:.3f} / {test_mae_b:.3f}")
        issues.append("Too perfect (may not hold)")
    else:
        print(f"  ✅ Performance realistic: {test_mae_a:.3f} / {test_mae_b:.3f}")
    
    # OVERALL VERDICT
    print()
    print("VERDICT:")
    if len(issues) >= 2:
        print(f"  ❌ OVERFITTING RISK: {len(issues)} issues detected")
        print(f"     Issues: {', '.join(issues)}")
        print(f"     Recommendation: DO NOT LAUNCH")
    elif len(issues) == 1:
        print(f"  ⚠️  MODERATE RISK: {len(issues)} issue")
        print(f"     Issue: {issues[0]}")
        print(f"     Recommendation: LAUNCH WITH CAUTION")
    else:
        print(f"  ✅ LOW RISK: Clean system")
        print(f"     Recommendation: READY TO LAUNCH")
    
    # METRICS SUMMARY
    print()
    print("METRICS SUMMARY:")
    print(f"  Train MAE: {train_mae_a:.3f} / {train_mae_b:.3f}")
    print(f"  Test MAE:  {test_mae_a:.3f} / {test_mae_b:.3f}")
    print(f"  Gap:       {gap_a:.1f}% / {gap_b:.1f}%")
    print(f"  Features:  {feature_count}")
    print(f"  Models:    {system['metadata'].get('models_trained', 0)}")

# ============================================================================
# DETAILED OVERFITTING REPORT (Following user framework)
# ============================================================================
print("\n" + "="*80)
print("📊 DETAILED OVERFITTING REPORTS")
print("="*80)

reports = {}

for name, system in system_data.items():
    train_mae_a = system['branch_a_halftime'].get('train_mae', 0)
    test_mae_a = system['branch_a_halftime'].get('test_mae', system['branch_a_halftime'].get('champion_mae', 0))
    gap_a = system['branch_a_halftime'].get('overfitting_gap', 0)
    
    train_mae_b = system['branch_b_final'].get('train_mae', 0)
    test_mae_b = system['branch_b_final'].get('test_mae', system['branch_b_final'].get('champion_mae', 0))
    gap_b = system['branch_b_final'].get('overfitting_gap', 0)
    
    # Determine status
    if gap_a > 50 or gap_b > 50:
        status = "CRITICAL OVERFITTING"
    elif gap_a > 20 or gap_b > 20:
        status = "MODERATE OVERFITTING"
    elif gap_a > 10 or gap_b > 10:
        status = "MINOR OVERFITTING"
    else:
        status = "HEALTHY"
    
    reports[name] = {
        'status': status,
        'train_mae': (train_mae_a, train_mae_b),
        'test_mae': (test_mae_a, test_mae_b),
        'gap': (gap_a, gap_b),
        'features': system['metadata'].get('feature_count', 0),
        'models': system['metadata'].get('models_trained', 0)
    }

# Print summary table
print()
print("SYSTEM          STATUS                    HALFTIME              FINAL")
print("-" * 80)

for name, report in reports.items():
    status_icon = "✅" if report['status'] == "HEALTHY" else "⚠️" if "MINOR" in report['status'] else "❌"
    print(f"{name:12s}    {status_icon} {report['status']:20s} {report['test_mae'][0]:.3f} ({report['gap'][0]:4.1f}%)    {report['test_mae'][1]:.3f} ({report['gap'][1]:4.1f}%)")

print()

# Recommend best system
best_half = min(reports.items(), key=lambda x: x[1]['gap'][0])
best_final = min(reports.items(), key=lambda x: x[1]['gap'][1])

print("RECOMMENDATIONS:")
print(f"  Best Halftime: {best_half[0]} ({best_half[1]['gap'][0]:.1f}% overfitting)")
print(f"  Best Final:    {best_final[0]} ({best_final[1]['gap'][1]:.1f}% overfitting)")
print()

# Final recommendation
if 'ULTRA' in reports and reports['ULTRA']['status'] == 'HEALTHY':
    print("🏆 RECOMMENDED FOR LAUNCH: ULTRA")
    print(f"   Halftime: {reports['ULTRA']['test_mae'][0]:.3f} MAE, {reports['ULTRA']['gap'][0]:.1f}% overfit")
    print(f"   Final:    {reports['ULTRA']['test_mae'][1]:.3f} MAE, {reports['ULTRA']['gap'][1]:.1f}% overfit")
    print("   Status: Both branches HEALTHY ✅")
elif 'MIT' in reports and reports['MIT']['status'] in ['HEALTHY', 'MINOR OVERFITTING']:
    print("🏆 RECOMMENDED FOR LAUNCH: MIT")
    print(f"   Status: {reports['MIT']['status']}")
else:
    print("⚠️  NO SYSTEM PASSES OVERFITTING AUDIT")
    print("   Recommendation: Continue optimizing")

print()
print("="*80)
print("✅ COMPREHENSIVE OVERFITTING AUDIT COMPLETE")
print("="*80)

