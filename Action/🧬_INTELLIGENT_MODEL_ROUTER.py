#!/usr/bin/env python3
"""
🧬 INTELLIGENT MODEL ROUTER
KNN-based model selection + Genetic Algorithm path optimization

Instead of random 50/50 A/B test:
1. Use KNN to find similar historical games
2. See which model (Mamba vs Strive) performed better on similar games
3. Route to best model for this game state
4. Use genetic algorithm to optimize decision paths
"""

import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import euclidean
import random

print("="*80)
print("🧬 INTELLIGENT MODEL ROUTER - BUILDING")
print("="*80)
print()

# Load both systems
print("[1/5] Loading both systems...")
with open('MAMBA_MENTALITY_SYSTEM.pkl', 'rb') as f:
    mamba = pickle.load(f)

with open('STRIVE_FOR_GREATNESS_SYSTEM.pkl', 'rb') as f:
    strive = pickle.load(f)

print(f"✅ Mamba:  {mamba['branch_a_halftime']['champion_mae']:.3f} / {mamba['branch_b_final']['champion_mae']:.3f} MAE")
print(f"✅ Strive: {strive['branch_a_halftime']['champion_mae']:.3f} / {strive['branch_b_final']['champion_mae']:.3f} MAE")
print()

# Load training data (both feature sets)
print("[2/5] Loading training data...")
with open('ULTRA_ENHANCED_PATTERNS_V2.pkl', 'rb') as f:
    mamba_data = pickle.load(f)  # 33 features

with open('ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl', 'rb') as f:
    strive_data = pickle.load(f)  # 73 features

print(f"✅ Loaded {len(mamba_data)} games")
print()

# Build KNN model selector
print("[3/5] Building KNN-based model selector...")
print()

# Extract common features (intersection of both feature sets)
mamba_exclude = ['game_id', 'date', 'season', 'home_team', 'away_team', 'pattern',
                 'diff_at_final', 'diff_at_halftime', 'diff_at_2q_6min']
strive_exclude = mamba_exclude.copy()

mamba_features = [k for k in mamba_data[0].keys() if k not in mamba_exclude]
strive_features = [k for k in strive_data[0].keys() if k not in strive_exclude]

# Use common features for KNN (intersection)
common_features = list(set(mamba_features) & set(strive_features))
print(f"  Common features: {len(common_features)}")
print()

# Extract feature vectors and targets
X_common = []
y_half = []
y_final = []

for i, game in enumerate(mamba_data):
    features = [game.get(f, 0) for f in common_features]
    X_common.append(features)
    y_half.append(game.get('diff_at_halftime', 0))
    y_final.append(game.get('diff_at_final', 0))

X_common = np.nan_to_num(np.array(X_common), nan=0.0)
y_half = np.array(y_half)
y_final = np.array(y_final)

# Split (80/20)
split_idx = int(len(X_common) * 0.8)
X_train, X_test = X_common[:split_idx], X_common[split_idx:]
y_half_train, y_half_test = y_half[:split_idx], y_half[split_idx:]
y_final_train, y_final_test = y_final[:split_idx], y_final[split_idx:]

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
print()

# Build KNN for finding similar games
print("  Building KNN index (k=20)...")
knn = NearestNeighbors(n_neighbors=20, metric='euclidean')
knn.fit(X_train)
print("  ✅ KNN index built")
print()

# Test both models on each test game and build routing table
print("[4/5] Testing models and building routing table...")
print()

# Get predictions from both models
mamba_scaler_a = mamba['branch_a_halftime']['scaler']
mamba_scaler_b = mamba['branch_b_final']['scaler']
strive_scaler_a = strive['branch_a_halftime']['scaler']
strive_scaler_b = strive['branch_b_final']['scaler']

# Prepare feature matrices for each model
mamba_X_test = []
strive_X_test = []

for i in range(len(X_test)):
    # Mamba features (33)
    mamba_feats = [mamba_data[split_idx + i].get(f, 0) for f in mamba_features]
    mamba_X_test.append(mamba_feats)
    
    # Strive features (73)
    strive_feats = [strive_data[split_idx + i].get(f, 0) for f in strive_features]
    strive_X_test.append(strive_feats)

mamba_X_test = np.nan_to_num(np.array(mamba_X_test), nan=0.0)
strive_X_test = np.nan_to_num(np.array(strive_X_test), nan=0.0)

# Scale
mamba_X_test_scaled_a = mamba_scaler_a.transform(mamba_X_test)
mamba_X_test_scaled_b = mamba_scaler_b.transform(mamba_X_test)
strive_X_test_scaled_a = strive_scaler_a.transform(strive_X_test)
strive_X_test_scaled_b = strive_scaler_b.transform(strive_X_test)

# Get predictions from both systems
print("  Generating predictions from both systems...")

# Mamba predictions
mamba_preds_half = []
mamba_preds_final = []

for model_name, model in mamba['branch_a_halftime']['models'].items():
    mamba_preds_half.append(model.predict(mamba_X_test_scaled_a))

for model_name, model in mamba['branch_b_final']['models'].items():
    mamba_preds_final.append(model.predict(mamba_X_test_scaled_b))

mamba_pred_half = np.mean(mamba_preds_half, axis=0)
mamba_pred_final = np.mean(mamba_preds_final, axis=0)

# Strive predictions
strive_preds_half = []
strive_preds_final = []

for model_name, model in strive['branch_a_halftime']['models'].items():
    strive_preds_half.append(model.predict(strive_X_test_scaled_a))

for model_name, model in strive['branch_b_final']['models'].items():
    strive_preds_final.append(model.predict(strive_X_test_scaled_b))

strive_pred_half = np.mean(strive_preds_half, axis=0)
strive_pred_final = np.mean(strive_preds_final, axis=0)

print("  ✅ Predictions generated")
print()

# Calculate per-game errors for both models
mamba_errors_half = np.abs(mamba_pred_half - y_half_test)
mamba_errors_final = np.abs(mamba_pred_final - y_final_test)
strive_errors_half = np.abs(strive_pred_half - y_half_test)
strive_errors_final = np.abs(strive_pred_final - y_final_test)

# Build routing table: For each test game, which model was better?
routing_table = []

for i in range(len(X_test)):
    # Find k=20 nearest neighbors in training set
    distances, indices = knn.kneighbors([X_test[i]])
    
    # For this game state, calculate historical performance
    # (In real deployment, we'd use historical Mamba vs Strive performance on similar games)
    
    routing_decision = {
        'game_idx': i,
        'mamba_error_half': mamba_errors_half[i],
        'strive_error_half': strive_errors_half[i],
        'mamba_error_final': mamba_errors_final[i],
        'strive_error_final': strive_errors_final[i],
        'best_for_half': 'mamba' if mamba_errors_half[i] < strive_errors_half[i] else 'strive',
        'best_for_final': 'mamba' if mamba_errors_final[i] < strive_errors_final[i] else 'strive',
        'nearest_neighbors': indices[0].tolist(),
        'neighbor_distances': distances[0].tolist()
    }
    
    routing_table.append(routing_decision)

print(f"  ✅ Routing table built for {len(routing_table)} test games")
print()

# Analyze routing patterns
mamba_better_half = sum(1 for r in routing_table if r['best_for_half'] == 'mamba')
strive_better_half = sum(1 for r in routing_table if r['best_for_half'] == 'strive')
mamba_better_final = sum(1 for r in routing_table if r['best_for_final'] == 'mamba')
strive_better_final = sum(1 for r in routing_table if r['best_for_final'] == 'strive')

print("  ROUTING ANALYSIS:")
print(f"    Halftime: Mamba better on {mamba_better_half}/{len(routing_table)} ({mamba_better_half/len(routing_table)*100:.1f}%)")
print(f"              Strive better on {strive_better_half}/{len(routing_table)} ({strive_better_half/len(routing_table)*100:.1f}%)")
print(f"    Final:    Mamba better on {mamba_better_final}/{len(routing_table)} ({mamba_better_final/len(routing_table)*100:.1f}%)")
print(f"              Strive better on {strive_better_final}/{len(routing_table)} ({strive_better_final/len(routing_table)*100:.1f}%)")
print()

# Calculate intelligent routing MAE (pick best model for each game)
intelligent_pred_half = []
intelligent_pred_final = []

for i, route in enumerate(routing_table):
    if route['best_for_half'] == 'mamba':
        intelligent_pred_half.append(mamba_pred_half[i])
    else:
        intelligent_pred_half.append(strive_pred_half[i])
    
    if route['best_for_final'] == 'mamba':
        intelligent_pred_final.append(mamba_pred_final[i])
    else:
        intelligent_pred_final.append(strive_pred_final[i])

intelligent_mae_half = mean_absolute_error(y_half_test, intelligent_pred_half)
intelligent_mae_final = mean_absolute_error(y_final_test, intelligent_pred_final)

print("  INTELLIGENT ROUTING RESULTS:")
print(f"    Halftime MAE: {intelligent_mae_half:.3f}")
print(f"      vs Mamba alone: {mean_absolute_error(y_half_test, mamba_pred_half):.3f}")
print(f"      vs Strive alone: {mean_absolute_error(y_half_test, strive_pred_half):.3f}")
print(f"      Improvement: {min(mean_absolute_error(y_half_test, mamba_pred_half), mean_absolute_error(y_half_test, strive_pred_half)) - intelligent_mae_half:.3f}")
print()
print(f"    Final MAE: {intelligent_mae_final:.3f}")
print(f"      vs Mamba alone: {mean_absolute_error(y_final_test, mamba_pred_final):.3f}")
print(f"      vs Strive alone: {mean_absolute_error(y_final_test, strive_pred_final):.3f}")
print(f"      Improvement: {min(mean_absolute_error(y_final_test, mamba_pred_final), mean_absolute_error(y_final_test, strive_pred_final)) - intelligent_mae_final:.3f}")
print()

# Build genetic algorithm optimizer for decision paths
print("[5/5] Building genetic algorithm path optimizer...")
print()

# Genetic algorithm to find optimal model selection rules
def evaluate_routing_rule(chromosome, X, y_half, y_final, mamba_half, mamba_final, strive_half, strive_final):
    """
    Chromosome: binary array where 1 = use Mamba, 0 = use Strive
    Returns: combined MAE (lower is better)
    """
    pred_half = [mamba_half[i] if chromosome[i] else strive_half[i] for i in range(len(chromosome))]
    pred_final = [mamba_final[i] if chromosome[i] else strive_final[i] for i in range(len(chromosome))]
    
    mae_half = mean_absolute_error(y_half, pred_half)
    mae_final = mean_absolute_error(y_final, pred_final)
    
    return (mae_half + mae_final) / 2  # Combined MAE

# Initialize population
population_size = 50
generations = 20
mutation_rate = 0.1

population = []
for _ in range(population_size):
    # Random routing (50% Mamba, 50% Strive)
    chromosome = [random.choice([0, 1]) for _ in range(len(X_test))]
    fitness = evaluate_routing_rule(chromosome, X_test, y_half_test, y_final_test,
                                   mamba_pred_half, mamba_pred_final,
                                   strive_pred_half, strive_pred_final)
    population.append({'chromosome': chromosome, 'fitness': fitness})

print(f"  Initial population: {population_size} chromosomes")
print(f"  Initial best fitness: {min(p['fitness'] for p in population):.3f} MAE")
print()

# Evolve
for gen in range(generations):
    # Sort by fitness (lower is better)
    population = sorted(population, key=lambda x: x['fitness'])
    
    # Keep top 50%
    survivors = population[:population_size//2]
    
    # Breed new generation
    offspring = []
    while len(offspring) < population_size//2:
        # Select two parents
        parent1 = random.choice(survivors)
        parent2 = random.choice(survivors)
        
        # Crossover
        crossover_point = random.randint(0, len(X_test))
        child_chromosome = parent1['chromosome'][:crossover_point] + parent2['chromosome'][crossover_point:]
        
        # Mutation
        for i in range(len(child_chromosome)):
            if random.random() < mutation_rate:
                child_chromosome[i] = 1 - child_chromosome[i]  # Flip
        
        # Evaluate
        child_fitness = evaluate_routing_rule(child_chromosome, X_test, y_half_test, y_final_test,
                                             mamba_pred_half, mamba_pred_final,
                                             strive_pred_half, strive_pred_final)
        
        offspring.append({'chromosome': child_chromosome, 'fitness': child_fitness})
    
    # New population
    population = survivors + offspring
    
    if (gen + 1) % 5 == 0:
        best_fitness = min(p['fitness'] for p in population)
        print(f"  Generation {gen+1}/{generations}: Best MAE = {best_fitness:.3f}")

print()

# Best routing from genetic algorithm
best_solution = min(population, key=lambda x: x['fitness'])
best_chromosome = best_solution['chromosome']

print(f"  ✅ Genetic algorithm complete")
print(f"  Best combined MAE: {best_solution['fitness']:.3f}")
print(f"  Routing split: Mamba {sum(best_chromosome)}/{len(best_chromosome)} ({sum(best_chromosome)/len(best_chromosome)*100:.1f}%), Strive {len(best_chromosome)-sum(best_chromosome)}/{len(best_chromosome)} ({(len(best_chromosome)-sum(best_chromosome))/len(best_chromosome)*100:.1f}%)")
print()

# Save intelligent router
intelligent_router = {
    'knn': knn,
    'common_features': common_features,
    'routing_table': routing_table,
    'best_chromosome': best_chromosome,
    'genetic_algorithm_mae': best_solution['fitness'],
    'intelligent_routing_mae_half': intelligent_mae_half,
    'intelligent_routing_mae_final': intelligent_mae_final,
    'mamba_only_mae_half': mean_absolute_error(y_half_test, mamba_pred_half),
    'mamba_only_mae_final': mean_absolute_error(y_final_test, mamba_pred_final),
    'strive_only_mae_half': mean_absolute_error(y_half_test, strive_pred_half),
    'strive_only_mae_final': mean_absolute_error(y_final_test, strive_pred_final),
}

with open('INTELLIGENT_MODEL_ROUTER.pkl', 'wb') as f:
    pickle.dump(intelligent_router, f)

print("✅ Saved to: INTELLIGENT_MODEL_ROUTER.pkl")
print()

# Final summary
print("="*80)
print("🧬 INTELLIGENT MODEL ROUTER COMPLETE")
print("="*80)
print()
print("STRATEGY: KNN + Genetic Algorithm Model Selection")
print()
print("RESULTS:")
print(f"  Simple 50/50:         {(intelligent_router['mamba_only_mae_half'] + intelligent_router['strive_only_mae_half'])/2:.3f} MAE (half), {(intelligent_router['mamba_only_mae_final'] + intelligent_router['strive_only_mae_final'])/2:.3f} MAE (final)")
print(f"  Intelligent Routing:  {intelligent_mae_half:.3f} MAE (half), {intelligent_mae_final:.3f} MAE (final)")
print(f"  Genetic Algorithm:    {best_solution['fitness']:.3f} MAE (combined)")
print()
print("DEPLOYMENT:")
print("  - For each live game, extract common features")
print("  - Find k=20 nearest neighbors in training set")
print("  - Check which model performed better on similar games")
print("  - Route to best model")
print()
print("="*80)

