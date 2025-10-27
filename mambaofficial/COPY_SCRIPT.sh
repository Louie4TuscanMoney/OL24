#!/bin/bash

echo "🐍 COPYING ALL MAMBA FILES..."
echo ""

# Create subdirectories
mkdir -p mambaofficial/models
mkdir -p mambaofficial/training_data
mkdir -p mambaofficial/documentation
mkdir -p mambaofficial/scripts
mkdir -p mambaofficial/config

echo "📂 Copying model files..."
# Copy main Mamba models
cp -v Action/MAMBA_MENTALITY_SYSTEM.pkl mambaofficial/models/
cp -v Action/MAMBA_MENTALITY_SYSTEM_V1.pkl mambaofficial/models/

echo ""
echo "📊 Copying training data..."
# Copy all pattern/training data
cp -v Action/ENHANCED_PATTERNS_FULL.pkl mambaofficial/training_data/
cp -v Action/ENHANCED_PATTERNS_WITH_TEAM.pkl mambaofficial/training_data/
cp -v Action/patterns_2025_preseason_FULL_67_FEATURES.pkl mambaofficial/training_data/
cp -v Action/patterns_2025_preseason.pkl mambaofficial/training_data/
cp -v Action/COMPLETE_PATTERN_PIPELINE.pkl mambaofficial/training_data/
cp -v Action/ULTRA_ENHANCED_PATTERNS_V3_67_FEATURES.pkl mambaofficial/training_data/
cp -v Action/ULTRA_OPTIMIZED_PATTERNS_2021_2025.pkl mambaofficial/training_data/

echo ""
echo "📄 Copying documentation..."
# Copy Mamba documentation
cp -v "🎯_MAMBA_BALANCED_BETTING_OPTIMIZATION.md" mambaofficial/documentation/
cp -v "Action/🐍_MAMBA_VS_GREATNESS.md" mambaofficial/documentation/

echo ""
echo "🔧 Copying scripts..."
# Copy Mamba scripts
cp -v "Action/🐍_MAMBA_MENTALITY_LAUNCH.sh" mambaofficial/scripts/
cp -v "Action/🔧_RETRAIN_MAMBA_QUICK.py" mambaofficial/scripts/

echo ""
echo "⚙️ Copying config..."
# Copy Mamba config
cp -v "5. Live System/mamba_betting_config.py" mambaofficial/config/

echo ""
echo "✅ MAMBA OFFICIAL FOLDER COMPLETE!"

