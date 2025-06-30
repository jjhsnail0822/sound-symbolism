#!/bin/bash

# Semantic Dimension Heatmap Analysis Workflow
# This script demonstrates the complete workflow from matrix extraction to plotting

set -e  # Exit on any error

echo "=== Semantic Dimension Heatmap Analysis Workflow ==="
echo

# Configuration
MODEL_PATH="Qwen/Qwen2.5-Omni-7B"
DATA_PATH="data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"
OUTPUT_DIR="results/experiments/understanding/attention_heatmap"
DATA_TYPE="audio"
LANGUAGES="en"
MAX_SAMPLES=10  # Start with a small number for testing

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Data Type: $DATA_TYPE"
echo "  Languages: $LANGUAGES"
echo "  Max Samples: $MAX_SAMPLES"
echo

# Step 1: Extract attention matrices
echo "=== Step 1: Extracting Attention Matrices ==="
python src/analysis/heatmap/semdim_heatmap.py \
    --model "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --data-type "$DATA_TYPE" \
    --languages "$LANGUAGES" \
    --max-samples "$MAX_SAMPLES"

echo
echo "Matrix extraction completed!"
echo

# Step 2: Check what files were created
echo "=== Step 2: Checking Generated Files ==="
python src/analysis/heatmap/heatmap_plot.py \
    --output-dir "$OUTPUT_DIR" \
    --data-type "$DATA_TYPE" \
    --summary

echo

# Step 3: Generate heatmaps
echo "=== Step 3: Generating Heatmaps ==="
python src/analysis/heatmap/heatmap_plot.py \
    --output-dir "$OUTPUT_DIR" \
    --data-type "$DATA_TYPE" \
    --auto-process-all

echo
echo "=== Analysis Complete! ==="
echo "Results saved to: $OUTPUT_DIR"
echo
echo "You can now:"
echo "1. View individual heatmaps in the output directory"
echo "2. Run with different parameters:"
echo "   - Change --max-samples for more data"
echo "   - Change --languages for different languages"
echo "   - Change --data-type for different input types"
echo "3. Generate specific plots:"
echo "   python src/analysis/heatmap/heatmap_plot.py --word-tokens 'word' --dimension1 'dim1' --dimension2 'dim2' --lang en" 