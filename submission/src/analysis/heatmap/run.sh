#!/bin/bash

# python src/analysis/heatmap/semdim_heatmap.py --max-samples 5000 --data-type audio --flip
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 3000 --data-type ipa --constructed
# python src/analysis/heatmap/batch_semdim_heatmap.py --start-index 0 --max-samples 5000 --data-type audio --constructed
python src/analysis/heatmap/compute_attention_by_language.py