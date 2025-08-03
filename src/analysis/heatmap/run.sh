#!/bin/bash
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 5000 --data-type audio --flip
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 3000 --data-type ipa --constructed
# python src/analysis/heatmap/batch_semdim_heatmap.py --max-samples 2000 --data-type audio --languages ko

python src/analysis/heatmap/multi_threading_attention_computation.py
# python src/analysis/heatmap/compute_attention_by_language.py
# python src/analysis/heatmap/batch_semdim_heatmap.py --max-samples 3000 --data-type audio --constructed --flip