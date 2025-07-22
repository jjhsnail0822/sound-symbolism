#!/bin/bash

# sbatch --gres=gpu:1 -p suma_rtx4090 -q big_qos /scratch2/sheepswool/workspace/sound-symbolism/src/analysis/heatmap/run.sh --pty bash -i
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 5000 --data-type audio --flip
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 3000 --data-type ipa --constructed
python src/analysis/heatmap/batch_semdim_heatmap.py --start-index 0 --max-samples 1000 --data-type audio --languages ko --flip