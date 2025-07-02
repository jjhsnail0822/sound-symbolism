#!/bin/bash

# sbatch --gres=gpu:2 -p big_suma_rtx3090 -q big_qos /scratch2/sheepswool/workspace/sound-symbolism/src/analysis/heatmap/run.sh --pty bash -i
python src/analysis/heatmap/semdim_heatmap.py --max-samples 5000 --data-type audio