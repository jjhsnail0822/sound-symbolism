#!/bin/bash
# sbatch --gres=gpu:1 -p suma_rtx4090 -q base_qos /scratch2/sheepswool/workspace/sound-symbolism/src/analysis/heatmap/run.sh --pty bash -i
# sbatch --gres=gpu:0 -p dell_cpu -q big_qos /scratch2/sheepswool/workspace/sound-symbolism/src/analysis/heatmap/run.sh --pty bash -i
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 5000 --data-type audio --flip
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 3000 --data-type ipa --constructed
# python src/analysis/heatmap/batch_semdim_heatmap.py --max-samples 2000 --data-type audio --languages ko

python src/analysis/heatmap/multi_threading_attention_computation.py
# python src/analysis/heatmap/compute_attention_by_language.py
# python src/analysis/heatmap/batch_semdim_heatmap.py --max-samples 3000 --data-type audio --constructed --flip