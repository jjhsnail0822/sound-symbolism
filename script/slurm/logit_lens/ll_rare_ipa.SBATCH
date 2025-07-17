#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=ll_rare_ipa
#SBATCH --qos=base_qos
#SBATCH -o /home/gv_nlp/projects/sound-symbolism/out/ll_rare_ipa_%j.out

source ~/.zshrc
cd /home/gv_nlp/projects/sound-symbolism
echo $PWD
conda activate ss
export PYTHONPATH=/home/gv_nlp/projects/sound-symbolism:$PYTHONPATH

python src/experiments/semantic_dimension/qwen_omni_inference_logit_lens.py -w rare -i ipa 