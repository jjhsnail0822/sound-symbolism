#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=kld_constructed_ipa_and_audio
#SBATCH --qos=base_qos
#SBATCH -o /home/gv_nlp/projects/sound-symbolism/out/kld_constructed_ipa_and_audio_%j.out

source ~/.zshrc
cd /home/gv_nlp/projects/sound-symbolism
echo $PWD
conda activate ss
export PYTHONPATH=/home/gv_nlp/projects/sound-symbolism:$PYTHONPATH

for sem_dim in "good" "beautiful" "pleasant" "strong" "big" "rugged" "active" "fast" "sharp" "realistic" "structured" "ordinary" "interesting" "simple" "abrupt" "exciting" "hard" "happy" "harsh" "heavy" "inhibited" "masculine" "solid" "tense" "dangerous"
do
  echo "Processing semantic dimension: $sem_dim"
  python src/experiments/semantic_dimension/qwen_omni_inference_logit_lens_kld.py -w constructed -i ipa_and_audio -s"$sem_dim"
done 