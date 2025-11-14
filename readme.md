# [AAAI-26 Oral] Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism
Code and data for _“Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism”_. [[paper]](https://arxiv.org/abs/2511.10045)
- Venue: AAAI 2026 Main Technical Track (Oral)
- Authors: Jinhong Jeong*, Sunghyun Lee*, Jaeyoung Lee, Seonah Han, Youngjae Yu

![Image](https://github.com/user-attachments/assets/915a27ea-f91c-46c8-81ae-a0adfe28fd54)

## Index

- [About](#about)
- [Overview](#overview)
- [Interesting Results](#interesting-results)
- [Usage](#usage)
  - [Installation](#installation)
  - [Scripts](#scripts)
  - [Datasets](#datasets)
- [Environments](#environments)
  - [Pre-Requisites](#pre-requisites)
  - [Development Environment](#development-environment)
- [Citation](#citation)

##  About
This repository provides the code, data preprocessing, evaluation scripts, and analysis tools used in the paper _“Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism”_. The project introduces **LEX-ICON**, a multilingual mimetic-word dataset (natural and constructed words) and experiments that probe Multimodal Large Language Models (MLLMs) for phonetic–semantic associations and layer-wise attention patterns. See the paper for full details and results.

## Overview
This artifact packages everything needed to verify the AAAI-26 paper results: the curated LEX-ICON dataset, multimodal prompts (text, IPA, and audio), MCQ evaluation pipelines, and attention analysis scripts.  
- Self-contained reproduction bundle with frozen dependencies.  
- Supports both local checkpoints (Qwen/Qwen2.5-Omni-7B) and hosted APIs (GPT-4o-mini, Gemini 1.5 Pro).  
- Provides plotting utilities for semantic-dimension trends, modality advantages, and layer-wise attention.

## Interesting Results
- **Phonetic intuition holds across natural and constructed words.** Macro-F1 scores exceed the 0.50 baseline in 84.2% of semantic dimensions for natural mimetic words and 68.4% for constructed pseudo-words, showing that MLLMs infer meaning from both memorized and novel word forms rather than memorization alone.
- **Human-aligned trends.** Qwen2.5-Omni-7B achieves the strongest Pearson correlation (up to r = 0.579) with human judgements across semantic dimensions, while all models remain closest to human patterns on constructed words, supporting the reliability of the pseudo ground truth.
- **Modality-specific strengths.** Audio inputs particularly boost performance on acoustically grounded dimensions (e.g., big vs. small, fast vs. slow), whereas textual inputs (orthographic and IPA) excel on articulatory or visually driven contrasts such as sharp vs. round and beautiful vs. ugly.
- **Attention fraction score follows iconic phonemes.** Layer-wise analyses of Qwen2.5-Omni-7B show attention fraction scores above 0.50 for canonical sound-symbolic pairs (e.g., /p/ ↔ sharp, /m/ ↔ round), with late decoder layers emphasizing these associations more strongly for constructed words, evidencing internal alignment between phonemes and inferred meanings.

## Usage

### Installation
1. Clone the repository
```bash
git clone https://github.com/jjhsnail0822/sound-symbolism.git
cd sound-symbolism
```

2. Create the curated conda environment (includes CUDA, audio, and visualization deps)
```bash
conda env create -f environment.yml
conda activate symbolism
```
   - If you prefer `pip`, you can fall back to `pip install -r requirements.txt` from the repo root, but several audio dependencies are easier with conda.

### Scripts

**Core MCQ experiments**
```bash
python src/experiments/semantic_dimension/mcq_experiments_all_at_once.py \
  --model "Qwen/Qwen2.5-Omni-7B" \
  --gpu 1 \
  --retry-failed-answers
```
   - Replace `--model` with `gpt-4o` or `gemini-2.5-flash` and add `--api` once corresponding API keys are configured.

**Attention analysis**
```bash
bash src/analysis/heatmap/run.sh                     # generates attention caches
```

**Reproduce final plots**
```bash
python src/analysis/plotting/layer_attention_score_plot.py
python src/analysis/plotting/modality_advantage_plot.py
python src/analysis/plotting/semantic_dimension_plot.py
```

###  Datasets

**Paths to the LEX-ICON dataset**
```
# Natural Word Group
data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json
# Constructed Word Group
data/processed/art/semantic_dimension/semantic_dimension_binary_gt.json
```

This repository does not include TTS audio files due to the file sizes. You can easily reproduce the audio data with the codes in `src/dataset/audio/`.

## Environments

### Pre-Requisites
- git (>=2.30)
- Python 3.9 / 3.10+
- CUDA-enabled GPU recommended for model inference (e.g., RTX 3090 / 4090)
- Linux / macOS recommended for reproduction scripts (audio toolchain compatibility)
- Optional: API keys for closed-source models (if using GPT/Gemini via API)

### Development Environment
1. Fork the repository and clone your fork:
   ```bash
   git clone git@github.com:<your-handle>/sound-symbolism.git
   cd sound-symbolism
   git remote add upstream https://github.com/jjhsnail0822/sound-symbolism.git
   ```
2. Create and activate the research environment:
   ```bash
   conda env create -f environment.yml
   conda activate symbolism
   export PYTHONPATH=$(pwd)/src  # add to your shell profile for convenience
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feat/<short-description>
   ```

| Path | Purpose |
|------|---------|
| `data/processed` | Canonical LEX-ICON subsets (natural + constructed) used in the experiments. |
| `data/prompts` | Prompt templates and multiple-choice questions delivered to the models. |
| `src/dataset` | Scripts for regenerating phonetic resources, prompts, and TTS wav files. |
| `src/experiments` | Entry points for MCQ inference, human evaluation parsing, and API integrations. |
| `src/analysis` | Attention alignment, visualization, and statistics utilities. |
| `results` | Default output directory for metrics, figures, and serialized attention scores. |

## Citation
If this repository supports your research, please cite:
```
@misc{jeong2025languagemodelsassociatesound,
      title={Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism}, 
      author={Jinhong Jeong and Sunghyun Lee and Jaeyoung Lee and Seonah Han and Youngjae Yu},
      year={2025},
      eprint={2511.10045},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.10045}, 
}
```
For any questions or issues, feel free to contact the authors!
