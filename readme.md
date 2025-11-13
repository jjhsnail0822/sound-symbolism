# Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism
- Code and data for _“Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism”_ (AAAI 2026, Oral).
- Short intro: research code + dataset (LEX-ICON) + evaluation & attention-analysis scripts

## :ledger: Index

- [About](#beginner-about)
- [Overview](#mag_right-overview)
- [Paper](#bookmark_tabs-paper)
- [Interesting Results](#tada-interesting-results)
- [Usage](#zap-usage)
  - [Installation](#electric_plug-installation)
- [Development](#wrench-development)
  - [Pre-Requisites](#notebook-pre-requisites)
  - [Development Environment](#nut_and_bolt-development-environment)
  - [File Structure](#file_folder-file-structure)
- [Citation](#bookmark-citation)

##  :beginner: About
This repository provides the code, data preprocessing, evaluation scripts, and analysis tools used in the paper _“Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism”_. The project introduces **LEX-ICON**, a multilingual mimetic-word dataset (natural + constructed words) and experiments that probe Multimodal Large Language Models (MLLMs) for phonetic–semantic associations and layer-wise attention patterns. See the paper for full details and results.

## :mag_right: Overview
This artifact packages everything needed to verify the AAAI-26 paper results: the curated LEX-ICON dataset, multimodal prompts (text, IPA, audio), MCQ evaluation pipelines, and attention-alignment analysis scripts.  
- Self-contained reproduction bundle with frozen dependencies.  
- Supports both local checkpoints (Qwen/Qwen2.5-Omni-7B) and hosted APIs (GPT-4o-mini, Gemini 1.5 Pro).  
- Provides plotting utilities for semantic-dimension trends, modality advantages, and layer-wise attention.

## :bookmark_tabs: Paper
**Title:** _Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism_  

**Venue:** AAAI-26 (Oral)  

**Authors:** *Jinhong Jeong, *Sunghyun Lee, Jaeyoung Lee, Seonah Han, Youngjae Yu

## :tada: Interesting Results
- **RQ1 — Phonetic intuition holds across modalities.** Macro-F1 scores exceed the 0.50 baseline in 84.2% of semantic dimensions for natural mimetic words and 68.4% for constructed pseudo-words, showing that MLLMs infer meaning from both memorized and novel word forms rather than memorization alone.
- **Human-aligned trends.** Qwen2.5-Omni-7B achieves the strongest Pearson correlation (up to r = 0.579) with human judgements across semantic dimensions, while all models remain closest to human patterns on constructed words, supporting the reliability of the pseudo ground truth.
- **Modality-specific strengths.** Audio inputs particularly boost performance on acoustically grounded dimensions (e.g., big vs. small, fast vs. slow), whereas textual inputs (orthographic and IPA) excel on articulatory or visually driven contrasts such as sharp vs. round and beautiful vs. ugly; these trends yield a Pearson r = 0.681 between audio gains in natural and constructed sets.
- **RQ2 — Attention follows iconic phonemes.** Layer-wise analyses of Qwen2.5-Omni-7B show fraction attention scores above 0.50 for canonical sound-symbolic pairs (e.g., /p/ ↔ sharp, /m/ ↔ round), with late decoder layers emphasizing these associations more strongly for constructed words, evidencing internal alignment between phonemes and inferred meanings.

## :zap: Usage

###  :electric_plug: Installation
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

**Dataset refresh (optional; processed JSON + audio already included)**
```bash
bash src/dataset/constructed_word/run.sh
python src/dataset/prompt_generation/semantic_dimension/semantic_dimension.py --output data/prompts/semantic_dimension
```

**Core MCQ experiments (AAAI-26 Table 2)**
```bash
python src/experiments/semantic_dimension/mcq_experiments_all_at_once.py \
  --model "Qwen/Qwen2.5-Omni-7B" \
  --retry-failed-answers
```
   - Replace `--model` with `gpt-4o` or `gemini-2.5-flash` and add `--api` once corresponding API keys are configured.

**Attention analysis (AAAI-26 Figure 3)**
```bash
bash src/analysis/heatmap/run.sh                     # generates attention caches
python src/analysis/plotting/layer_attention_score_plot.py
```

**Human evaluation post-processing**
```bash
python src/experiments/semantic_dimension/human_eval/analyze_humaneval.py \
  --input data/human_eval --output results/experiments/semantic_dimension/human_eval
python src/analysis/statistics/semantic_dimension_result_statistics.py
```

**Reproduce final plots**
```bash
python src/analysis/plotting/modality_advantage_plot.py --results-dir results/experiments/semantic_dimension
python src/analysis/plotting/semantic_dimension_plot.py --stats-path results/statistics/semdim_stat.json
```

##  :wrench: Development

### :notebook: Pre-Requisites
- git (>=2.30)
- Python 3.9 / 3.10+
- CUDA-enabled GPU recommended for model inference (e.g., RTX 3090 / 4090)
- Linux / macOS recommended for reproduction scripts (audio toolchain compatibility)
- Optional: API keys for closed-source models (if using GPT/Gemini via API)

###  :nut_and_bolt: Development Environment
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

###  :file_folder: File Structure
```
sound-symbolism/
├── data/
│   ├── human_eval/
│   │   ├── label_studio_survey_0.json
│   │   ├── label_studio_survey_1.json
│   │   ├── label_studio_survey_2.json
│   │   ├── label_studio_survey_3.json
│   │   ├── label_studio_survey_4.json
│   │   ├── label_studio_survey_5.json
│   │   ├── label_studio_survey_6.json
│   │   ├── label_studio_survey_7.json
│   │   ├── label_studio_survey_8.json
│   │   └── label_studio_survey_9.json
│   ├── processed/
│   │   ├── art/
│   │   │   ├── alignment/
│   │   │   │   └── constructed_words.json
│   │   │   ├── constructed_words.json
│   │   │   ├── resources/
│   │   │   │   ├── ARPA_pronunciation_dict.txt
│   │   │   │   ├── feature_to_score.json
│   │   │   │   ├── ipa_to_alphabet.json
│   │   │   │   ├── ipa_to_feature.json
│   │   │   │   ├── ipa_to_word.json
│   │   │   │   └── phonemes.json
│   │   │   └── semantic_dimension/
│   │   │       └── semantic_dimension_binary_gt.json
│   │   └── nat/
│   │       ├── alignment/
│   │       │   ├── en.json
│   │       │   ├── fr.json
│   │       │   ├── ja.json
│   │       │   └── ko.json
│   │       ├── common_words.json
│   │       ├── en.json
│   │       ├── fr.json
│   │       ├── ja.json
│   │       ├── ko.json
│   │       ├── rare_words.json
│   │       ├── resources/
│   │       │   ├── en_MFA_pronunciation_dict.txt
│   │       │   ├── fr_MFA_pronunciation_dict.txt
│   │       │   ├── ipa_to_feature.json
│   │       │   ├── ja_MFA_pronunciation_dict.txt
│   │       │   └── ko_MFA_pronunciation_dict.txt
│   │       └── semantic_dimension/
│   │           └── semantic_dimension_binary_gt.json
│   └── prompts/
│       └── prompts.json
├── environment.yml
├── paths_to_LEX-ICON.txt
└── src/
    ├── analysis/
    │   ├── aligning/
    │   │   ├── ipa_map.py
    │   │   ├── mfa_main.py
    │   │   ├── mfa_preperation.py
    │   │   ├── mfa_wrapper.py
    │   │   └── parse_textgrid.py
    │   ├── heatmap/
    │   │   ├── batch_semdim_heatmap.py
    │   │   ├── combine_pkl_files.py
    │   │   ├── multi_threading_attention_computation.py
    │   │   ├── renewal_semdim_heatmap.py
    │   │   ├── requirements_attention.txt
    │   │   └── run.sh
    │   ├── plotting/
    │   │   ├── layer_attention_score_plot.py
    │   │   ├── modality_advantage_plot.py
    │   │   └── semantic_dimension_plot.py
    │   └── statistics/
    │       ├── semantic_dimension_distribution.py
    │       └── semantic_dimension_result_statistics.py
    ├── dataset/
    │   ├── audio/
    │   │   ├── tts_generation.py
    │   │   └── tts_generation_google.py
    │   ├── constructed_word/
    │   │   ├── annotate_score.py
    │   │   ├── generate_word.py
    │   │   ├── phoneme_data.py
    │   │   ├── process_ipa_word_dict.py
    │   │   └── run.sh
    │   ├── preprocessing/
    │   │   ├── process_ja_raw.py
    │   │   ├── process_ja_raw_2.py
    │   │   ├── process_ko_raw.py
    │   │   ├── raw_data_cleaning.py
    │   │   └── split_by_lexical_familiarity.py
    │   ├── prompt_generation/
    │   │   └── semantic_dimension/
    │   │       └── semantic_dimension.py
    │   └── semantic_dimension/
    │       ├── semantic_dimension_binary_gt.py
    │       └── semantic_dimension_llm_generation.py
    ├── experiments/
    │   └── semantic_dimension/
    │       ├── gemini_inference.py
    │       ├── gpt_inference.py
    │       ├── human_eval/
    │       │   ├── analyze_humaneval.py
    │       │   ├── analyze_humaneval_individual.py
    │       │   └── generate_survey.py
    │       ├── mcq_experiments_all_at_once.py
    │       └── qwen_omni_inference.py
    └── utils/
        ├── calibrate_results_for_gemini.py
        ├── convert_to_ipa.py
        ├── ipa_check.py
        ├── ipa_consistency.py
        ├── ipa_crawler.py
        └── translate_meanings.py
```

| Path | Purpose |
|------|---------|
| `data/processed` | Canonical LEX-ICON subsets (natural + constructed) used in AAAI-26 experiments. |
| `data/prompts` | Prompt templates and multiple-choice questions delivered to the models. |
| `src/dataset` | Scripts for regenerating phonetic resources, prompts, and TTS wav files. |
| `src/experiments` | Entry points for MCQ inference, human evaluation parsing, and API integrations. |
| `src/analysis` | Attention alignment, visualization, and statistics utilities. |
| `results` | Default output directory for metrics, figures, and serialized attention scores. |

## :bookmark: Citation
If this repository supports your research, please cite:
```
@inproceedings{jeong2026soundsymbolism,
  title     = {Do Language Models Associate Sound with Meaning? A Multimodal Study of Sound Symbolism},
  author    = {Jinhong Jeong, Sunghyun Lee, Jaeyoung Lee, Seonah Han, Youngjae Yu},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```
For any questions or issues, feel free to open an issue on this repository!