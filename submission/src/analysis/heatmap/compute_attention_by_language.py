# python src/analysis/heatmap/compute_attention_by_language.py

import json
import os
import re
import gc
from typing import Union, Dict, List, Tuple
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import argparse
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Qwen2_5OmniProcessor

model_path = "Qwen/Qwen2.5-Omni-7B"
data_type = "audio"
nat_langs = ["en", "fr", "ja", "ko"]
con_langs = ["art"]
lang = "art"

def get_constructed(language=lang):    
    if language in ["en", "fr", "ja", "ko"]:
        return False
    elif language in ["art", "con"]:
        global lang
        lang = "art"
        return True
constructed = get_constructed(lang)
layer_starts = [0, 9, 18]
layer_ends = [8, 17,27]
CHECK_MODEL_RESPONSE = True
COMPUTE_RULE = "fraction"
USE_SOFTMAX = True
sampling_rate = 20

def get_data_path(lang, data_type):
    if constructed or (lang == "art") or (lang == "con"):
        lang = "art"
        data_path = "data/processed/art/semantic_dimension/semantic_dimension_binary_gt.json"
        output_dir = f"results/experiments/understanding/attention_heatmap/con/semantic_dimension/{data_type}/{lang}/generation_attention"
    else:
        data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"
        output_dir = f"results/experiments/understanding/attention_heatmap/nat/semantic_dimension/{data_type}/{lang}/generation_attention"
    return data_path, output_dir

processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

semantic_dimension_map = [
    "good", "bad", "beautiful", "ugly", "pleasant", "unpleasant", "strong", "weak", "big", "small", 
    "rugged", "delicate", "active", "passive", "fast", "slow", "sharp", "round", "realistic", "fantastical", 
    "structured", "disorganized", "orginary", "unique", "interesting", "uninteresting", "simple", "complex", 
    "abrupt", "continuous", "exciting", "calming", "hard", "soft", "happy", "sad", "harsh", "mellow", 
    "heavy", "light", "inhibited", "free", "masculine", "feminine", "solid", "nonsolid", "tense", "relaxed", 
    "dangerous", "safe"
]
dim_pairs = [
    ("abrupt", "continuous"), ("continuous", "abrupt"), ("active", "passive"), ("passive", "active"),
    ("beautiful", "ugly"), ("ugly", "beautiful"), ("big", "small"), ("small", "big"),
    ("dangerous", "safe"), ("safe", "dangerous"), ("exciting", "calming"), ("calming", "exciting"),
    ("fast", "slow"), ("slow", "fast"), ("good", "bad"), ("bad", "good"),
    ("happy", "sad"), ("sad", "happy"), ("hard", "soft"), ("soft", "hard"),
    ("harsh", "mellow"), ("mellow", "harsh"), ("heavy", "light"), ("light", "heavy"),
    ("inhibited", "free"), ("free", "inhibited"), ("interesting", "uninteresting"), ("uninteresting", "interesting"),
    ("masculine", "feminine"), ("feminine", "masculine"), ("orginary", "unique"), ("unique", "orginary"),
    ("pleasant", "unpleasant"), ("unpleasant", "pleasant"), ("realistic", "fantastical"), ("fantastical", "realistic"),
    ("rugged", "delicate"), ("delicate", "rugged"), ("sharp", "round"), ("round", "sharp"),
    ("simple", "complex"), ("complex", "simple"), ("solid", "nonsolid"), ("nonsolid", "solid"),
    ("strong", "weak"), ("weak", "strong"), ("structured", "disorganized"), ("disorganized", "structured"),
    ("tense", "relaxed"), ("relaxed", "tense"),
]
ipa_symbols = [
    'a', 'ɑ', 'æ', 'ɐ', 'ə', 'ɚ', 'ɝ', 'ɛ', 'ɜ', 'e', 'ɪ', 'i', 'ɨ', 'ɯ', 'o', 'ɔ', 'ʊ', 'u', 'ʌ', 'ʉ',
    'b', 'β', 'c', 'ç', 'd', 'ð', 'f', 'ɡ', 'ɣ', 'h', 'ɦ', 'j', 'k', 'l', 'ɭ', 'ʟ', 'm', 'ɱ', 'n', 'ŋ',
    'ɲ', 'p', 'ɸ', 'q', 'r', 'ɾ', 'ɹ', 'ʁ', 's', 'ʃ', 't', 'θ', 'v', 'w', 'x', 'χ', 'z', 'ʒ', 'ʔ', 'ʕ',
    'ʡ', 'ʢ', 'ʘ', 'ǀ', 'ǃ', 'ǂ', 'ǁ', 'ɓ', 'ɗ', 'ʄ', 'ɠ', 'ʛ', 'ɦ', 'ʍ', 'ɥ', 'ʜ', 'ʎ', 
    'ɺ', 'ɻ', 'ɽ', 'ʀ', 'ʂ', 'ʈ', 'ʋ', 'ʐ', 'ʑ', 'ʝ', 'ʞ', 'ʟ', 'ʠ', 'ʡ', 'ʢ', 'ʣ', 'ʤ', 'ʥ', 'ʦ',
    'ʧ', 'ʨ', 'ʩ', 'ʪ', 'ʫ', 'ʬ', 'ʭ', 'ʮ', 'ʯ',
    'ɴ', 'ɕ', 'd͡ʑ', 't͡ɕ', 'ʑ', 'ɰ', 'ã', 'õ', 'ɯ̃', 'ĩ', 'ẽ', 'ɯː', 'aː', 'oː', 'iː', 'eː'
]
consonants = [
    'b','d','ɟ','ɡ','ɖ','ɢ',    # voiced_stops
    'p','t','c','k','ʈ','q',    # voiceless_stops
    'v','ð','z','ʒ','ʐ','ʝ','ɣ','ʁ','ʕ','ɦ','β',    # voiced_fricatives
    'f','s','ʃ','θ','ç','x','χ','ħ','h','ɸ','ʂ','ɬ',    # voiceless_fricatives
    'm','ɱ','n','ɳ','ɲ','ŋ','ɴ','l','ɭ','ʎ','ʟ','ʀ','r','ɾ','ɽ','ɹ','ɺ','ɻ','j','ɰ',    # sonorants
    'ʔ','w','ʡ','ʢ'   # others
]
vowels = [
    'i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ʘ', 'ɛ', 'œ', 'æ', 'a', 'ɶ',  # Front vowels
    'ɨ', 'ʉ', 'ɯ', 'u', 'ʊ', 'ɤ', 'o', 'ɜ', 'ɝ', 'ɞ', 'ʌ', 'ɔ', 'ɑ', 'ɒ'  # Central/Back vowels
]
# consonants = [
#     'p', 'b', 'ɸ', 'β', 'm', 'ɱ', # Bilabial
#     'f', 'v', # Labiodental
#     'θ', 'ð', # Dental
#     't', 'd', 's', 'z', 'n', 'r', 'ɾ', 'ɹ', 'l', 'ɬ', 'ɮ', # Alveolar
#     'ʃ', 'ʒ', 'ɻ', # Post-alveolar
#     'ʈ', 'ɖ', 'ʂ', 'ʐ', 'ɳ', 'ɽ', 'ɭ', # Retroflex
#     'c', 'ɟ', 'ç', 'ʝ', 'ɲ', 'j', 'ʎ', # Palatal
#     'k', 'ɡ', 'x', 'ɣ', 'ŋ', 'ɰ', 'ʟ', # Velar
#     'q', 'ɢ', 'χ', 'ʁ', 'ɴ', # Uvular
#     'ħ', 'ʕ', # Pharyngeal
#     'h', 'ɦ', 'ʔ' # Glottal
# ]

def clean_token(token:str) -> str:
    return re.sub(r'^[ĠĊ\[\],.:;!?\n\r\t]+|[ĠĊ\[\],.:;!?\n\r\t]+$', '', token)

def get_word_list(lang:str=lang, data_path:str=None) -> list[str]:
    if lang not in ["art", "en", "fr", "ja", "ko"]:
        raise ValueError(f"Language {lang} not supported")
    
    with open(data_path, "r") as f:
        data = json.load(f)[lang]
    word_list = []
    for sample in data:
        word = sample["word"]
        word_list.append(word)
    print(f"Number of words: {len(word_list)}")
    return word_list

def show_arguments(model_name:str=model_path, data_type:str=data_type, lang:str=lang, layer_start:int=0, layer_end:int=27, constructed:bool=constructed, check_model_response:bool=CHECK_MODEL_RESPONSE, compute_rule:str=COMPUTE_RULE, sampling_rate:int=None):
    print(f"Model: {model_name}")
    print(f"Data type: {data_type}")
    print(f"Language: {lang}")
    print(f"Layer start: {layer_start}")
    print(f"Layer end: {layer_end}")
    print(f"Constructed: {constructed}")
    print(f"Check model response: {check_model_response}")
    print(f"Compute rule: {compute_rule}")
    print(f"Sampling rate: {sampling_rate}")

def model_guessed_incorrectly(response, dim1, dim2, answer) -> bool:
    if dim1 == answer and response == "1":
        # print(f"Model guessed incorrectly.")
        return False
    elif dim2 == answer and response == "2":
        # print(f"Model guessed incorrectly.")
        return False
    # print(f"Model guessed correctly.")
    return True

def find_basic_info(word, output_dir=None, dim_pairs=dim_pairs, word_stats=None) -> tuple[list[str], dict]:
    ipa_list = []
    for dim1, dim2 in dim_pairs:
        data_dir = os.path.join(output_dir, f"{dim1}_{dim2}", f"{word}_{dim1}_{dim2}.pkl")
        alt_dir = os.path.join(output_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl")
        if not os.path.exists(data_dir) or not os.path.exists(alt_dir):
            continue
        else:
            # print(f"Word: {word}, Dim1: {dim1}, Dim2: {dim2}")
            alt = pkl.load(open(alt_dir, "rb"))
            ipa_tokens = alt["ipa_tokens"]
            converted_ipa_tokens = convert_ipa_tokens_to_ipa_string_per_token(ipa_tokens)
            ipa_list = [clean_token(token) for token in converted_ipa_tokens]
            word_stats = {ipa:{} for ipa in ipa_list}
            word_stats = {k: v for k, v in word_stats.items() if k != ""} # Remove empty text
            break
    return ipa_list, word_stats

def get_semdim_matrix_and_labels(word_stats, dim_pairs, ipa_list):
    """
    Return (semdim_list, matrix) for Y축: dim_pairs 순서
    """
    import numpy as np
    seen_pairs = set()
    semdim_list = []
    matrix_rows = []
    for dim1, dim2 in dim_pairs:
        if (dim2, dim1) in seen_pairs:
            continue
        vals_dim1 = []
        vals_dim2 = []
        for ipa in ipa_list:
            v1 = word_stats[ipa].get(dim1, np.nan)
            v2 = word_stats[ipa].get(dim2, np.nan)
            v1_rev = word_stats[ipa].get(dim2, np.nan)
            v2_rev = word_stats[ipa].get(dim1, np.nan)
            vals_dim1.append(np.nanmean([v1, v2_rev]))
            vals_dim2.append(np.nanmean([v2, v1_rev]))
        vals_dim1 = np.array(vals_dim1)
        vals_dim2 = np.array(vals_dim2)
        if not np.all(np.isnan(vals_dim1)):
            semdim_list.append(f"{dim1}")
            matrix_rows.append(vals_dim1)
        if not np.all(np.isnan(vals_dim2)):
            semdim_list.append(f"{dim2}")
            matrix_rows.append(vals_dim2)
        seen_pairs.add((dim1, dim2))
        seen_pairs.add((dim2, dim1))
    matrix = np.vstack(matrix_rows) if matrix_rows else np.zeros((0, len(ipa_list)))
    return semdim_list, matrix

def get_ordered_ipa_list(word_stats, vowels, consonants):
    """
    Return ipa_list for X축: vowels → consonants → 기타
    """
    ipa_set = set(word_stats.keys())
    ordered_ipa = []
    # 1. Vowels
    for v in vowels:
        if v in ipa_set:
            ordered_ipa.append(v)
    # 2. Consonants
    for c in consonants:
        if c in ipa_set:
            ordered_ipa.append(c)
    # 3. Others
    for ipa in word_stats.keys():
        if ipa not in ordered_ipa:
            ordered_ipa.append(ipa)
    return ordered_ipa

def plot_sampled_word_heatmap(word_stats, data_type, start_layer, end_layer, lang, save_path=None, suffix:str=None, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, dim_pairs=dim_pairs, answer_list=None, sampling_rate=1):
    import numpy as np
    import os
    if save_path is None:
        save_path = 'results/plots/attention/sampled_words/'
    os.makedirs(save_path, exist_ok=True)
    ipa_list = get_ordered_ipa_list(word_stats, vowels, consonants)
    semdim_list, matrix = get_semdim_matrix_and_labels(word_stats, dim_pairs, ipa_list)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        print(f"[WARN] Empty matrix for lang '{lang}'. Skipping heatmap.")
        return
    title = f'Lang: {lang} | rule: {compute_rule} | L{start_layer}-L{end_layer} | check_model_response: {check_model_response} | sampling_rate: {sampling_rate}\nIPA-Semantic Dimension Attention Heatmap'
    file_name = f"{lang.upper()}_{data_type}_generation_attention_L{start_layer}_L{end_layer}_sampling_{sampling_rate}"
    if compute_rule is not None:
        file_name += f"_rule-{compute_rule}"
    if check_model_response is not None:
        file_name += f"_check-{check_model_response}"
    if suffix:
        file_name += suffix
    file_name += ".png"
    file_path = os.path.join(save_path, file_name)
    draw_plot(ipa_list, semdim_list, answer_list, matrix, title, file_path, dim_pairs)

def plot_ranked_heatmap(word_stats, data_type, start_layer, end_layer, lang, save_path=None, suffix:str=None, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, dim_pairs=dim_pairs, answer_list=None, sampling_rate=1):
    import numpy as np
    import os
    if save_path is None:
        save_path = 'results/plots/attention/sampled_words/'
    os.makedirs(save_path, exist_ok=True)
    ipa_list = get_ordered_ipa_list(word_stats, vowels, consonants)
    semdim_list, matrix = rank_matrix_by_semdim(word_stats, dim_pairs, ipa_list)
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        print(f"[WARN] Empty matrix for lang '{lang}'. Skipping heatmap.")
        return
    title = f'Lang: {lang} | rule: {compute_rule} | L{start_layer}-L{end_layer} | check_model_response: {check_model_response} | sampling_rate: {sampling_rate}\nIPA-Semantic Dimension Attention Heatmap Ranked'
    file_name = f"{lang.upper()}_{data_type}_generation_attention_L{start_layer}_L{end_layer}_sampling_{sampling_rate}_ranked"
    if compute_rule is not None:
        file_name += f"_rule-{compute_rule}"
    if check_model_response is not None:
        file_name += f"_check-{check_model_response}"
    if suffix:
        file_name += suffix
    file_name += ".png"
    file_path = os.path.join(save_path, file_name)
    draw_plot(ipa_list, semdim_list, answer_list, matrix, title, file_path, dim_pairs, ranked=True)

def rank_matrix_by_semdim(word_stats, dim_pairs, ipa_list):
    """
    Convert attention scores to ranks for each semantic dimension.
    For each semantic dimension, rank IPA symbols by their attention scores.
    Higher scores get lower rank numbers (1st, 2nd, etc.).
    Missing values become nan.
    Returns: (semdim_list, matrix) where matrix[i,j] = rank of IPA j for semantic dimension i
    """
    import numpy as np
    from scipy.stats import rankdata
    
    # Get semantic dimension list in the same way as get_semdim_matrix_and_labels
    seen_pairs = set()
    semdim_list = []
    matrix_rows = []
    for dim1, dim2 in dim_pairs:
        if (dim2, dim1) in seen_pairs:
            continue
        vals_dim1 = []
        vals_dim2 = []
        for ipa in ipa_list:
            v1 = word_stats[ipa].get(dim1, np.nan)
            v2 = word_stats[ipa].get(dim2, np.nan)
            v1_rev = word_stats[ipa].get(dim2, np.nan)
            v2_rev = word_stats[ipa].get(dim1, np.nan)
            vals_dim1.append(np.nanmean([v1, v2_rev]))
            vals_dim2.append(np.nanmean([v2, v1_rev]))
        vals_dim1 = np.array(vals_dim1)
        vals_dim2 = np.array(vals_dim2)
        if not np.all(np.isnan(vals_dim1)):
            semdim_list.append(f"{dim1}")
            matrix_rows.append(vals_dim1)
        if not np.all(np.isnan(vals_dim2)):
            semdim_list.append(f"{dim2}")
            matrix_rows.append(vals_dim2)
        seen_pairs.add((dim1, dim2))
        seen_pairs.add((dim2, dim1))
    
    if not matrix_rows:
        return [], np.zeros((0, len(ipa_list)))
    
    # Convert scores to ranks for each semantic dimension
    matrix = np.zeros((len(semdim_list), len(ipa_list)))
    matrix[:] = np.nan  # Initialize with nan
    
    for i, semdim in enumerate(semdim_list):
        # Get scores for this semantic dimension across all IPA symbols
        scores = []
        valid_indices = []
        for j, ipa in enumerate(ipa_list):
            score = word_stats[ipa].get(semdim, np.nan)
            if not np.isnan(score):
                scores.append(score)
                valid_indices.append(j)
            # else: keep as nan
        
        if len(scores) > 0:
            # Convert scores to ranks (higher score = lower rank number)
            # Use 'ordinal' method to handle ties by giving them the same rank
            ranks = rankdata(scores, method='ordinal')
            # Reverse ranks so higher scores get lower rank numbers
            ranks = len(scores) - ranks + 1
            
            # Put ranks back in matrix
            for rank, idx in zip(ranks, valid_indices):
                matrix[i, idx] = rank
    
    return semdim_list, matrix

def plot_by_stats_with_ipa_wise(word_stats, data_type, start_layer, end_layer, lang, save_path=None, suffix:str=None, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, dim_pairs=dim_pairs, answer_list=None, use_softmax=False, sampling_rate=1):
    import numpy as np
    import os
    if save_path is None:
        save_path = 'results/plots/attention/sampled_words/'
    os.makedirs(save_path, exist_ok=True)
    ipa_list = get_ordered_ipa_list(word_stats, vowels, consonants)
    semdim_list, matrix = get_semdim_matrix_and_labels(word_stats, dim_pairs, ipa_list)
    if use_softmax:
        scaled_matrix = scale_matrix_by_ipa(matrix, softmax=True)
    else:
        scaled_matrix = matrix
    if scaled_matrix.shape[0] == 0 or scaled_matrix.shape[1] == 0:
        print(f"[WARN] Empty matrix for lang '{lang}'. Skipping heatmap.")
        return
    title = f'Lang: {lang} | Data type: {data_type} | rule: {compute_rule} | L{start_layer}-L{end_layer} | check_model_response: {check_model_response} | sampling_rate: {sampling_rate}\nIPA-Semantic Dimension Attention IPA-wise Scaled Heatmap (softmax: {use_softmax})'
    file_name = f"{lang.upper()}_{data_type}_attention_L{start_layer}_L{end_layer}_ipa_wise_scaled_softmax_{use_softmax}_sampling_{sampling_rate}"
    if compute_rule is not None:
        file_name += f"_rule-{compute_rule}"
    if check_model_response is not None:
        file_name += f"_check-{check_model_response}"
    if suffix:
        file_name += suffix
    file_name += ".png"
    file_path = os.path.join(save_path, file_name)
    draw_plot(ipa_list, semdim_list, answer_list, scaled_matrix, title, file_path, dim_pairs)

def draw_plot(ipa_list, semdim_list, answer_list, matrix, title, file_path, dim_pairs=None, ranked=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    fig, ax = plt.subplots(figsize=(max(10, len(ipa_list)*0.5), max(8, len(semdim_list)*0.3)))
    mask = np.isnan(matrix)
    if ranked:
        fmt = '.0f'
    else:
        fmt = '.3f'
    sns.heatmap(
        matrix, ax=ax,
        cmap='vlag', # 'vlag', 'coolwarm', 'RdYlGn_r'
        cbar=True,
        xticklabels=ipa_list, yticklabels=semdim_list,
        linewidths=0.2, linecolor='gray', square=False,
        annot=True, fmt=fmt, mask=mask, annot_kws={"size":8}
    )
    
    if dim_pairs is not None:
        drawn = set()
        for dim1, dim2 in dim_pairs:
            try:
                idx1 = semdim_list.index(dim1)
            except ValueError:
                idx1 = None
            try:
                idx2 = semdim_list.index(dim2)
            except ValueError:
                idx2 = None
            # Avoid drawing the same line twice
            if idx1 is not None and idx2 is not None:
                top = min(idx1, idx2)
                bottom = max(idx1, idx2)
                if top not in drawn:
                    ax.axhline(top, color='black', linewidth=2)
                    drawn.add(top)
                if (bottom+1) not in drawn:
                    ax.axhline(bottom+1, color='black', linewidth=2)
                    drawn.add(bottom+1)
            elif idx1 is not None:
                if idx1 not in drawn:
                    ax.axhline(idx1, color='black', linewidth=2)
                    drawn.add(idx1)
                if (idx1+1) not in drawn:
                    ax.axhline(idx1+1, color='black', linewidth=2)
                    drawn.add(idx1+1)
            elif idx2 is not None:
                if idx2 not in drawn:
                    ax.axhline(idx2, color='black', linewidth=2)
                    drawn.add(idx2)
                if (idx2+1) not in drawn:
                    ax.axhline(idx2+1, color='black', linewidth=2)
                    drawn.add(idx2+1)

    ax.set_xlabel('IPA Symbol', fontsize=12)
    ax.set_ylabel('Semantic Dimension', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    plt.setp(ax.get_xticklabels(), ha='right')
    plt.tight_layout()
    if answer_list is not None:
        yticklabels = ax.get_yticklabels()
        for i, label in enumerate(yticklabels):
            if label.get_text() in answer_list:
                label.set_fontweight('bold')
        ax.set_yticklabels(yticklabels)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Sampled word heatmap saved to {file_path}")
    plt.close()

def scale_matrix_by_ipa(matrix, softmax=False):
    scaled = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col_no_nan = np.nan_to_num(col, nan=0.0)
        s = np.sum(col_no_nan)
        if softmax:
            if s == 0:
                scaled[:, j] = col_no_nan
            else:
                scaled[:, j] = np.exp(col_no_nan) / np.sum(np.exp(col_no_nan))
        else:
            if s == 0:
                scaled[:, j] = col_no_nan
            else:
                scaled[:, j] = col_no_nan / s
    return scaled

def get_ipa_runs(ipa_list) -> list[tuple[str, int, int]]:
    """
    Given a list of IPA symbols, return a list of (ipa, start_idx, end_idx) for each run of the same IPA symbol.
    Example: [l, l, l, l, i] -> [('l', 0, 3), ('i', 4, 4)]
    """
    ipa_runs = []
    if not ipa_list:
        return ipa_runs
    prev_ipa = ipa_list[0]
    start_idx = 0
    for idx in range(1, len(ipa_list)):
        if ipa_list[idx] != prev_ipa:
            ipa_runs.append((prev_ipa, start_idx, idx-1))
            prev_ipa = ipa_list[idx]
            start_idx = idx
    ipa_runs.append((prev_ipa, start_idx, len(ipa_list)-1))
    return ipa_runs

def get_data(data:dict, alt:dict):
    attention_matrices = data["attention_matrices"]
    relevant_indices = data["relevant_indices"]
    dim1, dim2 = data["dimension1"], data["dimension2"]
    answer, word_tokens, option_tokens, tokens = data["answer"], data["word_tokens"], data["option_tokens"], data["tokens"]
    ipa_tokens = alt["ipa_tokens"]
    response, input_word, target_indices = alt["response"], alt["input_word"], alt["target_indices"]
    wlen, d1len, d2len = len(target_indices["word"]), len(target_indices["dim1"]), len(target_indices["dim2"])
    dim1_range = range(wlen, wlen+d1len)
    dim2_range = range(wlen+d1len, wlen+d1len+d2len)
    return attention_matrices, relevant_indices, dim1, dim2, answer, word_tokens, option_tokens, tokens, ipa_tokens, response, input_word, target_indices, wlen, d1len, d2len, dim1_range, dim2_range

def add_to_word_stats(word_stats, ipa, dim, values, ipa_symbols=ipa_symbols):
    if ipa in ipa_symbols and ipa not in word_stats.keys():
        word_stats[ipa] = {}
    if dim not in word_stats[ipa].keys():
        word_stats[ipa][dim] = []
    word_stats[ipa][dim].extend(values)
    return word_stats

def compute_ipa_semdim_score_with_naive_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats) -> dict:
    for ipa_idx, ipa in enumerate(ipa_list):
        if ipa == "":
            continue
        for dim, dim_range in [(dim1, dim1_range), (dim2, dim2_range)]:
            all_values = []
            for layer in range(start_layer, min(end_layer+1, n_layer)):
                for head in range(n_head):
                    layer_len = attn_layers[layer].shape[2]
                    if layer_len != wlen+d1len+d2len:
                        print(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                        break
                        raise ValueError(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                    for d_idx in dim_range:
                        if d_idx < attn_layers[layer].shape[2] and ipa_idx < attn_layers[layer].shape[3]:
                            v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                            all_values.append(v)
            word_stats = add_to_word_stats(word_stats, ipa, dim, all_values)
    return word_stats

def compute_semdim_score_with_naive_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats) -> dict:
    ipa_runs = get_ipa_runs(ipa_list)
    for ipa, start_idx, end_idx in ipa_runs:
        if ipa == "" or ipa == " ":
            continue
        for dim, dim_range in [(dim1, dim1_range), (dim2, dim2_range)]:
            all_values = []
            for layer in range(start_layer, min(end_layer+1, n_layer)):
                for head in range(n_head):
                    layer_len = attn_layers[layer].shape[2]
                    if layer_len != wlen+d1len+d2len:
                        break
                        raise ValueError(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                    for d_idx in dim_range:
                        sum_score = 0.0
                        for ipa_idx in range(start_idx, end_idx+1):
                            if d_idx < attn_layers[layer].shape[2] and ipa_idx < attn_layers[layer].shape[3]:
                                v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                                sum_score += v
                        all_values.append(sum_score)
            word_stats = add_to_word_stats(word_stats, ipa, dim, all_values)
    return word_stats

def compute_semdim_score_with_answer_only_rule(ipa_list, dim1, dim2, answer, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats) -> dict:
    ipa_runs = get_ipa_runs(ipa_list)
    if dim1 == answer:
        check_dim = dim1
        check_dim_range = dim1_range
        ignore_dim_range = dim2_range
    elif dim2 == answer:
        check_dim = dim2
        check_dim_range = dim2_range
        ignore_dim_range = dim1_range
    else:
        raise ValueError(f"Answer {answer} is not in dim1 or dim2")
    for ipa_idx, ipa in enumerate(ipa_list):
        if ipa == "":
            continue
        layer = start_layer
        attn = attn_layers[layer]
        final_check_values = []
        check_values = []
        ignore_values = []
        for layer in range(start_layer, min(end_layer+1, n_layer)):
            for head in range(n_head):
                layer_len = attn_layers[layer].shape[2]
                if layer_len != wlen+d1len+d2len:
                    print(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                    break
                for d_idx in check_dim_range:
                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                        v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                        check_values.append(v)
                check_sum = sum(check_values)
                
                for d_idx in ignore_dim_range:
                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                        v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                        ignore_values.append(v)
                ignore_sum = sum(ignore_values)

                denom = check_sum + ignore_sum
                if denom == 0:
                    frac_check = 0.0
                else:
                    frac_check = check_sum / denom
                final_check_values.append(frac_check)
        word_stats = add_to_word_stats(word_stats, ipa, check_dim, final_check_values)
    return word_stats

def compute_ipa_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats) -> dict:
    for ipa_idx, ipa in enumerate(ipa_list):
        if ipa == "":
            continue
        layer = start_layer
        attn = attn_layers[layer]
        final_dim1_values = []
        final_dim2_values = []
        dim1_values = []
        dim2_values = []
        for layer in range(start_layer, min(end_layer+1, n_layer)):
            for head in range(n_head):
                layer_len = attn_layers[layer].shape[2]
                if layer_len != wlen+d1len+d2len:
                    print(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                    break
                for d_idx in dim1_range:
                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                        v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                        dim1_values.append(v)
                dim1_sum = sum(dim1_values)
                
                for d_idx in dim2_range:
                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                        v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                        dim2_values.append(v)
                dim2_sum = sum(dim2_values)

                denom = dim1_sum + dim2_sum
                if denom == 0:
                    frac_dim1, frac_dim2 = 0.0, 0.0
                else:
                    frac_dim1 = dim1_sum / denom
                    frac_dim2 = dim2_sum / denom
                # print(f"Frac dim1: {frac_dim1}, Frac dim2: {frac_dim2}")
                final_dim1_values.append(frac_dim1)
                final_dim2_values.append(frac_dim2)
        word_stats = add_to_word_stats(word_stats, ipa, dim1, final_dim1_values)
        word_stats = add_to_word_stats(word_stats, ipa, dim2, final_dim2_values)
    return word_stats

def compute_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats) -> dict:
    ipa_runs = get_ipa_runs(ipa_list)
    for ipa, start_idx, end_idx in ipa_runs:
        if ipa == "":
            continue
        final_dim1_values = []
        final_dim2_values = []
        dim1_values = []
        dim2_values = []
        layer = start_layer
        attn = attn_layers[layer]
        for layer in range(start_layer, min(end_layer+1, n_layer)):
            for head in range(n_head):
                layer_len = attn_layers[layer].shape[2]
                if layer_len != wlen+d1len+d2len:
                    print(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                    break
                sum_dim1 = 0.0
                for d_idx in dim1_range:
                    for ipa_idx in range(start_idx, end_idx+1):
                        if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                            sum_dim1 += attn_layers[layer][0, head, d_idx, ipa_idx].item()
                dim1_values.append(sum_dim1)
                
                sum_dim2 = 0.0
                for d_idx in dim2_range:
                    for ipa_idx in range(start_idx, end_idx+1):
                        if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                            sum_dim2 += attn_layers[layer][0, head, d_idx, ipa_idx].item()
                dim2_values.append(sum_dim2)
                
                dim1_sum = sum(dim1_values)
                dim2_sum = sum(dim2_values)
                denom = dim1_sum + dim2_sum
                if denom == 0:
                    frac_dim1, frac_dim2 = 0.0, 0.0
                else:
                    frac_dim1 = dim1_sum / denom
                    frac_dim2 = dim2_sum / denom
                # print(f"Frac dim1: {frac_dim1}, Frac dim2: {frac_dim2}")
                final_dim1_values.append(frac_dim1)
                final_dim2_values.append(frac_dim2)
        word_stats = add_to_word_stats(word_stats, ipa, dim1, final_dim1_values)
        word_stats = add_to_word_stats(word_stats, ipa, dim2, final_dim2_values)
    return word_stats

def convert_ipa_tokens_to_ipa_string_per_token(ipa_tokens, processor=processor, ipa_symbols=ipa_symbols) -> list[str]:
    converted_ipa_tokens = []
    token_to_symbol_map = []
    i = 0
    symbol_for_token = [None] * len(ipa_tokens)
    while i < len(ipa_tokens):
        found = False
        for j in range(min(len(ipa_tokens), i+4), i, -1):
            tmp_token = ipa_tokens[i:j]
            tmp_converted_str = processor.tokenizer.convert_tokens_to_string(tmp_token).strip()
            # Only accept if it's a single valid IPA symbol and not ''
            if tmp_converted_str in ipa_symbols and tmp_converted_str != '':
                for k in range(i, j):
                    symbol_for_token[k] = tmp_converted_str
                i = j
                found = True
                break
        if not found:
            tmp_converted_str = processor.tokenizer.convert_tokens_to_string([ipa_tokens[i]]).strip()
            # Only accept if it's a single valid IPA symbol and not ''
            if tmp_converted_str in ipa_symbols and tmp_converted_str != '':
                symbol_for_token[i] = tmp_converted_str
            else:
                symbol_for_token[i] = ''
            i += 1
    return symbol_for_token

def first_load_data(word:str, load_dir:str):
    file_path = os.path.join(load_dir, f"{word}.pkl")
    if not os.path.exists(file_path):
        return None
    else:
        data = pkl.load(open(file_path, "rb"))
        return data

def return_data_from_word_data(word_data, word, dim1, dim2):
    data_key = os.path.join(f"{dim1}_{dim2}/{word}_{dim1}_{dim2}.pkl")
    alt_key = os.path.join(f"{word}_{dim1}_{dim2}_generation_analysis.pkl")
    if data_key not in word_data or alt_key not in word_data:
        return None, None
    else:
        data = word_data[data_key]
        alt = word_data[alt_key]
        return data, alt

def compute_single_word_attention_score(
        word:str,
        data_type:str=data_type,
        lang:str=lang,
        start_layer:int=0,
        end_layer:int=27,
        dim_pairs:list=dim_pairs,
        data_path:str=None,
        output_dir:str=None,
        word_stats:dict=None,
        check_model_response:bool=CHECK_MODEL_RESPONSE,
        compute_rule:str=COMPUTE_RULE,
        model_path:str=model_path,
    ) -> dict:
    ipa_list, word_stats = find_basic_info(word=word, output_dir=output_dir, dim_pairs=dim_pairs, word_stats=word_stats)
    if not ipa_list:
        return word_stats
    
    load_dir = f"results/experiments/understanding/attention_heatmap/combined/{data_type}/{lang}"
    # word_data = first_load_data(word=word, load_dir=load_dir) # TEST
    word_data = None
    incorrect_count = 0
    
    for dim1, dim2 in dim_pairs:
        data, alt = None, None
        # if word_data is not None: # TEST
        #     data, alt = return_data_from_word_data(word_data, word, dim1, dim2) # TEST

        if data is None or alt is None or word_data is None:
            data_dir = os.path.join(output_dir, f"{dim1}_{dim2}", f"{word}_{dim1}_{dim2}.pkl")
            alt_dir = os.path.join(output_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl")
            if not os.path.exists(data_dir) or not os.path.exists(alt_dir):
                continue
            # print(f"Found data for {word} {dim1}-{dim2}")
            data = pkl.load(open(data_dir, "rb"))
            alt = pkl.load(open(alt_dir, "rb"))
        attention_matrices, relevant_indices, dim1, dim2, answer, word_tokens, option_tokens, tokens, ipa_tokens, response, input_word, target_indices, wlen, d1len, d2len, dim1_range, dim2_range = get_data(data, alt)
        if check_model_response and model_guessed_incorrectly(response, dim1, dim2, answer):
            # print(f"Model guessed incorrectly for {word} {dim1}-{dim2}")
            incorrect_count += 1
            continue
            
        cleaned_ipa_tokens = [clean_token(token) for token in ipa_tokens]
        converted_ipa_tokens = convert_ipa_tokens_to_ipa_string_per_token(ipa_tokens)
        ipa_list = converted_ipa_tokens
        attn_layers = attention_matrices[0]
        n_layer = len(attn_layers)
        n_head = attn_layers[0].shape[1]
        if data_type == "ipa":
            if compute_rule == "naive":
                word_stats = compute_semdim_score_with_naive_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
            elif compute_rule == "fraction":
                word_stats = compute_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
            elif compute_rule == "answer_only":
                word_stats = compute_semdim_score_with_answer_only_rule(ipa_list, dim1, dim2, answer, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
        elif data_type == "audio":
            if compute_rule == "naive":
                word_stats = compute_semdim_score_with_naive_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
            elif compute_rule == "fraction":
                word_stats = compute_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
            elif compute_rule == "answer_only":
                word_stats = compute_semdim_score_with_answer_only_rule(ipa_list, dim1, dim2, answer, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        gc.collect()
        torch.cuda.empty_cache()
        # print(word_stats)
    # word_data.clear()
    del word_data
    gc.collect()
    torch.cuda.empty_cache()
    if len(dim_pairs) == incorrect_count:
        print(f"Model guessed incorrectly for all {len(dim_pairs)} dimensions for {word}")
    return word_stats

final_word_stats = {}

def compute_attention_by_language(
    lang: str = lang,
    data_type: str = data_type,
    data_path: str = None,
    output_dir: str = None,
    dim_pairs: list = dim_pairs,
    layer_start: int = 0,
    layer_end: int = 27,
    compute_rule: str = COMPUTE_RULE,
    check_model_response: bool = CHECK_MODEL_RESPONSE,
    final_word_stats:dict=final_word_stats,
    sampling_rate:int=1,
    single_language:bool=True,
    fraction_when_answer_only:bool=False,
):
    semdim_set = set()
    for dim1, dim2 in dim_pairs:
        semdim_set.add(dim1)
        semdim_set.add(dim2)
    semdim_list = sorted(list(semdim_set))
    word_list = get_word_list(lang=lang, data_path=data_path)
    word_count = 0
    processed_words = 0
    for word in tqdm(word_list):
        try:
            word_count += 1
            if word_count % sampling_rate != 0:
                continue
            word_stats = compute_single_word_attention_score(
                word=word, data_type=data_type, lang=lang, start_layer=layer_start, end_layer=layer_end,
                dim_pairs=dim_pairs, data_path=data_path, output_dir=output_dir,
                check_model_response=check_model_response, compute_rule=compute_rule, model_path=model_path
            )
            if word_stats is None:
                continue
            for ipa, dim_scores in word_stats.items():
                if not ipa or not dim_scores:
                    continue
                for dim, score in dim_scores.items():
                    if ipa not in final_word_stats.keys():
                        final_word_stats[ipa] = {}
                    if dim not in final_word_stats[ipa].keys():
                        final_word_stats[ipa][dim] = []
                    final_word_stats[ipa][dim].extend(score)
            processed_words += 1
            # if word_count > 100:
            #     break
        except Exception as e:
            print(f"Error processing word {word}: {e}")
            continue
    print(f"Processed {processed_words} words")
    # if single_language and not fraction_when_answer_only:
    final_word_stats = compute_mean_score(final_word_stats)
    # elif single_language and fraction_when_answer_only and COMPUTE_RULE == "answer_only":
    #     final_word_stats = compute_fraction_mean_of_answer_only(final_word_stats)
    return final_word_stats, processed_words

def compute_fraction_mean_of_answer_only(final_word_stats, dim_pairs=dim_pairs, ipa_symbols=ipa_symbols):
    for ipa, dim_scores in final_word_stats.items():
        dim_scores_key_list = list(dim_scores.keys())
        for dim1, dim2 in dim_pairs:
            if dim1 not in dim_scores_key_list or dim2 not in dim_scores_key_list:
                continue
            dim1_stats = dim_scores[dim1]
            dim2_stats = dim_scores[dim2]
            if len(dim1_stats) == 0 or len(dim2_stats) == 0:
                continue
            dim1_mean = sum(dim1_stats) / len(dim1_stats)
            dim2_mean = sum(dim2_stats) / len(dim2_stats)
            denom = dim1_mean + dim2_mean
            if denom == 0:
                dim1_mean = 0.0
                dim2_mean = 0.0
            else:
                dim1_frac = dim1_mean / denom
                dim2_frac = dim2_mean / denom
            final_word_stats[ipa][dim1] = dim1_frac
            final_word_stats[ipa][dim2] = dim2_frac
    return final_word_stats

def compute_mean_score(final_word_stats):
    for ipa, dim_scores in final_word_stats.items():
        for dim, scores in dim_scores.items():
            mean_score = sum(scores) / len(scores)
            final_word_stats[ipa][dim] = mean_score
    return final_word_stats

def save_file(final_word_stats, file_path):
    
    with open(file_path, "wb") as f:
        pkl.dump(final_word_stats, f)
    return

# layer_start, layer_end = layer_starts[2], layer_ends[2]
# lang = "art"
# data_type = "ipa"
# COMPUTE_RULE = "answer_only"
# FRACTION_WHEN_ANSWER_ONLY = True
# CHECK_MODEL_RESPONSE = True
# sampling_rate = 20
# constructed = get_constructed(lang)

# final_word_stats = {}
# data_path, output_dir = get_data_path(lang, data_type)
# show_arguments(data_type=data_type, lang=lang, compute_rule=COMPUTE_RULE, layer_start=layer_start, layer_end=layer_end, sampling_rate=sampling_rate)
# final_word_stats = compute_attention_by_language(lang=lang, layer_start=layer_start, layer_end=layer_end, final_word_stats=final_word_stats, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, data_path=data_path, output_dir=output_dir, sampling_rate=sampling_rate, fraction_when_answer_only=FRACTION_WHEN_ANSWER_ONLY)
# file_path = "src/analysis/heatmap/results"
# if not os.path.exists(file_path):
#     os.makedirs(file_path, exist_ok=True)
# file_name = f"{data_type}_{lang}_sampling_every_{sampling_rate}_{COMPUTE_RULE}_check_model_response_{CHECK_MODEL_RESPONSE}_{layer_start}_{layer_end}.pkl"
# save_file(final_word_stats, os.path.join(file_path, file_name))
# plot_ranked_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
# plot_sampled_word_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
# USE_SOFTMAX = True
# plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)
# USE_SOFTMAX = False
# plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)

# exit()
import time

# sampling_rate = 1
# constructed = True
# single_language = True
# for COMPUTE_RULE in ["answer_only", "fraction", "naive"]:
#     for data_type in ["ipa", "audio"]:
#         for layer_start, layer_end in zip(layer_starts, layer_ends):
#             start_time = time.time()
#             final_word_stats = {}
#             for lang in con_langs:
#                 data_path, output_dir = get_data_path(lang, data_type)
#                 show_arguments(data_type=data_type, lang=lang, compute_rule=COMPUTE_RULE, layer_start=layer_start, layer_end=layer_end, sampling_rate=sampling_rate)
#                 final_word_stats, processed_words = compute_attention_by_language(lang=lang, final_word_stats=final_word_stats, layer_start=layer_start, layer_end=layer_end, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, data_path=data_path, output_dir=output_dir, sampling_rate=sampling_rate, single_language=single_language)
#                 file_path = "src/analysis/heatmap/results"
#             if not single_language:
#                 final_word_stats = compute_mean_score(final_word_stats)
#             lang = "Constructed"
#             file_name = f"{data_type}_{lang}_{COMPUTE_RULE}_check_model_response_{CHECK_MODEL_RESPONSE}_{layer_start}_{layer_end}_sampling_every_{sampling_rate}_processed_words_{processed_words}.pkl"
#             save_file(final_word_stats, os.path.join(file_path, file_name))
#             plot_ranked_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
#             plot_sampled_word_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
#             USE_SOFTMAX = True
#             plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)
#             USE_SOFTMAX = False
#             plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)
#             end_time = time.time()
#             print(f"Time taken: {end_time - start_time} seconds")

sampling_rate = 1
constructed = False
single_language = False
FRACTION_WHEN_ANSWER_ONLY = True
CHECK_MODEL_RESPONSE = True
for COMPUTE_RULE in ["answer_only", "fraction", "naive"]:
    for data_type in ["ipa", "audio"]:
        for layer_start, layer_end in zip(layer_starts, layer_ends):
            final_word_stats = {}
            for lang in nat_langs:
                data_path, output_dir = get_data_path(lang, data_type)
                show_arguments(data_type=data_type, lang=lang, constructed=constructed, compute_rule=COMPUTE_RULE, layer_start=layer_start, layer_end=layer_end, sampling_rate=sampling_rate)
                final_word_stats, processed_words = compute_attention_by_language(lang=lang, data_path=data_path, layer_start=layer_start, layer_end=layer_end, output_dir=output_dir, final_word_stats=final_word_stats, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate, single_language=single_language)
                file_path = "src/analysis/heatmap/results"
            if not single_language:
                final_word_stats = compute_mean_score(final_word_stats)
            lang = "Natural"
            file_name = f"{data_type}_{lang}_{COMPUTE_RULE}_check_model_response_{CHECK_MODEL_RESPONSE}_{layer_start}_{layer_end}_processed_words_{processed_words}.pkl"
            save_file(final_word_stats, os.path.join(file_path, file_name))
            plot_ranked_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
            plot_sampled_word_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
            USE_SOFTMAX = True
            plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)
            USE_SOFTMAX = False
            plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)
            
