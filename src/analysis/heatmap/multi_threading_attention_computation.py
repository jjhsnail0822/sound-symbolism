# python src/analysis/heatmap/compute_attention_by_language.py

import json
import os
import re
import gc
import time
from typing import Union, Dict, List, Tuple
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
        output_dir = f"results/experiments/understanding/attention_heatmap/con/qwen7B/numpy/semantic_dimension/{data_type}/{lang}/generation_attention"
    else:
        data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"
        output_dir = f"results/experiments/understanding/attention_heatmap/nat/qwen7B/numpy/semantic_dimension/{data_type}/{lang}/generation_attention"
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
    'ʧ', 'ʨ', 'ʩ', 'ʪ', 'ʫ', 'ʬ', 'ʭ', 'ʮ', 'ʯ', 't͡ʃ'
    'ɴ', 'ɕ', 'd͡ʑ', 't͡ɕ', 'ʑ', 'ɰ', 'ã', 'õ', 'ɯ̃', 'ĩ', 'ẽ', 'ɯː', 'aː', 'oː', 'iː', 'eː', 'ej', 'ow'
]
consonants = [
    'b','d','ɟ','ɡ','ɖ','ɢ',    # voiced_stops
    'p','t','c','k','ʈ','q',    # voiceless_stops
    'v','ð','z','ʒ','ʐ','ʝ','ɣ','ʁ','ʕ','ɦ','β',    # voiced_fricatives
    'f','s','ʃ','θ','ç','x','χ','ħ','h','ɸ','ʂ','ɬ',    # voiceless_fricatives
    'm','ɱ','n','ɳ','ɲ','ŋ','ɴ','l','ɭ','ʎ','ʟ','ʀ','r','ɾ','ɽ','ɹ','ɺ','ɻ','j','ɰ',    # sonorants
    'ʔ','w','ʡ','ʢ', 'ʦ', 'ʧ', 't͡ʃ'   # others
]
vowels = [
    'i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ʘ', 'ɛ', 'œ', 'æ', 'a', 'ɶ', 'ej',  # Front vowels
    'ɨ', 'ʉ', 'ɯ', 'u', 'ʊ', 'ɤ', 'o', 'ɜ', 'ɝ', 'ɞ', 'ʌ', 'ɔ', 'ɑ', 'ɒ', 'ow'  # Central/Back vowels
]
ipa_dict = {
    "voiced_stops": ['b','d','ɟ','ɡ','ɖ','ɢ'],
    "voiceless_stops": ['p','t','c','k','ʈ','q'],
    "voiced_fricatives": ['v','ð','z','ʒ','ʐ','ʝ','ɣ','ʁ','ʕ','ɦ','β'],
    "voiceless_fricatives": ['f','s','ʃ','θ','ç','x','χ','ħ','h','ɸ','ʂ','ɬ'],
    "sonorants": ['m','ɱ','n','ɳ','ɲ','ŋ','ɴ','l','ɭ','ʎ','ʟ','ʀ','r','ɾ','ɽ','ɹ','ɺ','ɻ','j','ɰ'],
    "others": ['ʔ','w','ʡ','ʢ'],
    "front": ['i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ʘ', 'ɛ', 'œ', 'æ', 'a', 'ɶ', 'ej'],
    "back": ['ɨ', 'ʉ', 'ɯ', 'u', 'ʊ', 'ɤ', 'o', 'ɜ', 'ɝ', 'ɞ', 'ʌ', 'ɔ', 'ɑ', 'ɒ', 'ow'],
}

feature_to_score = json.load(open("data/processed/art/resources/feature_to_score.json", "r"))

def clean_token(token:str) -> str:
    return re.sub(r'^[ĠĊ\[\],.:;!?\n\r\t]+|[ĠĊ\[\],.:;!?\n\r\t]+$', '', token)

def get_word_list(lang:str=lang, data_path:str=None) -> list[str]:
    if lang not in ["art", "en", "fr", "ja", "ko"]:
        raise ValueError(f"Language {lang} not supported")
    
    with open(data_path, "r") as f:
        data = json.load(f)[lang]
    word_list = [sample["word"] for sample in data]
    print(f"Number of words: {len(word_list)}")
    return word_list

def show_arguments(model_name:str=model_path, data_type:str=data_type, lang:str=lang, layer_start:int=0, layer_end:int=27, constructed:bool=constructed, check_model_response:bool=CHECK_MODEL_RESPONSE, compute_rule:str=COMPUTE_RULE, sampling_rate:int=None):
    print(f"Model: {model_name}\nData type: {data_type}\nLanguage: {lang}\nLayer start: {layer_start}\nLayer end: {layer_end}\nConstructed: {constructed}\nCheck model response: {check_model_response}\nCompute rule: {compute_rule}\nSampling rate: {sampling_rate}")

def model_guessed_incorrectly(response, dim1, dim2, answer) -> bool:
    if dim1 == answer and response == "1":
        return False
    elif dim2 == answer and response == "2":
        return False
    return True

def find_basic_info(word, output_dir=None, dim_pairs=dim_pairs, word_stats=None) -> tuple[list[str], dict]:
    ipa_list = []
    for dim1, dim2 in dim_pairs:
        data_dir = os.path.join(output_dir, f"{dim1}_{dim2}", f"{word}_{dim1}_{dim2}.pkl")
        alt_dir = os.path.join(output_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl")
        if not os.path.exists(data_dir) or not os.path.exists(alt_dir):
            continue
        else:
            alt = pkl.load(open(alt_dir, "rb"))
            ipa_tokens = alt["ipa_tokens"]
            converted_ipa_tokens = convert_ipa_tokens_to_ipa_string_per_token(ipa_tokens)
            ipa_list = [clean_token(token) for token in converted_ipa_tokens]
            if "ʧ" in ipa_tokens[0].split() or "t͡ʃ" in ipa_tokens[0].split():
                print(f"Found ʧ or t͡ʃ in {word}, {ipa_tokens}")
                breakpoint()
            word_stats = {ipa:{} for ipa in ipa_list}
            word_stats = {k: v for k, v in word_stats.items() if k != ""} # Remove empty text
            break
    return ipa_list, word_stats

def get_semdim_matrix_and_labels(word_stats, dim_pairs, ipa_list):
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
    title = f'MT | {data_type} | Lang: {lang} | rule: {compute_rule} | L{start_layer}-L{end_layer} | check_model_response: {check_model_response} | sampling_rate: {sampling_rate}\nIPA-Semantic Dimension Attention Heatmap'
    # title = f'{data_type} | Lang: {lang} | rule: {compute_rule} | L{start_layer}-L{end_layer} | check_model_response: {check_model_response} | sampling_rate: {sampling_rate}\nIPA-Semantic Dimension Attention Heatmap'
    file_name = f"npmt_{lang.upper()}_{data_type}_generation_attention_L{start_layer}_L{end_layer}_sampling_{sampling_rate}"
    # file_name = f"{lang.upper()}_{data_type}_generation_attention_L{start_layer}_L{end_layer}_sampling_{sampling_rate}"
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
    file_name = f"npmt_{lang.upper()}_{data_type}_generation_attention_L{start_layer}_L{end_layer}_sampling_{sampling_rate}_ranked"
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
            # Handle both scalar and array cases
            if isinstance(score, (list, np.ndarray)):
                # If score is a list/array, take the mean
                if len(score) > 0:
                    score = np.mean(score)
                else:
                    score = np.nan
            
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
    title = f'MT | Lang: {lang} | Data type: {data_type} | rule: {compute_rule} | L{start_layer}-L{end_layer} | check_model_response: {check_model_response} | sampling_rate: {sampling_rate}\nIPA-Semantic Dimension Attention IPA-wise Scaled Heatmap (softmax: {use_softmax})'
    file_name = f"npmt_{lang.upper()}_{data_type}_attention_L{start_layer}_L{end_layer}_ipa_wise_scaled_softmax_{use_softmax}_sampling_{sampling_rate}"
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

def paper_plot(word_stats_list, data_type_list, lang_list, save_path=None, 
               compute_rule_list=None, 
               dim_pairs=dim_pairs, answer_list=None, 
               ipa_list=None, semdim_list=None, show_y_labels=True):
    # Default semantic dimensions
    if semdim_list is None:
        semdim_list = ['big', 'small', 'fast', 'slow']
    
    # Default IPA symbols
    if ipa_list is None:
        ipa_list = ['i', 'ɑ', 'p', 'ʃ', 't', 'ʦ', 'k', 'm', 'n']
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    
    if save_path is None:
        save_path = 'results/plots/attention/paper_plots/'
    os.makedirs(save_path, exist_ok=True)
    
    # Fixed file specifications
    file_spec_list = [
        # ("constructed", "audio", "fraction"),
        # ("constructed", "ipa", "fraction"),
        ("natural", "audio", "fraction"),
        ("natural", "ipa", "fraction"),
    ]
    
    # Create figure with 4 subplots - one for each file specification
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Process each subplot according to file_spec_list order
    for idx, (word_stats, data_type, lang, compute_rule) in enumerate(zip(word_stats_list, data_type_list, lang_list, compute_rule_list)):
        ax = axes[idx]
        print(f"Creating subplot {idx+1}/4: {lang.capitalize()} - {data_type.capitalize()}")
        
        # Filter data based on user-specified ipa_list and semdim_list
        # Only keep IPAs that exist in word_stats
        current_ipa_list = [ipa for ipa in ipa_list if ipa in word_stats]
        
        # Only keep semantic dimensions that exist in word_stats
        current_semdim_list = [dim for dim in semdim_list if any(dim in word_stats[ipa] for ipa in current_ipa_list if ipa in word_stats)]
        
        # Create matrix and highlight matrix
        matrix = np.zeros((len(current_semdim_list), len(current_ipa_list)))
        highlight_matrix = np.zeros((len(current_semdim_list), len(current_ipa_list)), dtype=bool)
        matrix[:] = np.nan
        
        # Create color matrix for visualization
        color_matrix = np.zeros((len(current_semdim_list), len(current_ipa_list)))
        color_matrix[:] = np.nan
        
        # Fill matrix with values and determine which cells to highlight
        for i, semdim in enumerate(current_semdim_list):
            for j, ipa in enumerate(current_ipa_list):
                if ipa in word_stats and semdim in word_stats[ipa]:
                    score = word_stats[ipa][semdim]
                    if isinstance(score, (list, np.ndarray)):
                        if len(score) > 0:
                            score = np.mean(score)
                        else:
                            score = np.nan
                    
                    if not np.isnan(score):
                        matrix[i, j] = score
                        
                        if score > 0.5:
                            color_matrix[i, j] = 0.5 + ((score - 0.5) / 0.5) * 0.2
                        else:
                            color_matrix[i, j] = 0.3 + (score / 0.5) * 0.2
                        
                        # Check if this semantic dimension is part of a pair
                        for dim1, dim2 in dim_pairs:
                            if semdim == dim1 or semdim == dim2:
                                other_dim = dim2 if semdim == dim1 else dim1
                                
                                if ipa in word_stats and other_dim in word_stats[ipa]:
                                    other_score = word_stats[ipa][other_dim]
                                    if isinstance(other_score, (list, np.ndarray)):
                                        if len(other_score) > 0:
                                            other_score = np.mean(other_score)
                                        else:
                                            other_score = np.nan
                                    
                                    if not np.isnan(other_score) and score > other_score:
                                        highlight_matrix[i, j] = True
                                break
        
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            print(f"[WARN] Empty matrix for lang '{lang}'. Skipping subplot.")
            continue
        
        # Create mask for highlighting
        mask = np.isnan(matrix)
        highlight_mask = highlight_matrix & ~mask
        
        # Create custom colormap
        from matplotlib.colors import LinearSegmentedColormap
        
        colors = ['#f03e3e', '#ff6b6b', '#ffe3e3', 'white', '#ebfbee', '#51cf66', '#2f9e44']
        positions = [0, 0.25, 0.4999, 0.5, 0.5001, 0.75, 1.0]
        custom_cmap = LinearSegmentedColormap.from_list('RedWhiteGreen', list(zip(positions, colors)), N=100)
        
        # Plot the heatmap
        sns.heatmap(
            color_matrix, ax=ax,
            cmap=custom_cmap,
            cbar=False,
            xticklabels=current_ipa_list, 
            yticklabels=current_semdim_list if (idx == 0 and show_y_labels) else [],
            linewidths=0.5, linecolor='black', square=True,
            annot=False,
            mask=mask,
            vmin=0.45, vmax=0.55,
            annot_kws={"size": 20}
        )
        
        # Make the outer border thicker
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        
        # Add thick lines at top and bottom of heatmap
        # Get the heatmap boundaries
        heatmap_height = len(current_semdim_list)
        heatmap_width = len(current_ipa_list)
        
        # Add thick horizontal line at the top of heatmap
        ax.axhline(y=0, xmin=0, xmax=1, color='black', linewidth=5)
        # Add thick horizontal line at the bottom of heatmap
        ax.axhline(y=heatmap_height, xmin=0, xmax=1, color='black', linewidth=5)
        
        # Make specific grid lines thicker
        try:
            alpha_idx = current_ipa_list.index('ɑ')
            if alpha_idx + 1 < len(current_ipa_list):
                for i in range(len(current_semdim_list) + 1):
                    ax.plot([alpha_idx + 1, alpha_idx + 1], [i, i], color='black', linewidth=3, solid_capstyle='butt')
        except ValueError:
            pass
        
        try:
            k_idx = current_ipa_list.index('k')
            if k_idx + 1 < len(current_ipa_list):
                for i in range(len(current_semdim_list) + 1):
                    ax.plot([k_idx + 1, k_idx + 1], [i, i], color='black', linewidth=3, solid_capstyle='butt')
        except ValueError:
            pass
        
        # Add value annotations
        for i in range(len(current_semdim_list)):
            for j in range(len(current_ipa_list)):
                if not np.isnan(matrix[i, j]):
                    value = matrix[i, j]
                    percentage = value * 100
                    formatted_value = f"{percentage:.1f}"
                    ax.text(j + 0.5, i + 0.5, formatted_value, 
                           ha='center', va='center', fontsize=20)
        
        # Add pair separators
        if dim_pairs is not None:
            drawn = set()
            for dim1, dim2 in dim_pairs:
                try:
                    idx1 = current_semdim_list.index(dim1)
                except ValueError:
                    idx1 = None
                try:
                    idx2 = current_semdim_list.index(dim2)
                except ValueError:
                    idx2 = None
                
                if idx1 is not None and idx2 is not None:
                    top = min(idx1, idx2)
                    bottom = max(idx1, idx2)
                    if top not in drawn:
                        ax.axhline(y=top, xmin=-100, xmax=1, color='black', linewidth=3)
                        drawn.add(top)
                    if (bottom+1) not in drawn:
                        ax.axhline(y=bottom+1, xmin=-100, xmax=1, color='black', linewidth=3)
                        drawn.add(bottom+1)
                elif idx1 is not None:
                    if idx1 not in drawn:
                        ax.axhline(y=idx1, xmin=-100, xmax=1, color='black', linewidth=3)
                        drawn.add(idx1)
                    if (idx1+1) not in drawn:
                        ax.axhline(y=idx1+1, xmin=-100, xmax=1, color='black', linewidth=3)
                        drawn.add(idx1+1)
                elif idx2 is not None:
                    if idx2 not in drawn:
                        ax.axhline(y=idx2, xmin=-100, xmax=1, color='black', linewidth=3)
                        drawn.add(idx2)
                    if (idx2+1) not in drawn:
                        ax.axhline(y=idx2+1, xmin=-100, xmax=1, color='black', linewidth=3)
                        drawn.add(idx2+1)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=24)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        if idx == 0 and show_y_labels:
            plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
        
        # Bold answer labels if provided (only for first subplot)
        if answer_list is not None and idx == 0 and show_y_labels:
            yticklabels = ax.get_yticklabels()
            for i, label in enumerate(yticklabels):
                if label.get_text() in answer_list:
                    label.set_fontweight('bold')
            ax.set_yticklabels(yticklabels)
        
        # Set title for each subplot
        if data_type == "ipa":
            ax.set_title(f'{lang.capitalize()} - IPA', fontsize=24, pad=10)
        elif data_type == "audio":
            ax.set_title(f'{lang.capitalize()} - Audio', fontsize=24, pad=10)
    
    plt.tight_layout()
    
    # Save the plot
    file_name = f"paper_plot_combined"
    if compute_rule_list and len(set(compute_rule_list)) == 1:
        file_name += f"_{compute_rule_list[0]}"
    pdf_file_name = file_name + ".pdf"
    png_file_name = file_name + ".png"
    pdf_file_path = os.path.join(save_path, pdf_file_name)
    png_file_path = os.path.join(save_path, png_file_name)
    
    plt.savefig(pdf_file_path, dpi=300, bbox_inches='tight')
    print(f"Combined paper plot saved to {pdf_file_path}")
    
    plt.savefig(png_file_path, dpi=300, bbox_inches='tight')
    print(f"Combined paper plot saved to {png_file_path}")
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
    # attention_matrices = data["attention_matrices"]
    # relevant_indices = data["relevant_indices"]
    # dim1, dim2 = data["dimension1"], data["dimension2"]
    # answer, word_tokens, option_tokens, tokens = data["answer"], data["word_tokens"], data["option_tokens"], data["tokens"]
    # ipa_tokens = alt["ipa_tokens"]
    # response, input_word, target_indices = alt["response"], alt["input_word"], alt["target_indices"]
    target_indices = alt["target_indices"]
    wlen, d1len, d2len = len(target_indices["word"]), len(target_indices["dim1"]), len(target_indices["dim2"])
    # dim1_range = range(wlen, wlen+d1len)
    # dim2_range = range(wlen+d1len, wlen+d1len+d2len)
    return data["attention_matrices"], data["relevant_indices"], data["dimension1"], data["dimension2"], data["answer"], data["word_tokens"], data["option_tokens"], data["tokens"], alt["ipa_tokens"], alt["response"], alt["input_word"], alt["target_indices"], len(target_indices["word"]), len(target_indices["dim1"]), len(target_indices["dim2"]), range(wlen, wlen+d1len), range(wlen+d1len, wlen+d1len+d2len)
    # return attention_matrices, relevant_indices, dim1, dim2, answer, word_tokens, option_tokens, tokens, ipa_tokens, response, input_word, target_indices, wlen, d1len, d2len, dim1_range, dim2_range

def add_to_word_stats(word_stats, ipa, dim, values, ipa_symbols=ipa_symbols):
    if ipa in ipa_symbols and ipa not in word_stats.keys():
        word_stats[ipa] = {}
    if dim not in word_stats[ipa].keys():
        word_stats[ipa][dim] = []
    word_stats[ipa][dim].extend(values)
    return word_stats

def add_to_word_layer_stats(word_layer_stats, layer, ipa, dim, values, ipa_symbols=ipa_symbols):
    if layer not in word_layer_stats:
        word_layer_stats[layer] = {}
    if ipa in ipa_symbols and ipa not in word_layer_stats[layer].keys():
        word_layer_stats[layer][ipa] = {}
    if dim not in word_layer_stats[layer][ipa].keys():
        word_layer_stats[layer][ipa][dim] = []
    word_layer_stats[layer][ipa][dim].extend(values)
    return word_layer_stats

def compute_semdim_score_with_naive_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats, word_layer_stats=None) -> tuple[dict, dict]:
    if word_layer_stats is None:
        word_layer_stats = {}
    ipa_runs = get_ipa_runs(ipa_list)
    for ipa, start_idx, end_idx in ipa_runs:
        if ipa == "" or ipa == " ":
            continue
        for dim, dim_range in [(dim1, dim1_range), (dim2, dim2_range)]:
            all_values = []
            for layer in range(start_layer, min(end_layer+1, n_layer)):
                layer_values = []
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
                        layer_values.append(sum_score)
                all_values.extend(layer_values)
                word_layer_stats = add_to_word_layer_stats(word_layer_stats, layer, ipa, dim, layer_values)
            word_stats = add_to_word_stats(word_stats, ipa, dim, all_values)
    return word_stats, word_layer_stats

def compute_semdim_score_with_answer_only_rule(ipa_list:list[str], dim1:str, dim2:str, answer:str, dim1_range:list[int], dim2_range:list[int], start_layer:int, end_layer:int, n_layer:int, n_head:int, wlen:int, d1len:int, d2len:int, attn_layers:list[np.array], word_stats:dict, word_layer_stats:dict=None) -> tuple[dict, dict]:
    if word_layer_stats is None:
        word_layer_stats = {}
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
        
        final_check_values = []
        
        for layer in range(start_layer, min(end_layer+1, n_layer)):
            layer_values = []
            for head in range(n_head):
                check_attention = attn_layers[layer][0, head, check_dim_range, ipa_idx]
                check_sum = np.sum(check_attention)
                
                ignore_attention = attn_layers[layer][0, head, ignore_dim_range, ipa_idx]
                ignore_sum = np.sum(ignore_attention)
                
                denom = check_sum + ignore_sum
                if denom == 0:
                    frac_check = 0.0
                else:
                    frac_check = check_sum / denom
                
                layer_values.append(frac_check)
            final_check_values.extend(layer_values)
            word_layer_stats = add_to_word_layer_stats(word_layer_stats, layer, ipa, check_dim, layer_values)
        
        word_stats = add_to_word_stats(word_stats, ipa, check_dim, final_check_values)
    
    return word_stats, word_layer_stats

def compute_semdim_score_with_fraction_rule(ipa_list:list[str], dim1:str, dim2:str, dim1_range:list[int], dim2_range:list[int], start_layer:int, end_layer:int, n_layer:int, n_head:int, wlen:int, d1len:int, d2len:int, attn_layers:list[np.array], word_stats:dict, word_layer_stats:dict=None) -> tuple[dict, dict]:
    if word_layer_stats is None:
        word_layer_stats = {}
    ipa_runs = get_ipa_runs(ipa_list)
    for ipa, start_idx, end_idx in ipa_runs:
        if ipa == "":
            continue
        final_dim1_values = []
        final_dim2_values = []
        for layer in range(start_layer, min(end_layer+1, n_layer)):
            layer_dim1_values = []
            layer_dim2_values = []
            for head in range(n_head):
                attention_slice_dim1 = attn_layers[layer][0, head, dim1_range, start_idx:end_idx+1]
                sum_dim1 = np.sum(attention_slice_dim1)
                
                attention_slice_dim2 = attn_layers[layer][0, head, dim2_range, start_idx:end_idx+1]
                sum_dim2 = np.sum(attention_slice_dim2)
                
                denom = sum_dim1 + sum_dim2
                if denom == 0:
                    frac_dim1, frac_dim2 = 0.0, 0.0
                else:
                    frac_dim1 = sum_dim1 / denom
                    frac_dim2 = sum_dim2 / denom
                # print(f"Frac dim1: {frac_dim1}, Frac dim2: {frac_dim2}")
                layer_dim1_values.append(frac_dim1)
                layer_dim2_values.append(frac_dim2)
            final_dim1_values.extend(layer_dim1_values)
            final_dim2_values.extend(layer_dim2_values)
            word_layer_stats = add_to_word_layer_stats(word_layer_stats, layer, ipa, dim1, layer_dim1_values)
            word_layer_stats = add_to_word_layer_stats(word_layer_stats, layer, ipa, dim2, layer_dim2_values)
        
        word_stats = add_to_word_stats(word_stats, ipa, dim1, final_dim1_values)
        word_stats = add_to_word_stats(word_stats, ipa, dim2, final_dim2_values)
    return word_stats, word_layer_stats

def convert_ipa_tokens_to_ipa_string_per_token(ipa_tokens, processor=processor, ipa_symbols=ipa_symbols) -> list[str]:
    token_cache = {}
    symbol_for_token = [None] * len(ipa_tokens)
    
    i = 0
    while i < len(ipa_tokens):
        found = False
        for j in range(min(len(ipa_tokens), i+4), i, -1):
            token_key = tuple(ipa_tokens[i:j])
            
            if token_key not in token_cache:
                token_cache[token_key] = processor.tokenizer.convert_tokens_to_string(ipa_tokens[i:j]).strip()
            
            converted_str = token_cache[token_key]
            if converted_str in ipa_symbols:
                for k in range(i, j):
                    symbol_for_token[k] = converted_str
                i = j
                found = True
                break
        
        if not found:
            single_key = tuple([ipa_tokens[i]])
            if single_key not in token_cache:
                token_cache[single_key] = processor.tokenizer.convert_tokens_to_string([ipa_tokens[i]]).strip()
            
            converted_str = token_cache[single_key]
            symbol_for_token[i] = converted_str if converted_str in ipa_symbols else ''
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
        word_layer_stats:dict=None,
        check_model_response:bool=CHECK_MODEL_RESPONSE,
        compute_rule:str=COMPUTE_RULE,
        model_path:str=model_path,
    ) -> tuple[dict, dict]:
    ipa_list, word_stats = find_basic_info(word=word, output_dir=output_dir, dim_pairs=dim_pairs, word_stats=word_stats)
    if not ipa_list:
        return word_stats, {}
    if word_layer_stats is None:
        word_layer_stats = {}
    load_dir = f"results/experiments/understanding/attention_heatmap/combined/{data_type}/{lang}/qwen7B/numpy"
    word_data = None
    incorrect_count = 0
    
    for dim1, dim2 in dim_pairs:
        data_dir = os.path.join(output_dir, f"{dim1}_{dim2}", f"{word}_{dim1}_{dim2}.pkl")
        alt_dir = os.path.join(output_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl")
        if not os.path.exists(data_dir) or not os.path.exists(alt_dir):
            continue
        data = pkl.load(open(data_dir, "rb"))
        alt = pkl.load(open(alt_dir, "rb"))
        attention_matrices, _, dim1, dim2, answer, word_tokens, option_tokens, tokens, ipa_tokens, response, input_word, target_indices, wlen, d1len, d2len, dim1_range, dim2_range = get_data(data, alt)
        if check_model_response and model_guessed_incorrectly(response, dim1, dim2, answer):
            # print(f"Model guessed incorrectly for {word} {dim1}-{dim2}")
            incorrect_count += 1
            continue
        
        ipa_list = convert_ipa_tokens_to_ipa_string_per_token(ipa_tokens)
        attn_layers = attention_matrices[0]
        n_layer = len(attn_layers)
        n_head = attn_layers[0].shape[1]
        if compute_rule == "fraction":
            word_stats, word_layer_stats = compute_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats, word_layer_stats)
        elif compute_rule == "answer_only":
            word_stats, word_layer_stats = compute_semdim_score_with_answer_only_rule(ipa_list, dim1, dim2, answer, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats, word_layer_stats)
        elif compute_rule == "naive":
            word_stats, word_layer_stats = compute_semdim_score_with_naive_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats, word_layer_stats)
    del word_data
    if len(dim_pairs) == incorrect_count:
        print(f"Model guessed incorrectly for all {len(dim_pairs)} dimensions for {word}")
    return word_stats, word_layer_stats

final_word_stats = {}

def process_single_word_thread_safe(word, data_type, lang, layer_start, layer_end, dim_pairs, data_path, output_dir, check_model_response, compute_rule, model_path):
    """Thread-safe function to process a single word"""
    try:
        word_stats, word_layer_stats = compute_single_word_attention_score(
            word=word, data_type=data_type, lang=lang, start_layer=layer_start, end_layer=layer_end,
            dim_pairs=dim_pairs, data_path=data_path, output_dir=output_dir,
            check_model_response=check_model_response, compute_rule=compute_rule, model_path=model_path
        )
        return word, word_stats, word_layer_stats
    except Exception as e:
        print(f"Error processing word {word}: {e}")
        return word, None, None

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
    final_word_layer_stats:dict=None,
    sampling_rate:int=1,
    single_language:bool=True,
    layer_one_by_one:bool=False,
    max_workers:int=24,
):
    if final_word_layer_stats is None:
        final_word_layer_stats = {}
    semdim_set = set()
    for dim1, dim2 in dim_pairs:
        semdim_set.add(dim1)
        semdim_set.add(dim2)
    semdim_list = sorted(list(semdim_set))
    word_list = get_word_list(lang=lang, data_path=data_path)
    filtered_word_list = []
    word_count = 0
    for word in word_list:
        word_count += 1
        if word_count % sampling_rate == 0:
            filtered_word_list.append(word)
    
    print(f"Processing {len(filtered_word_list)} words with {max_workers} threads")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_word = {
            executor.submit(
                process_single_word_thread_safe,
                word, data_type, lang, layer_start, layer_end, dim_pairs, 
                data_path, output_dir, check_model_response, compute_rule, model_path
            ): word for word in filtered_word_list
        }
        
        for future in tqdm(as_completed(future_to_word), total=len(filtered_word_list), desc="Processing words"):
            word = future_to_word[future]
            try:
                word, word_stats, word_layer_stats = future.result()
                if word_stats is not None:
                    results.append((word, word_stats, word_layer_stats))
            except Exception as e:
                print(f"Exception occurred for word {word}: {e}")
    
    processed_words = 0
    for word, word_stats, word_layer_stats in results:
        if word_stats is None:
            continue
        # Process final_word_stats (existing logic)
        for ipa, dim_scores in word_stats.items():
            if not ipa or not dim_scores:
                continue
            for dim, score in dim_scores.items():
                if ipa not in final_word_stats:
                    final_word_stats[ipa] = {}
                if dim not in final_word_stats[ipa]:
                    final_word_stats[ipa][dim] = []
                try:
                    final_word_stats[ipa][dim].extend(score)
                except Exception as e:
                    print(f"Error extending score for {ipa} {dim} {score}: {e}")
                    breakpoint()
        
        # Process final_word_layer_stats (new logic)
        for layer, layer_stats in word_layer_stats.items():
            if layer not in final_word_layer_stats:
                final_word_layer_stats[layer] = {}
            for ipa, dim_scores in layer_stats.items():
                if not ipa or not dim_scores:
                    continue
                if ipa not in final_word_layer_stats[layer]:
                    final_word_layer_stats[layer][ipa] = {}
                for dim, score in dim_scores.items():
                    if dim not in final_word_layer_stats[layer][ipa]:
                        final_word_layer_stats[layer][ipa][dim] = []
                    final_word_layer_stats[layer][ipa][dim].extend(score)
        processed_words += 1
    
    print(f"Processed {processed_words} words successfully")
    return final_word_stats, final_word_layer_stats, processed_words

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

def compute_mean_score_by_layer(final_word_layer_stats):
    for layer, layer_stats in final_word_layer_stats.items():
        for ipa, dim_scores in layer_stats.items():
            for dim, scores in dim_scores.items():
                mean_score = sum(scores) / len(scores)
                final_word_layer_stats[layer][ipa][dim] = mean_score
    return final_word_layer_stats

def save_file(final_word_stats, file_path, final_word_layer_stats=None):
    with open(file_path, "wb") as f:
        pkl.dump(final_word_stats, f)
        print(f"Saved file to {file_path}")
    if final_word_layer_stats is not None:
        # Save layer stats to a separate file
        layer_file_path = file_path.replace('.pkl', '_layer_stats.pkl')
        with open(layer_file_path, "wb") as f:
            pkl.dump(final_word_layer_stats, f)
            print(f"Saved layer stats to {layer_file_path}")
    return

### For Constructed
# layer_start, layer_end = layer_starts[1], layer_ends[1]
# lang = "art"
# data_type = "audio"
# COMPUTE_RULE = "fraction"
# FRACTION_WHEN_ANSWER_ONLY = True
# CHECK_MODEL_RESPONSE = True
# sampling_rate = 1
# constructed = get_constructed(lang)
# single_language = True
# final_word_stats = {}
# data_path, output_dir = get_data_path(lang, data_type)
# show_arguments(data_type=data_type, lang=lang, compute_rule=COMPUTE_RULE, layer_start=layer_start, layer_end=layer_end, sampling_rate=sampling_rate)
# final_word_stats, final_word_layer_stats, processed_words = compute_attention_by_language(
#     lang=lang, 
#     layer_start=layer_start, 
#     layer_end=layer_end, 
#     final_word_stats=final_word_stats, 
#     compute_rule=COMPUTE_RULE, 
#     check_model_response=CHECK_MODEL_RESPONSE, 
#     data_path=data_path, 
#     output_dir=output_dir, 
#     sampling_rate=sampling_rate, 
#     single_language=single_language,
#     max_workers=24
# )

# # Compute mean scores before plotting
# final_word_stats = compute_mean_score(final_word_stats)
# final_word_layer_stats = compute_mean_score_by_layer(final_word_layer_stats)

# file_path = "src/analysis/heatmap/results"
# if not os.path.exists(file_path):
#     os.makedirs(file_path, exist_ok=True)
# lang = "Constructed"
# file_name = f"np_{data_type}_{lang}_sampling_every_{sampling_rate}_{COMPUTE_RULE}_check_model_response_{CHECK_MODEL_RESPONSE}_{layer_start}_{layer_end}_processed_words_{processed_words}.pkl"

# final_word_stats = load_file(file_path=os.path.join(file_path, file_name))
# final_word_layer_stats = load_file(file_path=os.path.join(file_path, file_name.replace(".pkl", "_layer_stats.pkl")))

# save_file(final_word_stats, os.path.join(file_path, file_name), final_word_layer_stats)
# plot_ranked_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
# plot_sampled_word_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
# USE_SOFTMAX = True
# plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)
# USE_SOFTMAX = False
# plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)

base_file_path = "src/analysis/heatmap/results/{lang}_{data_type}_{compute_rule}.pkl"
supl_file_path = "src/analysis/heatmap/results/{lang}_{data_type}_{compute_rule}_layer_stats.pkl"
file_spec_list = [
    ("natural", "ipa", "fraction"),
    ("natural", "audio", "fraction"),
    ("constructed", "ipa", "fraction"),
    ("constructed", "audio", "fraction"),
]

# Load all data for combined plot
word_stats_list = []
data_type_list = []
lang_list = []
compute_rule_list = []

for file_spec in file_spec_list:
    lang, data_type, compute_rule = file_spec
    file_path = base_file_path.format(lang=lang, data_type=data_type, compute_rule=compute_rule)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            final_word_stats = pkl.load(f)
    else:
        supl_file_path = supl_file_path.format(lang=lang, data_type=data_type, compute_rule=compute_rule)
        with open(supl_file_path, "rb") as f:
            final_word_stats = pkl.load(f)
    
    word_stats_list.append(final_word_stats)
    data_type_list.append(data_type)
    lang_list.append(lang)
    compute_rule_list.append(compute_rule)

# Generate combined paper plot
paper_plot(word_stats_list, data_type_list, lang_list, compute_rule_list=compute_rule_list, show_y_labels=True)

# exit()

### For Natural
# sampling_rate = 1
# constructed = False
# single_language = False
# FRACTION_WHEN_ANSWER_ONLY = True
# CHECK_MODEL_RESPONSE = True
# layer_starts = [0]
# layer_ends = [1]
# data_type = "ipa"
# COMPUTE_RULE = "fraction"
# layer_start, layer_end = layer_starts[0], layer_ends[0]
# final_word_stats = {}
# final_word_layer_stats = {}
# final_processed_words = 0
# for lang in nat_langs:
#     data_path, output_dir = get_data_path(lang, data_type)
#     show_arguments(data_type=data_type, lang=lang, constructed=constructed, compute_rule=COMPUTE_RULE, layer_start=layer_start, layer_end=layer_end, sampling_rate=sampling_rate)
#     temp_word_stats, temp_word_layer_stats, processed_words = compute_attention_by_language(lang=lang, data_path=data_path, layer_start=layer_start, layer_end=layer_end, output_dir=output_dir, final_word_stats={}, final_word_layer_stats={}, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate, single_language=True, max_workers=1)
#     # Merge results
#     for ipa, dim_scores in temp_word_stats.items():
#         if ipa not in final_word_stats:
#             final_word_stats[ipa] = {}
#         for dim, scores in dim_scores.items():
#             if dim not in final_word_stats[ipa]:
#                 final_word_stats[ipa][dim] = []
#             final_word_stats[ipa][dim].extend(scores)
    
#     for layer, layer_stats in temp_word_layer_stats.items():
#         if layer not in final_word_layer_stats:
#             final_word_layer_stats[layer] = {}
#         for ipa, dim_scores in layer_stats.items():
#             if ipa not in final_word_layer_stats[layer]:
#                 final_word_layer_stats[layer][ipa] = {}
#             for dim, scores in dim_scores.items():
#                 if dim not in final_word_layer_stats[layer][ipa]:
#                     final_word_layer_stats[layer][ipa][dim] = []
#                 final_word_layer_stats[layer][ipa][dim].extend(scores)
#     final_processed_words += processed_words
#     file_path = "src/analysis/heatmap/results"

# # Compute mean scores after all languages are processed
# final_word_stats = compute_mean_score(final_word_stats)
# final_word_layer_stats = compute_mean_score_by_layer(final_word_layer_stats)
# lang = "Natural"
# file_name = f"{lang}_{data_type}_{COMPUTE_RULE}.pkl"
# save_file(final_word_stats, os.path.join(file_path, file_name), final_word_layer_stats)
# plot_ranked_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
# plot_sampled_word_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
# USE_SOFTMAX = True
# plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)
# USE_SOFTMAX = False
# plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)

# sampling_rate = 1
# constructed = True
# single_language = True
# layer_starts = [0]
# layer_ends = [27]
# for COMPUTE_RULE in ["fraction", "answer_only", "naive"]:
#     for data_type in ["ipa", "audio"]:
#         for layer_start, layer_end in zip(layer_starts, layer_ends):
#             final_word_stats = {}
#             final_word_layer_stats = {}
#             for lang in con_langs:
#                 data_path, output_dir = get_data_path(lang, data_type)
#                 show_arguments(data_type=data_type, lang=lang, compute_rule=COMPUTE_RULE, layer_start=layer_start, layer_end=layer_end, sampling_rate=sampling_rate)
#                 temp_word_stats, temp_word_layer_stats, processed_words = compute_attention_by_language(
#                     lang=lang, 
#                     layer_start=layer_start, 
#                     layer_end=layer_end, 
#                     final_word_stats={}, 
#                     final_word_layer_stats={},
#                     compute_rule=COMPUTE_RULE, 
#                     check_model_response=CHECK_MODEL_RESPONSE, 
#                     data_path=data_path, 
#                     output_dir=output_dir, 
#                     sampling_rate=sampling_rate, 
#                     single_language=single_language,
#                     max_workers=8
#                 )
#                 # Merge results
#                 for ipa, dim_scores in temp_word_stats.items():
#                     if ipa not in final_word_stats:
#                         final_word_stats[ipa] = {}
#                     for dim, scores in dim_scores.items():
#                         if dim not in final_word_stats[ipa]:
#                             final_word_stats[ipa][dim] = []
#                         final_word_stats[ipa][dim].extend(scores)
                
#                 for layer, layer_stats in temp_word_layer_stats.items():
#                     if layer not in final_word_layer_stats:
#                         final_word_layer_stats[layer] = {}
#                     for ipa, dim_scores in layer_stats.items():
#                         if ipa not in final_word_layer_stats[layer]:
#                             final_word_layer_stats[layer][ipa] = {}
#                         for dim, scores in dim_scores.items():
#                             if dim not in final_word_layer_stats[layer][ipa]:
#                                 final_word_layer_stats[layer][ipa][dim] = []
#                             final_word_layer_stats[layer][ipa][dim].extend(scores)
#                 file_path = "src/analysis/heatmap/results"
            
#             # Compute mean scores after all languages are processed
#             final_word_stats = compute_mean_score(final_word_stats)
#             final_word_layer_stats = compute_mean_score_by_layer(final_word_layer_stats)
#             lang = "Constructed"
#             file_name = f"np_{data_type}_{lang}_{COMPUTE_RULE}_check_model_response_{CHECK_MODEL_RESPONSE}_{layer_start}_{layer_end}_sampling_every_{sampling_rate}_processed_words_{processed_words}.pkl"
#             save_file(final_word_stats, os.path.join(file_path, file_name), final_word_layer_stats)
#             plot_ranked_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
#             plot_sampled_word_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
#             USE_SOFTMAX = True
#             plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)
#             USE_SOFTMAX = False
#             plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)

# sampling_rate = 1
# constructed = False
# single_language = False
# FRACTION_WHEN_ANSWER_ONLY = True
# CHECK_MODEL_RESPONSE = True
# layer_starts = [0]
# layer_ends = [1]
# for data_type in ["ipa", "audio"]:
#     # for COMPUTE_RULE in ["fraction", "answer_only", "naive"]:
#     for COMPUTE_RULE in ["fraction", "answer_only"]:
#         for layer_start, layer_end in zip(layer_starts, layer_ends):
#             final_word_stats = {}
#             final_word_layer_stats = {}
#             for lang in nat_langs:
#                 data_path, output_dir = get_data_path(lang, data_type)
#                 show_arguments(data_type=data_type, lang=lang, constructed=constructed, compute_rule=COMPUTE_RULE, layer_start=layer_start, layer_end=layer_end, sampling_rate=sampling_rate)
#                 temp_word_stats, temp_word_layer_stats, processed_words = compute_attention_by_language(lang=lang, data_path=data_path, layer_start=layer_start, layer_end=layer_end, output_dir=output_dir, final_word_stats={}, final_word_layer_stats={}, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate, single_language=True)
#                 # Merge results
#                 for ipa, dim_scores in temp_word_stats.items():
#                     if ipa not in final_word_stats:
#                         final_word_stats[ipa] = {}
#                     for dim, scores in dim_scores.items():
#                         if dim not in final_word_stats[ipa]:
#                             final_word_stats[ipa][dim] = []
#                         final_word_stats[ipa][dim].extend(scores)
                
#                 for layer, layer_stats in temp_word_layer_stats.items():
#                     if layer not in final_word_layer_stats:
#                         final_word_layer_stats[layer] = {}
#                     for ipa, dim_scores in layer_stats.items():
#                         if ipa not in final_word_layer_stats[layer]:
#                             final_word_layer_stats[layer][ipa] = {}
#                         for dim, scores in dim_scores.items():
#                             if dim not in final_word_layer_stats[layer][ipa]:
#                                 final_word_layer_stats[layer][ipa][dim] = []
#                             final_word_layer_stats[layer][ipa][dim].extend(scores)
#                 file_path = "src/analysis/heatmap/results"
            
#             # Compute mean scores after all languages are processed
#             final_word_stats = compute_mean_score(final_word_stats)
#             final_word_layer_stats = compute_mean_score_by_layer(final_word_layer_stats)
#             lang = "Natural"
#             file_name = f"np_{data_type}_{lang}_{COMPUTE_RULE}_check_model_response_{CHECK_MODEL_RESPONSE}_{layer_start}_{layer_end}_processed_words_{processed_words}.pkl"
#             save_file(final_word_stats, os.path.join(file_path, file_name), final_word_layer_stats)
#             plot_ranked_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
#             plot_sampled_word_heatmap(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, sampling_rate=sampling_rate)
#             USE_SOFTMAX = True
#             plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)
#             USE_SOFTMAX = False
#             plot_by_stats_with_ipa_wise(final_word_stats, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, use_softmax=USE_SOFTMAX, sampling_rate=sampling_rate)