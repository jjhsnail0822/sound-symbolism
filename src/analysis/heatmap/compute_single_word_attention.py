# python src/analysis/heatmap/compute_single_word_attention.py

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
data_type = "ipa"
lang = "art"
constructed = True
word = "fah-sho"
layer_start = 0
layer_end = 8
CHECK_MODEL_RESPONSE = True
COMPUTE_RULE = "fraction"
if constructed or (lang == "art") or (lang == "con"):
    lang = "art"
    data_path = "data/processed/art/semantic_dimension/semantic_dimension_binary_gt.json"
    output_dir = f"results/experiments/understanding/attention_heatmap/con/semantic_dimension/{data_type}/{lang}/generation_attention"
else:
    data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"
    output_dir = f"results/experiments/understanding/attention_heatmap/nat/semantic_dimension/{data_type}/{lang}/generation_attention"

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
    ("abrupt", "continuous"), ("continuous", "abrupt"),
    ("active", "passive"), ("passive", "active"),
    ("beautiful", "ugly"), ("ugly", "beautiful"),
    ("big", "small"), ("small", "big"),
    ("dangerous", "safe"), ("safe", "dangerous"),
    ("exciting", "calming"), ("calming", "exciting"),
    ("fast", "slow"), ("slow", "fast"),
    ("good", "bad"), ("bad", "good"),
    ("happy", "sad"), ("sad", "happy"),
    ("hard", "soft"), ("soft", "hard"),
    ("harsh", "mellow"), ("mellow", "harsh"),
    ("heavy", "light"), ("light", "heavy"),
    ("inhibited", "free"), ("free", "inhibited"),
    ("interesting", "uninteresting"), ("uninteresting", "interesting"),
    ("masculine", "feminine"), ("feminine", "masculine"),
    ("orginary", "unique"), ("unique", "orginary"),
    ("pleasant", "unpleasant"), ("unpleasant", "pleasant"),
    ("realistic", "fantastical"), ("fantastical", "realistic"),
    ("rugged", "delicate"), ("delicate", "rugged"),
    ("sharp", "round"), ("round", "sharp"),
    ("simple", "complex"), ("complex", "simple"),
    ("solid", "nonsolid"), ("nonsolid", "solid"),
    ("strong", "weak"), ("weak", "strong"),
    ("structured", "disorganized"), ("disorganized", "structured"),
    ("tense", "relaxed"), ("relaxed", "tense"),
]
ipa_symbols = [
    'a', 'ɑ', 'æ', 'ɐ', 'ə', 'ɚ', 'ɝ', 'ɛ', 'ɜ', 'e', 'ɪ', 'i', 'ɨ', 'ɯ', 'o', 'ɔ', 'ʊ', 'u', 'ʌ', 'ʉ',
    'b', 'β', 'c', 'ç', 'd', 'ð', 'f', 'ɡ', 'ɣ', 'h', 'ɦ', 'j', 'k', 'l', 'ɭ', 'ʟ', 'm', 'ɱ', 'n', 'ŋ',
    'ɲ', 'p', 'ɸ', 'q', 'r', 'ɾ', 'ɹ', 'ʁ', 's', 'ʃ', 't', 'θ', 'v', 'w', 'x', 'χ', 'z', 'ʒ', 'ʔ', 'ʕ',
    'ʡ', 'ʢ', 'ʘ', 'ǀ', 'ǃ', 'ǂ', 'ǁ', 'ɓ', 'ɗ', 'ʄ', 'ɠ', 'ʛ', 'ɦ', 'ʍ', 'ɥ', 'ʜ', 'ʢ', 'ʎ', 'ʟ',
    'ɺ', 'ɻ', 'ɽ', 'ʀ', 'ʂ', 'ʈ', 'ʋ', 'ʐ', 'ʑ', 'ʝ', 'ʞ', 'ʟ', 'ʠ', 'ʡ', 'ʢ', 'ʣ', 'ʤ', 'ʥ', 'ʦ',
    'ʧ', 'ʨ', 'ʩ', 'ʪ', 'ʫ', 'ʬ', 'ʭ', 'ʮ', 'ʯ',
    'ɴ', 'ɕ', 'd͡ʑ', 't͡ɕ', 'ʑ', 'ɰ', 'ã', 'õ', 'ɯ̃', 'ĩ', 'ẽ', 'ɯː', 'aː', 'oː', 'iː', 'eː'
]
vowels = [
    'i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ɛ', 'œ', 'æ', 'a', 'ɶ',  # Front vowels
    'ɨ', 'ʉ', 'ɯ', 'u', 'ɤ', 'o', 'ɜ', 'ɞ', 'ʌ', 'ɔ', 'ɑ', 'ɒ'  # Central/Back vowels
]
consonants = [
    'p', 'b', 'ɸ', 'β', 'm', 'ɱ', # Bilabial
    'f', 'v', # Labiodental
    'θ', 'ð', # Dental
    't', 'd', 's', 'z', 'n', 'r', 'ɾ', 'ɹ', 'l', 'ɬ', 'ɮ', # Alveolar
    'ʃ', 'ʒ', 'ɻ', # Post-alveolar
    'ʈ', 'ɖ', 'ʂ', 'ʐ', 'ɳ', 'ɽ', 'ɭ', # Retroflex
    'c', 'ɟ', 'ç', 'ʝ', 'ɲ', 'j', 'ʎ', # Palatal
    'k', 'ɡ', 'x', 'ɣ', 'ŋ', 'ɰ', 'ʟ', # Velar
    'q', 'ɢ', 'χ', 'ʁ', 'ɴ', # Uvular
    'ħ', 'ʕ', # Pharyngeal
    'h', 'ɦ', 'ʔ' # Glottal
]

def clean_token(token:str) -> str:
    return re.sub(r'^[ĠĊ\[\],.:;!?\n\r\t]+|[ĠĊ\[\],.:;!?\n\r\t]+$', '', token)

def show_arguments(model_name:str=model_path, data_type:str=data_type, lang:str=lang, layer_start:int=layer_start, layer_end:int=layer_end, constructed:bool=constructed, word:str=word, check_model_response:bool=CHECK_MODEL_RESPONSE, compute_rule:str=COMPUTE_RULE):
    print(f"Model: {model_name}")
    print(f"Data type: {data_type}")
    print(f"Language: {lang}")
    print(f"Layer start: {layer_start}")
    print(f"Layer end: {layer_end}")
    print(f"Constructed: {constructed}")
    print(f"Word: {word}")
    print(f"Check model response: {check_model_response}")
    print(f"Compute rule: {compute_rule}")


def model_guessed_incorrectly(response, dim1, dim2, answer):
    print(f"Response: {response}, Dim1: {dim1}, Dim2: {dim2}, Answer: {answer}")
    # breakpoint()
    if dim1 == answer and response == "1":
        return False
    elif dim2 == answer and response == "2":
        return False
    return True

def find_basic_info(word=word, output_dir=output_dir, dim_pairs=dim_pairs):
    for dim1, dim2 in dim_pairs:
        data_dir = os.path.join(output_dir, f"{dim1}_{dim2}", f"{word}_{dim1}_{dim2}.pkl")
        alt_dir = os.path.join(output_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl")
        if not os.path.exists(data_dir) or not os.path.exists(alt_dir):
            continue
        else:
            print(f"Word: {word}, Dim1: {dim1}, Dim2: {dim2}")
            alt = pkl.load(open(alt_dir, "rb"))
            ipa_tokens = alt["ipa_tokens"]
            converted_ipa_tokens = convert_ipa_tokens_to_ipa_string_per_token(ipa_tokens)
            ipa_list = [clean_token(token) for token in converted_ipa_tokens]
            word_stats = {ipa:{} for ipa in ipa_list}
            word_stats = {k: v for k, v in word_stats.items() if k != ""} # Remove empty text
            break
    if ipa_list is None:
        raise ValueError(f"No data found for word: {word}")
    return ipa_list, word_stats

def plot_sampled_word_heatmap(word_stats, word, data_type, start_layer, end_layer, lang, save_path=None, suffix:str=None, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE, dim_pairs=dim_pairs):
    if save_path is None:
        save_path = 'results/plots/attention/sampled_words/'
    os.makedirs(save_path, exist_ok=True)
    ipa_list = list(word_stats.keys())
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
    fig, ax = plt.subplots(figsize=(max(10, len(ipa_list)*0.3), max(8, len(semdim_list)*0.3)))
    sns.heatmap(matrix, ax=ax, cmap='YlGnBu', cbar=True,
                xticklabels=ipa_list, yticklabels=semdim_list, linewidths=0.2, linecolor='gray', square=False)
    for i in range(2, len(semdim_list), 2):
        ax.axhline(i, color='black', linewidth=2)
    ax.set_xlabel('IPA Symbol', fontsize=12)
    ax.set_ylabel('Semantic Dimension', fontsize=12)
    title = f'Sampled word: {word} | lang: {lang} | rule: {compute_rule} | L{start_layer}-L{end_layer} | check_model_response: {check_model_response}\nIPA-Semantic Dimension Attention Heatmap'
    ax.set_title(title, fontsize=14, pad=15)
    plt.setp(ax.get_xticklabels(), ha='right')
    plt.tight_layout()
    clean_word = re.sub(r'[^\w\-]', '_', word)
    file_name = f"sampled_word_{clean_word}_{lang}_{data_type}_generation_attention_L{start_layer}_L{end_layer}"
    if compute_rule is not None:
        file_name += f"_rule-{compute_rule}"
    if check_model_response is not None:
        file_name += f"_check-{check_model_response}"
    if suffix:
        file_name += suffix
    file_name += ".png"
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Sampled word heatmap saved to {file_path}")
    plt.close()

def get_ipa_runs(ipa_list):
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

def get_data(data, alt):
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

def compute_ipa_semdim_score_with_vanilla_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats):
    # breakpoint()
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
                        # breakpoint()
                        continue
                        raise ValueError(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                    for d_idx in dim_range:
                        if d_idx < attn_layers[layer].shape[2] and ipa_idx < attn_layers[layer].shape[3]:
                            v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                            all_values.append(v)
            mean_val = float(np.mean(all_values)) if all_values else np.nan
            word_stats[ipa][dim] = mean_val
    return word_stats

def compute_audio_semdim_score_with_vanilla_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats):
    ipa_runs = get_ipa_runs(ipa_list)
    for ipa, start_idx, end_idx in ipa_runs:
        if ipa == "":
            continue
        for dim, dim_range in [(dim1, dim1_range), (dim2, dim2_range)]:
            all_values = []
            for layer in range(start_layer, min(end_layer+1, n_layer)):
                for head in range(n_head):
                    layer_len = attn_layers[layer].shape[2]
                    if layer_len != wlen+d1len+d2len:
                        continue
                        raise ValueError(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                    for d_idx in dim_range:
                        for ipa_idx in range(start_idx, end_idx+1):
                            if d_idx < attn_layers[layer].shape[2] and ipa_idx < attn_layers[layer].shape[3]:
                                v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                                all_values.append(v)
            mean_val = float(np.mean(all_values)) if all_values else np.nan
            word_stats[ipa][dim] = mean_val
    return word_stats

def compute_ipa_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats):
    for ipa_idx, ipa in enumerate(ipa_list):
        if ipa == "":
            continue
        dim1_values = []
        dim2_values = []
        layer = start_layer
        attn = attn_layers[layer]
        for layer in range(start_layer, min(end_layer+1, n_layer)):
            for head in range(n_head):
                layer_len = attn_layers[layer].shape[2]
                if layer_len != wlen+d1len+d2len:
                    print(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                    continue
                for d_idx in dim1_range:
                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                        v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                        dim1_values.append(v)
                for d_idx in dim2_range:
                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                        v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                        dim2_values.append(v)
        if len(dim1_values) > 0 and len(dim2_values) > 0:
            mean_dim1 = sum(dim1_values) / len(dim1_values)
            mean_dim2 = sum(dim2_values) / len(dim2_values)
            word_stats[ipa][dim1] = mean_dim1
            word_stats[ipa][dim2] = mean_dim2
    return word_stats

def compute_audio_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats):
    ipa_runs = get_ipa_runs(ipa_list)
    for ipa, start_idx, end_idx in ipa_runs:
        if ipa == "":
            continue
        dim1_values = []
        dim2_values = []
        layer = start_layer
        attn = attn_layers[layer]
        for layer in range(start_layer, min(end_layer+1, n_layer)):
            for head in range(n_head):
                layer_len = attn_layers[layer].shape[2]
                if layer_len != wlen+d1len+d2len:
                    print(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                    continue
                sum_dim1 = 0.0
                for d_idx in dim1_range:
                    for ipa_idx in range(start_idx, end_idx+1):
                        if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                            sum_dim1 += attn_layers[layer][0, head, d_idx, ipa_idx].item()
                # dim1_values.append(sum_dim1)
                sum_dim2 = 0.0
                for d_idx in dim2_range:
                    for ipa_idx in range(start_idx, end_idx+1):
                        if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                            sum_dim2 += attn_layers[layer][0, head, d_idx, ipa_idx].item()
                denom = sum_dim1 + sum_dim2
                if denom == 0:
                    frac_dim1, frac_dim2 = 0.0, 0.0
                else:
                    frac_dim1 = sum_dim1 / denom
                    frac_dim2 = sum_dim2 / denom
                dim1_values.append(frac_dim1)
                dim2_values.append(frac_dim2)
        if len(dim1_values) > 0 and len(dim2_values) > 0:
            mean_dim1 = sum(dim1_values) / len(dim1_values)
            mean_dim2 = sum(dim2_values) / len(dim2_values)
            word_stats[ipa][dim1] = mean_dim1
            word_stats[ipa][dim2] = mean_dim2
    return word_stats

def convert_ipa_tokens_to_ipa_string_per_token(ipa_tokens, processor=processor, ipa_symbols=ipa_symbols):
    converted_ipa_tokens = []
    token_to_symbol_map = []
    i = 0
    symbol_for_token = [None] * len(ipa_tokens)
    while i < len(ipa_tokens):
        found = False
        for j in range(min(len(ipa_tokens), i+4), i, -1):
            tmp_token = ipa_tokens[i:j]
            tmp_converted_str = processor.tokenizer.convert_tokens_to_string(tmp_token).strip()
            if tmp_converted_str in ipa_symbols:
                for k in range(i, j):
                    symbol_for_token[k] = tmp_converted_str
                i = j
                found = True
                break
        if not found:
            tmp_converted_str = processor.tokenizer.convert_tokens_to_string([ipa_tokens[i]]).strip()
            symbol_for_token[i] = tmp_converted_str
            i += 1
    return symbol_for_token

def compute_single_word_attention_score(
        word:str=word,
        data_type:str=data_type,
        lang:str=lang,
        start_layer:int=layer_start,
        end_layer:int=layer_end,
        dim_pairs:list=dim_pairs,
        data_path:str=data_path,
        output_dir:str=output_dir,
        ipa_list:list=None,
        word_stats:dict=None,
        check_model_response:bool=CHECK_MODEL_RESPONSE,
        compute_rule:str=COMPUTE_RULE,
        model_path:str=model_path,
    ) -> dict:
    ipa_list, word_stats = find_basic_info(word=word, output_dir=output_dir, dim_pairs=dim_pairs)
    print(f"Word stats at start : {word_stats}")

    for dim1, dim2 in dim_pairs:
        data_dir = os.path.join(output_dir, f"{dim1}_{dim2}", f"{word}_{dim1}_{dim2}.pkl")
        alt_dir = os.path.join(output_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl")
        if not os.path.exists(data_dir) or not os.path.exists(alt_dir):
            continue
        print(f"Found data for {word} {dim1}-{dim2}")
        data = pkl.load(open(data_dir, "rb"))
        alt = pkl.load(open(alt_dir, "rb"))
        attention_matrices, relevant_indices, dim1, dim2, answer, word_tokens, option_tokens, tokens, ipa_tokens, response, input_word, target_indices, wlen, d1len, d2len, dim1_range, dim2_range = get_data(data, alt)
        if check_model_response and model_guessed_incorrectly(response, dim1, dim2, answer):
            continue
            
        cleaned_ipa_tokens = [clean_token(token) for token in ipa_tokens]
        converted_ipa_tokens = convert_ipa_tokens_to_ipa_string_per_token(ipa_tokens)
        ipa_list = converted_ipa_tokens
        attn_layers = attention_matrices[0]
        n_layer = len(attn_layers)
        n_head = attn_layers[0].shape[1]
        if data_type == "ipa":
            if compute_rule == "vanilla":
                # word_stats = compute_ipa_semdim_score_with_vanilla_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
                word_stats = compute_audio_semdim_score_with_vanilla_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
            elif compute_rule == "fraction":
                # word_stats = compute_ipa_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
                word_stats = compute_audio_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
        elif data_type == "audio":
            if compute_rule == "vanilla":
                word_stats = compute_audio_semdim_score_with_vanilla_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
            elif compute_rule == "fraction":
                word_stats = compute_audio_semdim_score_with_fraction_rule(ipa_list, dim1, dim2, dim1_range, dim2_range, start_layer, end_layer, n_layer, n_head, wlen, d1len, d2len, attn_layers, word_stats)
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    print(word_stats)
    return word_stats

show_arguments()
word_stats = compute_single_word_attention_score()
plot_sampled_word_heatmap(word_stats, word, data_type, layer_start, layer_end, lang, compute_rule=COMPUTE_RULE, check_model_response=CHECK_MODEL_RESPONSE)