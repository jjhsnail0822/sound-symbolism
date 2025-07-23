import json
import os
import re
import gc
from typing import Union, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import argparse
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# python src/analysis/heatmap/compute_attention_score.py --data-type audio --start-layer 18 --end-layer 27 --constructed
# python src/analysis/heatmap/compute_attention_score.py --data-type ipa --start-layer 18 --end-layer 27 --constructed
# python src/analysis/heatmap/compute_attention_score.py --data-type ipa
# ipa_to_feature_map = json.load(open("./data/constructed_words/ipa_to_feature.json"))
# feature_to_score_map = json.load(open("./data/constructed_words/feature_to_score.json"))
data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"

class AttentionScoreCalculator:
    def __init__(
        self,
        model_path: str,
        data_type: str,
        lang: str,
        layer_type: str,
        head: Union[int, str],
        layer: Union[int, str],
        compute_type: str,
        constructed: bool = False,
    ):
        self.model_path = model_path
        self.data_type = data_type
        self.lang = lang
        self.layer_type = layer_type
        self.head = head
        self.layer = layer
        self.compute_type = compute_type
        self.constructed = constructed

        if constructed:
            self.data_path = "data/processed/art/semantic_dimension/semantic_dimension_binary_gt.json"
            self.output_dir = "results/experiments/understanding/attention_heatmap/con"
        else:
            self.data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"
            self.output_dir = "results/experiments/understanding/attention_heatmap/nat"

        self.semantic_dimension_map = [
            "good", "bad", "beautiful", "ugly", "pleasant", "unpleasant", "strong", "weak", "big", "small", 
            "rugged", "delicate", "active", "passive", "fast", "slow", "sharp", "round", "realistic", "fantastical", 
            "structured", "disorganized", "orginary", "unique", "interesting", "uninteresting", "simple", "complex", 
            "abrupt", "continuous", "exciting", "calming", "hard", "soft", "happy", "sad", "harsh", "mellow", 
            "heavy", "light", "inhibited", "free", "masculine", "feminine", "solid", "nonsolid", "tense", "relaxed", 
            "dangerous", "safe"
        ]
        self.dim_pairs = [
            ("good", "bad"), ("beautiful", "ugly"), ("pleasant", "unpleasant"), ("strong", "weak"),
            ("big", "small"), ("rugged", "delicate"), ("active", "passive"), ("fast", "slow"),
            ("sharp", "round"), ("realistic", "fantastical"), ("structured", "disorganized"), ("orginary", "unique"),
            ("interesting", "uninteresting"), ("simple", "complex"), ("abrupt", "continuous"), ("exciting", "calming"),
            ("hard", "soft"), ("happy", "sad"), ("harsh", "mellow"), ("heavy", "light"),
            ("inhibited", "free"), ("masculine", "feminine"), ("solid", "nonsolid"), ("tense", "relaxed"),
            ("dangerous", "safe")
        ]
        
        self.ipa_symbols = [
            'a', 'ɑ', 'æ', 'ɐ', 'ə', 'ɚ', 'ɝ', 'ɛ', 'ɜ', 'e', 'ɪ', 'i', 'ɨ', 'ɯ', 'o', 'ɔ', 'ʊ', 'u', 'ʌ', 'ʉ',
            'b', 'β', 'c', 'ç', 'd', 'ð', 'f', 'ɡ', 'ɣ', 'h', 'ɦ', 'j', 'k', 'l', 'ɭ', 'ʟ', 'm', 'ɱ', 'n', 'ŋ',
            'ɲ', 'p', 'ɸ', 'q', 'r', 'ɾ', 'ɹ', 'ʁ', 's', 'ʃ', 't', 'θ', 'v', 'w', 'x', 'χ', 'z', 'ʒ', 'ʔ', 'ʕ',
            'ʡ', 'ʢ', 'ʘ', 'ǀ', 'ǃ', 'ǂ', 'ǁ', 'ɓ', 'ɗ', 'ʄ', 'ɠ', 'ʛ', 'ɦ', 'ʍ', 'ɥ', 'ʜ', 'ʢ', 'ʎ', 'ʟ',
            'ɺ', 'ɻ', 'ɽ', 'ʀ', 'ʂ', 'ʈ', 'ʋ', 'ʐ', 'ʑ', 'ʝ', 'ʞ', 'ʟ', 'ʠ', 'ʡ', 'ʢ', 'ʣ', 'ʤ', 'ʥ', 'ʦ',
            'ʧ', 'ʨ', 'ʩ', 'ʪ', 'ʫ', 'ʬ', 'ʭ', 'ʮ', 'ʯ',
            'ɴ', 'ɕ', 'd͡ʑ', 't͡ɕ', 'ʑ', 'ɰ', 'ã', 'õ', 'ɯ̃', 'ĩ', 'ẽ', 'ɯː', 'aː', 'oː', 'iː', 'eː'
        ]
        
        # Define IPA sorting order: vowels first, then consonants by place of articulation
        # Vowels (front to back, high to low)
        self.vowels = [
            'i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ɛ', 'œ', 'æ', 'a', 'ɶ',  # Front vowels
            'ɨ', 'ʉ', 'ɯ', 'u', 'ɤ', 'o', 'ɜ', 'ɞ', 'ʌ', 'ɔ', 'ɑ', 'ɒ'  # Central/Back vowels
        ]
        
        # Consonants by place of articulation (bilabial to glottal)
        self.consonants = [
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
    
    def _clean_token(self, token:str) -> str:
        return re.sub(r'^[ĠĊ\[\],.:;!?\n\r\t]+|[ĠĊ\[\],.:;!?\n\r\t]+$', '', token)
    
    def extract_ipa_tokens_from_word(self, tokens:list[str]) -> list[str]:
        ipa_tokens = []
        for token in tokens:
            clean_token = self._clean_token(token)
            if not clean_token or clean_token.strip() == '':
                continue
            if clean_token in self.ipa_symbols:
                ipa_tokens.append(clean_token)
        return ipa_tokens
    
    def extract_ipa_from_tokens(self, tokens:list[str]) -> list[str]:
        ipa_tokens = []
        for token in tokens:
            clean_token = self._clean_token(token)
            if clean_token and clean_token in self.ipa_symbols:
                ipa_tokens.append(clean_token)
        return ipa_tokens
    
    def load_matrix(self, data_type: str, word_tokens: str, dimension1: str, dimension2: str, lang: str) -> tuple[dict, str, str, str, list[str], list[str]]:
        file_path = os.path.join(self.output_dir, "semantic_dimension", data_type, lang, "generation_attention",
            f"{word_tokens}_{dimension1}_{dimension2}_generation_analysis.pkl")

        with open(file_path, "rb") as f:
            data = pkl.load(f)
        return data["attention_matrix"], dimension1, dimension2, data.get("answer"), word_tokens, [dimension1, dimension2]
    
    def extract_ipa_attention_scores(self, attention_matrix, tokens, relevant_indices, dimension1, dimension2, answer=None) -> dict:
        if not isinstance(attention_matrix, torch.Tensor):
            attention_matrix = torch.tensor(attention_matrix)
        dim1_indices = []
        dim2_indices = []
        for i, token in enumerate(tokens):
            clean_token = self._clean_token(token)
            if clean_token == dimension1 or dimension1 in clean_token:
                dim1_indices.append(i)
            elif clean_token == dimension2 or dimension2 in clean_token:
                dim2_indices.append(i)
        ipa_dim1_scores = {}
        ipa_dim2_scores = {}
        ipa_list = [self._clean_token(token) for token in tokens]
        if self.data_type == "audio":
            ipa_runs = self.get_ipa_runs(ipa_list)
            for ipa, start_idx, end_idx in ipa_runs:
                if not ipa or ipa.strip() == '' or ipa not in self.ipa_symbols:
                    continue
                dim1_score = 0.0
                dim2_score = 0.0
                valid_dim1_pairs = 0
                valid_dim2_pairs = 0
                for dim_idx in dim1_indices:
                    for ipa_idx in range(start_idx, end_idx+1):
                        if dim_idx < attention_matrix.shape[0] and ipa_idx < attention_matrix.shape[1]:
                            score = attention_matrix[ipa_idx, dim_idx].item()
                            dim1_score += score
                            valid_dim1_pairs += 1
                            if ipa_idx < attention_matrix.shape[0] and dim_idx < attention_matrix.shape[1]:
                                reverse_score = attention_matrix[dim_idx, ipa_idx].item()
                                dim1_score += reverse_score
                                valid_dim1_pairs += 1
                for dim_idx in dim2_indices:
                    for ipa_idx in range(start_idx, end_idx+1):
                        if dim_idx < attention_matrix.shape[0] and ipa_idx < attention_matrix.shape[1]:
                            score = attention_matrix[ipa_idx, dim_idx].item()
                            dim2_score += score
                            valid_dim2_pairs += 1
                            if ipa_idx < attention_matrix.shape[0] and dim_idx < attention_matrix.shape[1]:
                                reverse_score = attention_matrix[dim_idx, ipa_idx].item()
                                dim2_score += reverse_score
                                valid_dim2_pairs += 1
                if valid_dim1_pairs > 0:
                    avg_dim1_score = dim1_score / valid_dim1_pairs
                    if ipa not in ipa_dim1_scores:
                        ipa_dim1_scores[ipa] = []
                    ipa_dim1_scores[ipa].append(avg_dim1_score)
                if valid_dim2_pairs > 0:
                    avg_dim2_score = dim2_score / valid_dim2_pairs
                    if ipa not in ipa_dim2_scores:
                        ipa_dim2_scores[ipa] = []
                    ipa_dim2_scores[ipa].append(avg_dim2_score)
        else:
            for i, token in enumerate(tokens):
                clean_token = self._clean_token(token)
                if not clean_token or clean_token.strip() == '' or clean_token not in self.ipa_symbols:
                    continue
                dim1_score = 0.0
                dim2_score = 0.0
                valid_dim1_pairs = 0
                valid_dim2_pairs = 0
                for dim_idx in dim1_indices:
                    if dim_idx < attention_matrix.shape[0] and i < attention_matrix.shape[1]:
                        score = attention_matrix[i, dim_idx].item()
                        dim1_score += score
                        valid_dim1_pairs += 1
                        if i < attention_matrix.shape[0] and dim_idx < attention_matrix.shape[1]:
                            reverse_score = attention_matrix[dim_idx, i].item()
                            dim1_score += reverse_score
                            valid_dim1_pairs += 1
                for dim_idx in dim2_indices:
                    if dim_idx < attention_matrix.shape[0] and i < attention_matrix.shape[1]:
                        score = attention_matrix[i, dim_idx].item()
                        dim2_score += score
                        valid_dim2_pairs += 1
                        if i < attention_matrix.shape[0] and dim_idx < attention_matrix.shape[1]:
                            reverse_score = attention_matrix[dim_idx, i].item()
                            dim2_score += reverse_score
                            valid_dim2_pairs += 1
                if valid_dim1_pairs > 0:
                    avg_dim1_score = dim1_score / valid_dim1_pairs
                    if clean_token not in ipa_dim1_scores:
                        ipa_dim1_scores[clean_token] = []
                    ipa_dim1_scores[clean_token].append(avg_dim1_score)
                if valid_dim2_pairs > 0:
                    avg_dim2_score = dim2_score / valid_dim2_pairs
                    if clean_token not in ipa_dim2_scores:
                        ipa_dim2_scores[clean_token] = []
                    ipa_dim2_scores[clean_token].append(avg_dim2_score)
        if answer is not None:
            if answer == dimension1:
                return {'dim1_scores': ipa_dim1_scores, 'dimension1': dimension1, 'dimension2': dimension2}
            elif answer == dimension2:
                return {'dim2_scores': ipa_dim2_scores, 'dimension1': dimension1, 'dimension2': dimension2}
            else:
                return {'dim1_scores': {}, 'dim2_scores': {}, 'dimension1': dimension1, 'dimension2': dimension2}
        return {'dim1_scores': ipa_dim1_scores, 'dim2_scores': ipa_dim2_scores, 'dimension1': dimension1, 'dimension2': dimension2}
    
    def aggregate_scores_across_files(self, data_type: str, lang: str) -> dict:
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        analysis_dir = os.path.join(base_dir, "generation_attention")
        all_ipa_semdim_scores = {}
        
        for filename in os.listdir(analysis_dir):
            if not filename.endswith('.pkl'):
                continue
            file_path = os.path.join(analysis_dir, filename)
            with open(file_path, "rb") as f:
                data:dict = pkl.load(f)
            
            generation_analysis = data["generation_analysis"]
            dimension1 = data["dimension1"]
            dimension2 = data["dimension2"]
            answer = data.get("answer", "")
            input_word = data.get("input_word", "") or generation_analysis.get("input_word", "")
            step_analyses = generation_analysis.get('step_analyses', [])

            if step_analyses and len(step_analyses) > 0:
                step_analysis:dict = step_analyses[0]
                word_dim1_raw_matrix = step_analysis.get('word_dim1_raw_matrix', None)
                word_dim2_raw_matrix = step_analysis.get('word_dim2_raw_matrix', None)
                breakpoint()
                target_dimension = None
                target_matrix = None
                if answer == dimension1 and word_dim1_raw_matrix is not None:
                    target_dimension = dimension1
                    target_matrix = word_dim1_raw_matrix
                elif answer == dimension2 and word_dim2_raw_matrix is not None:
                    target_dimension = dimension2
                    target_matrix = word_dim2_raw_matrix
                
                if target_dimension is not None and target_matrix is not None:
                    avg_target_score = np.mean(target_matrix)
                    if not input_word:
                        input_word = generation_analysis.get("input_word", "")
                    if input_word:
                        ipa_symbols = []
                        tokens = generation_analysis.get("tokens", [])
                        if tokens:
                            ipa_symbols = self.extract_ipa_from_tokens(tokens)
                        else:
                            for ipa_part in input_word.split():
                                clean_ipa = self._clean_token(ipa_part)
                                if clean_ipa and clean_ipa in self.ipa_symbols:
                                    ipa_symbols.append(clean_ipa)
                        
                        if ipa_symbols:
                            for ipa in ipa_symbols:
                                key = (ipa, target_dimension)
                                if key not in all_ipa_semdim_scores:
                                    all_ipa_semdim_scores[key] = []
                                all_ipa_semdim_scores[key].append(avg_target_score)
            attention_matrix = data["attention_matrix"]
            tokens = data.get("tokens", [])
            dimension1 = data["dimension1"]
            dimension2 = data["dimension2"]
            answer = data.get("answer", "")
            relevant_indices = data.get("relevant_indices", None)
            input_word = data.get("input_word", "")
            if not input_word and "word_tokens" in data:
                input_word = data["word_tokens"]
            
            if answer in [dimension1, dimension2]:
                ipa_scores = self.extract_ipa_attention_scores(
                    attention_matrix, tokens, relevant_indices, dimension1, dimension2, answer
                )
                if ipa_scores:
                    if answer == dimension1 and 'dim1_scores' in ipa_scores:
                        target_scores = ipa_scores['dim1_scores']
                    elif answer == dimension2 and 'dim2_scores' in ipa_scores:
                        target_scores = ipa_scores['dim2_scores']
                    else:
                        target_scores = {}
                    for ipa, scores in target_scores.items():
                        key = (ipa, answer)
                        if key not in all_ipa_semdim_scores:
                            all_ipa_semdim_scores[key] = []
                        all_ipa_semdim_scores[key].extend(scores)
        stats = {}
        for (ipa, semdim), scores in all_ipa_semdim_scores.items():
            filtered_scores = [score for score in scores if score > 0.0]

            if filtered_scores:
                arr = np.array(filtered_scores)
                stats.setdefault(semdim, {})[ipa] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr)),
                    'count': int(len(arr)),
                    'q25': float(np.percentile(arr, 25)),
                    'q75': float(np.percentile(arr, 75)),
                }

        all_semdims = sorted(stats.keys())
        all_ipas = sorted(set(ipa for semdim in stats for ipa in stats[semdim]))
        data:list = []
        for ipa in all_ipas:
            row = []
            for semdim in all_semdims:
                if ipa in stats[semdim]:
                    row.append(stats[semdim][ipa]['mean'])
                else:
                    row.append(float('nan'))
            data.append(row)
        
        all_semdims = []
        for d1, d2 in self.dim_pairs:
            if d1 in stats:
                all_semdims.append(d1)
            if d2 in stats:
                all_semdims.append(d2)
        return stats
    
    def aggregate_scores_across_files_v2(self, data_type: str, lang: str, start_layer:int=0, end_layer:int=27):
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        analysis_dir = os.path.join(base_dir, "generation_attention")
        all_scores = {}  # (ipa, semdim, layer, head): [score, ...]
        num_of_files = 0
        for foldername in tqdm(os.listdir(analysis_dir), total=len(os.listdir(analysis_dir)), desc="Processing files"):
            if foldername.endswith('.pkl') or foldername.endswith(".json"):
                continue
            semdim_dir = os.path.join(analysis_dir, foldername)
            for filename in os.listdir(semdim_dir):
                num_of_files += 1
                if num_of_files % 1000 == 0:
                    print(f"Processing {num_of_files:>7,}th file : {filename}")
                data = pkl.load(open(os.path.join(semdim_dir, filename), 'rb'))
                word, dim1, dim2 = filename.rsplit("_", 2)
                if dim2.endswith(".pkl"):
                    dim2 = dim2[:-4]
                alt_data = pkl.load(open(os.path.join(analysis_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl"), 'rb'))
                target_indices = alt_data['generation_analysis']['step_analyses'][0]['target_indices']
                wlen = len(target_indices['word'])
                d1len = len(target_indices['dim1'])
                d2len = len(target_indices['dim2'])
                input_word = alt_data['generation_analysis']['input_word']
                
                tokens = alt_data['generation_analysis'].get("tokens", [])
                if tokens:
                    input_word_list = self.extract_ipa_from_tokens(tokens)
                else:
                    input_word_list = [self._clean_token(ipa) for ipa in input_word.split() if self._clean_token(ipa) in self.ipa_symbols]
                
                if not input_word_list:
                    print(f"Warning: No valid IPA symbols found for word {word}")
                    continue
                
                attention_matrices = data['attention_matrices']
                # Handle different attention matrix formats
                if isinstance(attention_matrices[0], (list, tuple)):
                    # Format: [step][layer][head][N,N]
                    attn_layers = attention_matrices[0]
                    n_layer = len(attn_layers)
                    n_head = len(attn_layers[0]) if attn_layers else 0
                else:
                    # Format: [step][batch][layer][head][N,N] or [step][layer][head][N,N]
                    attn_layers = attention_matrices[0]
                    if attn_layers.ndim == 4:  # [batch][layer][head][N,N]
                        attn_layers = attn_layers[0]  # Remove batch dimension
                    n_layer = attn_layers.shape[0] if attn_layers.ndim >= 3 else 1
                    n_head = attn_layers.shape[1] if attn_layers.ndim >= 3 else 1
                
                word_range = range(0, wlen)
                dim1_range = range(wlen, wlen+d1len)
                dim2_range = range(wlen+d1len, wlen+d1len+d2len)
                for semdim, dim_range in zip([dim1, dim2], [dim1_range, dim2_range]):
                    if self.data_type == "audio":
                        ipa_runs = self.get_ipa_runs(input_word_list)
                        for ipa, start_idx, end_idx in ipa_runs:
                            if ipa_idx >= wlen:
                                continue
                            for layer in range(start_layer, min(end_layer+1, n_layer)):
                                if isinstance(attn_layers, (list, tuple)):
                                    attn:torch.Tensor = attn_layers[layer]
                                else:
                                    attn:torch.Tensor = attn_layers[layer]
                                if attn.ndim == 4:
                                    for head in range(attn.shape[1]):
                                        if len(dim_range) == 0:
                                            continue
                                        valid_scores = []
                                        for d_idx in dim_range:
                                            for ipa_idx in range(start_idx, end_idx+1):
                                                if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                                                    valid_scores.append(attn[0, head, d_idx, ipa_idx].item())
                                                    breakpoint()
                                            score = float(np.mean(valid_scores)) if valid_scores else 0.0
                                            key = (ipa, semdim, layer, head)
                                            all_scores.setdefault(key, []).append(score)
                                # elif attn.ndim == 3:
                                #     for head in range(n_head):
                                #         if len(dim_range) == 0:
                                #             continue
                                #         valid_scores = []
                                #         for d_idx in dim_range:
                                #             for ipa_idx in range(start_idx, end_idx+1):
                                #                 if d_idx < attn.shape[1] and ipa_idx < attn.shape[2]:
                                #                     valid_scores.append(attn[head, d_idx, ipa_idx].item())
                                #             score = float(np.mean(valid_scores)) if valid_scores else 0.0
                                #             key = (ipa, semdim, layer, head)
                                #             all_scores.setdefault(key, []).append(score)
                                # elif attn.ndim == 2:
                                #     if len(dim_range) == 0:
                                #         continue
                                #     valid_scores = []
                                #     for d_idx in dim_range:
                                #         for ipa_idx in range(start_idx, end_idx+1):
                                #             if d_idx < attn.shape[0] and ipa_idx < attn.shape[1]:
                                #                 valid_scores.append(attn[d_idx, ipa_idx].item())
                                #         score = float(np.mean(valid_scores)) if valid_scores else 0.0
                                #         key = (ipa, semdim, layer, 0)
                                #         all_scores.setdefault(key, []).append(score)
                                else:
                                    print(f"[WARN] Unexpected attn shape: {attn.shape}")
                                    breakpoint()
                                    continue
                    else:
                        for ipa_idx, ipa in enumerate(input_word_list):
                            if ipa_idx >= wlen:  # Skip if IPA index exceeds word length
                                continue
                            for layer in range(start_layer, min(end_layer+1, n_layer)):
                                if isinstance(attn_layers, (list, tuple)):
                                    attn = attn_layers[layer]
                                else:
                                    attn = attn_layers[layer]
                                if attn.ndim == 4:
                                    for head in range(attn.shape[1]):
                                        if len(dim_range) == 0:
                                            continue
                                        valid_scores = []
                                        for d_idx in dim_range:
                                            if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                                                valid_scores.append(attn[0, head, d_idx, ipa_idx].item())
                                        score = float(np.mean(valid_scores)) if valid_scores else 0.0
                                        key = (ipa, semdim, layer, head)
                                        all_scores.setdefault(key, []).append(score)
                                # elif attn.ndim == 3:
                                #     for head in range(n_head):
                                #         if len(dim_range) == 0:
                                #             continue
                                #         valid_scores = []
                                #         for d_idx in dim_range:
                                #             if d_idx < attn.shape[1] and ipa_idx < attn.shape[2]:
                                #                 valid_scores.append(attn[head, d_idx, ipa_idx].item())
                                #         score = float(np.mean(valid_scores)) if valid_scores else 0.0
                                #         key = (ipa, semdim, layer, head)
                                #         all_scores.setdefault(key, []).append(score)
                                # elif attn.ndim == 2:
                                #     if len(dim_range) == 0:
                                #         continue
                                #     valid_scores = []
                                #     for d_idx in dim_range:
                                #         if d_idx < attn.shape[0] and ipa_idx < attn.shape[1]:
                                #             valid_scores.append(attn[d_idx, ipa_idx].item())
                                #         score = float(np.mean(valid_scores)) if valid_scores else 0.0
                                #         key = (ipa, semdim, layer, 0)
                                #         all_scores.setdefault(key, []).append(score)
                                else:
                                    print(f"[WARN] Unexpected attn shape: {attn.shape}")
                                    breakpoint()
                                    continue
        
        stats = {}
        for (ipa, semdim, layer, head), scores in all_scores.items():
            arr = np.array(scores)
            stats.setdefault(ipa, {}).setdefault(semdim, {}).setdefault('layerwise', {})[(layer, head)] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
                'count': int(len(arr)),
                'q25': float(np.percentile(arr, 25)),
                'q75': float(np.percentile(arr, 75)),
            }
        for ipa in stats:
            for semdim in stats[ipa]:
                all_means = [v['mean'] for v in stats[ipa][semdim]['layerwise'].values()]
                if all_means:
                    arr = np.array(all_means)
                    stats[ipa][semdim]['all'] = {
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr)),
                        'min': float(np.min(arr)),
                        'max': float(np.max(arr)),
                        'median': float(np.median(arr)),
                        'count': int(len(arr)),
                        'q25': float(np.percentile(arr, 25)),
                        'q75': float(np.percentile(arr, 75)),
                    }
        return stats
    
    def aggregate_scores_across_files_multi(self, data_type: str, langs: list) -> dict:
        all_ipa_semdim_scores = {}
        total_file_count = 0
        for lang in langs:
            result:dict = self.aggregate_scores_across_files(data_type, lang)
            if not result:
                continue
            stats:dict = result
            for semdim, ipa_dict in stats.items():
                for ipa, v in ipa_dict.items():
                    key = (ipa, semdim)
                    if key not in all_ipa_semdim_scores:
                        all_ipa_semdim_scores[key] = []
                    all_ipa_semdim_scores[key].append(v['mean'])
            total_file_count += result['file_count']
        # Aggregate statistics for each (ipa, semdim)
        stats = {}
        for (ipa, semdim), scores in all_ipa_semdim_scores.items():
            filtered_scores = [score for score in scores if score > 0.0]
            if filtered_scores:
                arr = np.array(filtered_scores)
                stats.setdefault(semdim, {})[ipa] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr)),
                    'count': int(len(arr)),
                    'q25': float(np.percentile(arr, 25)),
                    'q75': float(np.percentile(arr, 75)),
                }
        return {
            'ipa_semdim_stats': stats,
            'file_count': total_file_count
        }

    def create_phoneme_semdim_matrix(self, aggregated_scores:dict) -> tuple[np.ndarray, list[str], list[str]]:
        print(aggregated_scores.keys())
        breakpoint()
        ipa_semdim_stats = aggregated_scores
        if not ipa_semdim_stats:
            return None, [], []
        
        all_ipa_symbols = set()
        all_semantic_dimensions = set()
        
        for semdim in ipa_semdim_stats:
            all_semantic_dimensions.add(semdim)
            for ipa in ipa_semdim_stats[semdim]:
                all_ipa_symbols.add(ipa)
        
        ipa_list = sorted(list(all_ipa_symbols))
        semdim_list = sorted(list(all_semantic_dimensions))
        matrix = np.zeros((len(ipa_list), len(semdim_list)))
        
        for i, ipa in enumerate(ipa_list):
            for j, semdim in enumerate(semdim_list):
                if semdim in ipa_semdim_stats and ipa in ipa_semdim_stats[semdim]:
                    matrix[i, j] = ipa_semdim_stats[semdim][ipa]['mean']
        
        return matrix, ipa_list, semdim_list
    
    def plot_ipa_semdim_heatmap(self, ipa_semdim_stats, save_path=None, lang=None, data_type=None) -> None:
        """
        Plot and save a heatmap: X-axis=IPA symbols, Y-axis=semantic dimensions, values=mean attention scores.
        """
        # Create ordered semantic dimension list based on self.dim_pairs
        ordered_semdim_list = []
        for d1, d2 in self.dim_pairs:
            if d1 in ipa_semdim_stats:
                ordered_semdim_list.append(d1)
            if d2 in ipa_semdim_stats:
                ordered_semdim_list.append(d2)
        
        # Collect all unique IPA symbols and sort them by phonetic features
        ipa_set = set()
        for semdim in ipa_semdim_stats:
            ipa_set.update(ipa_semdim_stats[semdim].keys())
        
        # Sort IPA symbols according to our defined order
        sorted_ipa_list = []
        for ipa in self.vowels + self.consonants:
            if ipa in ipa_set:
                sorted_ipa_list.append(ipa)
        
        # Add any remaining IPA symbols that weren't in our predefined lists
        remaining_ipas = [ipa for ipa in sorted(ipa_set) if ipa not in sorted_ipa_list]
        sorted_ipa_list.extend(remaining_ipas)
        
        # Create matrix using ordered semantic dimensions (transposed: rows=semantic dimensions, cols=IPA symbols)
        matrix = []
        for semdim in ordered_semdim_list:
            row = []
            for ipa in sorted_ipa_list:
                if ipa in ipa_semdim_stats[semdim]:
                    row.append(ipa_semdim_stats[semdim][ipa]['mean'])
                else:
                    row.append(float('nan'))
            matrix.append(row)
        matrix = np.array(matrix)
        
        # Plot heatmap (transposed: X-axis=IPA symbols, Y-axis=semantic dimensions)
        fig, ax = plt.subplots(figsize=(max(12, len(sorted_ipa_list)*0.3), max(10, len(ordered_semdim_list)*0.3)))
        
        # Create heatmap
        im = sns.heatmap(matrix, ax=ax, cmap='YlGnBu', cbar=True, 
                        xticklabels=sorted_ipa_list, yticklabels=ordered_semdim_list, 
                        linewidths=0.2, linecolor='gray', square=False)
        
        # Add thick borders to separate dimension pairs
        for i in range(len(ordered_semdim_list)):
            # Add horizontal lines to separate dimension pairs
            if i > 0 and i % 2 == 0:  # Every 2 dimensions (after each pair)
                ax.axhline(y=i, color='black', linewidth=2)
        
        # Add vertical lines to separate vowel and consonant groups
        vowel_count = sum(1 for ipa in sorted_ipa_list if ipa in self.vowels)
        if vowel_count > 0:
            ax.axvline(x=vowel_count, color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        # Add labels for vowel/consonant separation
        if vowel_count > 0:
            ax.text(vowel_count/2, -0.5, 'Vowels', ha='center', va='top', 
                   fontsize=10, fontweight='bold', color='red')
            ax.text(vowel_count + (len(sorted_ipa_list) - vowel_count)/2, -0.5, 'Consonants', 
                   ha='center', va='top', fontsize=10, fontweight='bold', color='red')
        
        ax.set_xlabel('IPA Symbol', fontsize=14)
        ax.set_ylabel('Semantic Dimension', fontsize=14)
        ax.set_title(f'IPA-Semantic Dimension Attention Heatmap ({lang}, {data_type}, generation attention)', fontsize=16, pad=15)
        
        plt.setp(ax.get_xticklabels(), ha='right')
        
        plt.tight_layout()
        if save_path is None:
            save_path = 'results/plots/attention/'
        os.makedirs(save_path, exist_ok=True)
        file_name = f"ipa_semdim_attention_heatmap_{lang}_{data_type}_generation_attention.png"
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"IPA-Semantic Dimension heatmap saved to {file_path}")
        plt.close()
    
    def plot_single_word_attention_heatmap(self, attention_matrix, tokens, word_indices, dim1_indices, dim2_indices, 
                                         dimension1, dimension2, word_tokens, save_path=None, lang=None, 
                                         data_type=None, layer_idx=0, head_idx=0) -> str:
        # Convert to tensor if needed
        if not isinstance(attention_matrix, torch.Tensor):
            attention_matrix = torch.tensor(attention_matrix)
        
        # Get the attention matrix for specified layer and head
        if len(attention_matrix.shape) == 4:  # [layer, head, seq, seq]
            attn = attention_matrix[layer_idx, head_idx]
        else:
            breakpoint()
        # elif len(attention_matrix.shape) == 3:  # [layer, seq, seq]
        #     attn = attention_matrix[layer_idx]
        # else:  # [seq, seq]
        #     attn = attention_matrix
        
        min_word_idx = min(word_indices) if word_indices else 0
        max_dim_idx = max(dim1_indices + dim2_indices) if (dim1_indices or dim2_indices) else attn.shape[0]
        
        start_idx = max(0, min_word_idx - 2)  # Include some context
        end_idx = min(attn.shape[0], max_dim_idx + 3)  # Include some context
        
        region_attn = attn[start_idx:end_idx, start_idx:end_idx].numpy()
        region_tokens = tokens[start_idx:end_idx] if tokens else [f"token_{i}" for i in range(start_idx, end_idx)]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(region_attn, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(len(region_tokens)))
        ax.set_yticks(range(len(region_tokens)))
        ax.set_xticklabels(region_tokens, ha='right', fontsize=8)
        ax.set_yticklabels(region_tokens, fontsize=8)
        
        word_region = [i - start_idx for i in word_indices if start_idx <= i < end_idx]
        dim1_region = [i - start_idx for i in dim1_indices if start_idx <= i < end_idx]
        dim2_region = [i - start_idx for i in dim2_indices if start_idx <= i < end_idx]
        
        for idx in word_region:
            if 0 <= idx < len(region_tokens):
                rect = plt.Rectangle((idx-0.5, -0.5), 1, len(region_tokens), 
                                   linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7)
                ax.add_patch(rect)
        
        for idx in dim1_region:
            if 0 <= idx < len(region_tokens):
                rect = plt.Rectangle((idx-0.5, -0.5), 1, len(region_tokens), 
                                   linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
                ax.add_patch(rect)
        
        for idx in dim2_region:
            if 0 <= idx < len(region_tokens):
                rect = plt.Rectangle((idx-0.5, -0.5), 1, len(region_tokens), 
                                   linewidth=2, edgecolor='green', facecolor='none', alpha=0.7)
                ax.add_patch(rect)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Score', fontsize=12)
        
        title = f'Word Attention Heatmap: {word_tokens}\n{dimension1} vs {dimension2}'
        if len(attention_matrix.shape) >= 3:
            title += f' (Layer {layer_idx}, Head {head_idx})'
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Query Tokens', fontsize=12)
        ax.set_ylabel('Key Tokens', fontsize=12)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='blue', label='Word Tokens'),
            Patch(facecolor='none', edgecolor='red', label=f'{dimension1}'),
            Patch(facecolor='none', edgecolor='green', label=f'{dimension2}')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = 'results/plots/attention/'
        os.makedirs(save_path, exist_ok=True)
        
        clean_word = re.sub(r'[^\w\-]', '_', word_tokens)
        file_name = f"word_attention_heatmap_{clean_word}_{dimension1}_{dimension2}_{lang}_{data_type}_generation_attention_L{layer_idx}H{head_idx}.png"
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Word attention heatmap saved to {file_path}")
        plt.close()
        
        return file_path

    def plot_ipa_semdim_heatmap_with_layers(self, stats:dict, save_path:str, lang:str, data_type:str, start_layer:int, end_layer:int, condition_desc:str="") -> None:
        if not stats or not any(stats.values()):
            print(f"[WARN] No data to plot for lang={lang}, condition={condition_desc}")
            return
        
        # Define semantic dimension pairs in order
        dim_pairs = [
            ("good", "bad"), ("beautiful", "ugly"), ("pleasant", "unpleasant"), ("strong", "weak"),
            ("big", "small"), ("rugged", "delicate"), ("active", "passive"), ("fast", "slow"),
            ("sharp", "round"), ("realistic", "fantastical"), ("structured", "disorganized"), ("ordinary", "unique"),
            ("interesting", "uninteresting"), ("simple", "complex"), ("abrupt", "continuous"), ("exciting", "calming"),
            ("hard", "soft"), ("happy", "sad"), ("harsh", "mellow"), ("heavy", "light"),
            ("inhibited", "free"), ("masculine", "feminine"), ("solid", "nonsolid"), ("tense", "relaxed"),
            ("dangerous", "safe")
        ]
        
        # Create ordered semantic dimension list based on dim_pairs
        ordered_semdim_list = []
        for d1, d2 in dim_pairs:
            if d1 in stats or (isinstance(next(iter(stats.values())), dict) and any(d1 in ipa_stats for ipa_stats in stats.values())):
                ordered_semdim_list.append(d1)
            if d2 in stats or (isinstance(next(iter(stats.values())), dict) and any(d2 in ipa_stats for ipa_stats in stats.values())):
                ordered_semdim_list.append(d2)
        

        
        # Handle different stats structures
        if isinstance(next(iter(stats.values())), dict) and 'all' in next(iter(next(iter(stats.values())).values())):
            # Structure: stats[ipa][semdim]['all']['mean']
            ipa_set = set(stats.keys())
            semdim_set = {semdim for ipa in stats for semdim in stats[ipa]}
            if not semdim_set or not ipa_set:
                print(f"[WARN] No semantic dimensions or IPA symbols to plot for lang={lang}, condition={condition_desc}")
                return
        else:
            # Structure: stats[semdim][ipa] = float
            semdim_set = set(stats.keys())
            ipa_set = {ipa for semdim in stats for ipa in stats[semdim]}
            if not semdim_set or not ipa_set:
                print(f"[WARN] No semantic dimensions or IPA symbols to plot for lang={lang}, condition={condition_desc}")
                return
        
        # Filter ordered_semdim_list to only include dimensions that exist in the data
        ordered_semdim_list = [dim for dim in ordered_semdim_list if dim in semdim_set]
        
        # Sort IPA symbols according to our defined order
        sorted_ipa_list = []
        for ipa in self.vowels + self.consonants:
            if ipa in ipa_set:
                sorted_ipa_list.append(ipa)
        
        # Add any remaining IPA symbols that weren't in our predefined lists
        remaining_ipas = [ipa for ipa in sorted(ipa_set) if ipa not in sorted_ipa_list]
        sorted_ipa_list.extend(remaining_ipas)
        
        # Create matrix using ordered semantic dimensions and sorted IPA symbols
        matrix = np.zeros((len(ordered_semdim_list), len(sorted_ipa_list)))
        for i, semdim in enumerate(ordered_semdim_list):
            for j, ipa in enumerate(sorted_ipa_list):
                if isinstance(next(iter(stats.values())), dict) and 'all' in next(iter(next(iter(stats.values())).values())):
                    # Structure: stats[ipa][semdim]['all']['mean']
                    if ipa in stats and semdim in stats[ipa] and 'all' in stats[ipa][semdim]:
                        matrix[i, j] = stats[ipa][semdim]['all']['mean']
                    else:
                        matrix[i, j] = 0.0
                else:
                    # Structure: stats[semdim][ipa] = float
                    matrix[i, j] = stats[semdim].get(ipa, 0.0)

        fig, ax = plt.subplots(figsize=(max(12, len(sorted_ipa_list)*0.3), max(10, len(ordered_semdim_list)*0.3)))
        im = sns.heatmap(matrix, ax=ax, cmap='YlGnBu', cbar=True, 
                        xticklabels=sorted_ipa_list, yticklabels=ordered_semdim_list, 
                        linewidths=0.2, linecolor='gray', square=False)
        
        # Add thick borders to separate dimension pairs
        for i in range(len(ordered_semdim_list)):
            # Add horizontal lines to separate dimension pairs
            if i > 0 and i % 2 == 0:  # Every 2 dimensions (after each pair)
                ax.axhline(y=i, color='black', linewidth=2)
        
        # Add vertical lines to separate vowel and consonant groups
        vowel_count = sum(1 for ipa in sorted_ipa_list if ipa in self.vowels)
        if vowel_count > 0:
            ax.axvline(x=vowel_count, color='red', linewidth=2, linestyle='--', alpha=0.7)
        
        # Add labels for vowel/consonant separation
        if vowel_count > 0:
            ax.text(vowel_count/2, -0.5, 'Vowels', ha='center', va='top', 
                   fontsize=10, fontweight='bold', color='red')
            ax.text(vowel_count + (len(sorted_ipa_list) - vowel_count)/2, -0.5, 'Consonants', 
                   ha='center', va='top', fontsize=10, fontweight='bold', color='red')
        
        title = f"IPA-Semantic Dimension Attention Heatmap (L{start_layer}-{end_layer})\n{condition_desc} - Language: {lang}"
        ax.set_title(title, fontsize=16, pad=15)
        ax.set_xlabel('IPA Symbol', fontsize=14)
        ax.set_ylabel('Semantic Dimension', fontsize=14)
        plt.setp(ax.get_xticklabels(), ha='right')
        plt.tight_layout()
        if save_path is None:
            save_path = 'results/plots/attention/'
        import os
        os.makedirs(save_path, exist_ok=True)
        file_name = f"ipa_semdim_attention_heatmap_{lang}_{data_type}_generation_attention_L{start_layer}_L{end_layer}.png"
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"IPA-Semantic Dimension heatmap saved to {file_path}")
        plt.close()

    def aggregate_scores_with_response_condition(self, data_type:str, lang:str, start_layer:int=20, end_layer:int=27) -> dict[tuple[str, str, int, int], list[float]]:
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        analysis_dir = os.path.join(base_dir, "generation_attention")
        all_scores:dict[tuple[str, str, int, int], list[float]] = {}  # (ipa, semdim, layer, head): [score, ...]
        num_of_files = 0
        for foldername in tqdm(os.listdir(analysis_dir), total=len(os.listdir(analysis_dir)), desc="Processing files"):
            if foldername.endswith('.pkl') or foldername.endswith(".json"):
                continue
            semdim_dir = os.path.join(analysis_dir, foldername)
            for filename in os.listdir(semdim_dir):
                num_of_files += 1
                print(f"Processing {num_of_files:>7,}th file : {filename}")
                data = pkl.load(open(os.path.join(semdim_dir, filename), 'rb'))
                word, dim1, dim2 = filename.rsplit("_", 2)
                if dim2.endswith(".pkl"):
                    dim2 = dim2[:-4]
                alt_data = pkl.load(open(os.path.join(analysis_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl"), 'rb'))
                
                # Check response condition
                gen_analysis:dict = alt_data.get("generation_analysis", {})
                answer = gen_analysis.get("answer", None)
                response = gen_analysis.get("response", None)
                if not (answer and response):
                    continue
                resp_num = None
                if '1' in response:
                    resp_num = "1"
                elif '2' in response:
                    resp_num = "2"
                if not (
                    (resp_num == "1" and answer == dim1) or
                    (resp_num == "2" and answer == dim2)
                ):
                    continue
                
                target_indices = gen_analysis['step_analyses'][0]['target_indices']
                wlen = len(target_indices['word'])
                d1len = len(target_indices['dim1'])
                d2len = len(target_indices['dim2'])
                input_word = gen_analysis['input_word']
                
                # Extract IPA symbols properly
                tokens = gen_analysis.get("tokens", [])
                if tokens:
                    # Use tokens to extract IPA symbols
                    input_word_list = self.extract_ipa_from_tokens(tokens)
                else:
                    # Fallback to input_word parsing
                    input_word_list = [self._clean_token(ipa) for ipa in input_word.split() if self._clean_token(ipa) in self.ipa_symbols]
                
                # Ensure we have valid IPA symbols
                if not input_word_list:
                    print(f"Warning: No valid IPA symbols found for word {word}")
                    continue
                
                attention_matrices = data['attention_matrices']
                
                # Handle different attention matrix formats
                if isinstance(attention_matrices[0], (list, tuple)):
                    attn_layers = attention_matrices[0]
                    n_layer = len(attn_layers)
                    n_head = len(attn_layers[0]) if attn_layers else 0
                else:
                    attn_layers = attention_matrices[0]
                    if attn_layers.ndim == 4:
                        attn_layers = attn_layers[0]
                    n_layer = attn_layers.shape[0] if attn_layers.ndim >= 3 else 1
                    n_head = attn_layers.shape[1] if attn_layers.ndim >= 3 else 1
                
                word_range = range(0, wlen)
                dim1_range = range(wlen, wlen+d1len)
                dim2_range = range(wlen+d1len, wlen+d1len+d2len)
                
                # Determine which dimension is correct based on answer
                if answer == dim1:
                    correct_dim = dim1
                    correct_range = dim1_range
                    wrong_dim = dim2
                    wrong_range = dim2_range
                else:
                    correct_dim = dim2
                    correct_range = dim2_range
                    wrong_dim = dim1
                    wrong_range = dim1_range
                
                for ipa_idx, ipa in enumerate(input_word_list):
                    if ipa_idx >= wlen:
                        continue
                    for layer in range(start_layer, min(end_layer+1, n_layer)):
                        if isinstance(attn_layers, (list, tuple)):
                            attn:torch.Tensor = attn_layers[layer]
                        else:
                            attn:torch.Tensor = attn_layers[layer]
                        
                        if attn.ndim == 4:
                            for head in range(attn.shape[1]):
                                if len(correct_range) == 0:
                                    continue
                                valid_correct_scores = []
                                valid_wrong_scores = []
                                for d_idx in correct_range:
                                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                                        valid_correct_scores.append(attn[0, head, d_idx, ipa_idx].item())
                                for d_idx in wrong_range:
                                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                                        valid_wrong_scores.append(attn[0, head, d_idx, ipa_idx].item())
                                
                                correct_score = float(np.mean(valid_correct_scores)) if valid_correct_scores else 0.0
                                wrong_score = float(np.mean(valid_wrong_scores)) if valid_wrong_scores else 0.0
                                denom = correct_score + wrong_score
                                score = correct_score / denom if denom > 0 else 0.0
                                key = (ipa, correct_dim, layer, head)
                                all_scores.setdefault(key, []).append(score)
                        # elif attn.ndim == 3:
                        #     for head in range(n_head):
                        #         if len(correct_range) == 0:
                        #             continue
                        #         valid_correct_scores = []
                        #         valid_wrong_scores = []
                        #         for d_idx in correct_range:
                        #             if d_idx < attn.shape[1] and ipa_idx < attn.shape[2]:
                        #                 valid_correct_scores.append(attn[head, d_idx, ipa_idx].item())
                        #         for d_idx in wrong_range:
                        #             if d_idx < attn.shape[1] and ipa_idx < attn.shape[2]:
                        #                 valid_wrong_scores.append(attn[head, d_idx, ipa_idx].item())
                                
                        #         correct_score = float(np.mean(valid_correct_scores)) if valid_correct_scores else 0.0
                        #         wrong_score = float(np.mean(valid_wrong_scores)) if valid_wrong_scores else 0.0
                        #         denom = correct_score + wrong_score
                        #         score = correct_score / denom if denom > 0 else 0.0
                        #         key = (ipa, correct_dim, layer, head)
                        #         all_scores.setdefault(key, []).append(score)
                                
                        # elif attn.ndim == 2:
                        #     if len(correct_range) == 0:
                        #         continue
                        #     valid_correct_scores = []
                        #     valid_wrong_scores = []
                        #     for d_idx in correct_range:
                        #         if d_idx < attn.shape[0] and ipa_idx < attn.shape[1]:
                        #             valid_correct_scores.append(attn[d_idx, ipa_idx].item())
                        #     for d_idx in wrong_range:
                        #         if d_idx < attn.shape[0] and ipa_idx < attn.shape[1]:
                        #             valid_wrong_scores.append(attn[d_idx, ipa_idx].item())
                            
                        #     correct_score = float(np.mean(valid_correct_scores)) if valid_correct_scores else 0.0
                        #     wrong_score = float(np.mean(valid_wrong_scores)) if valid_wrong_scores else 0.0
                        #     denom = correct_score + wrong_score
                        #     score = correct_score / denom if denom > 0 else 0.0
                        #     key = (ipa, correct_dim, layer, 0)
                        #     all_scores.setdefault(key, []).append(score)
                        else:
                            print(f"[WARN] Unexpected attn shape: {attn.shape}")
                            breakpoint()
                            continue
        
        stats = {}
        for (ipa, semdim, layer, head), scores in all_scores.items():
            arr = np.array(scores)
            stats.setdefault(ipa, {}).setdefault(semdim, {}).setdefault('layerwise', {})[(layer, head)] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
                'count': int(len(arr)),
                'q25': float(np.percentile(arr, 25)),
                'q75': float(np.percentile(arr, 75)),
            }
        for ipa in stats:
            for semdim in stats[ipa]:
                all_means = [v['mean'] for v in stats[ipa][semdim]['layerwise'].values()]
                if all_means:
                    arr = np.array(all_means)
                    stats[ipa][semdim]['all'] = {
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr)),
                        'min': float(np.min(arr)),
                        'max': float(np.max(arr)),
                        'median': float(np.median(arr)),
                        'count': int(len(arr)),
                        'q25': float(np.percentile(arr, 25)),
                        'q75': float(np.percentile(arr, 75)),
                    }
        return stats

    def sample_single_word_response_condition(self, data_type:str, lang:str, start_layer:int=20, end_layer:int=27, num_samples:int=5):
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        analysis_dir = os.path.join(base_dir, "generation_attention")
        if not os.path.exists(analysis_dir):
            print(f"Directory not found: {analysis_dir}")
            return None
        file_dim_pairs = []
        all_word_stats = {}
        for foldername in tqdm(os.listdir(analysis_dir), total=len(os.listdir(analysis_dir)), desc="Processing files"):
            if foldername.endswith('.pkl') or foldername.endswith(".json"):
                continue
            semdim_dir = os.path.join(analysis_dir, foldername)
            for filename in os.listdir(semdim_dir):
                if not filename.endswith('.pkl'):
                    continue
                word, dim1, dim2 = filename.rsplit("_", 2)
                if dim2.endswith(".pkl"):
                    dim2 = dim2[:-4]
                gen_analysis_path = os.path.join(analysis_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl")
                if not os.path.exists(gen_analysis_path):
                    continue
                alt_data = pkl.load(open(gen_analysis_path, 'rb'))
                gen_analysis = alt_data.get("generation_analysis", {})
                answer = gen_analysis.get("answer", None)
                response = gen_analysis.get("response", None)
                if not (answer and response):
                    continue
                resp_num = None
                if '1' in response:
                    resp_num = "1"
                elif '2' in response:
                    resp_num = "2"
                if (resp_num == "1" and answer == dim1) or (resp_num == "2" and answer == dim2):
                    file_dim_pairs.append((word, dim1, dim2, filename, semdim_dir, [dim1, dim2]))
        all_words = list(set([f[0] for f in file_dim_pairs]))
        sampled_words = []
        word_idx = 0
        while len(sampled_words) < num_samples and word_idx < len(all_words):
            word = all_words[word_idx]
            word_idx += 1
            word_files = [f for f in file_dim_pairs if f[0] == word]
            if not word_files:
                continue
            _, dim1, dim2, filename, semdim_dir, _ = word_files[0]
            if f"{dim1}_{dim2}" in semdim_dir:
                semdim_dir = semdim_dir.rsplit("/", 1)[0]
                filename = filename[:-4] + "_generation_analysis.pkl"
            else:
                semdim_dir = os.path.join(semdim_dir, f"{dim1}_{dim2}")
            data = pkl.load(open(os.path.join(semdim_dir, filename), 'rb'))
            if "ipa_tokens" not in data.keys():
                print(f"[DEBUG] No ipa tokens in {filename}")
                continue
            else:
                print("Found data with ipa tokens. Continue your job")
            
            alt_data = pkl.load(open(os.path.join(analysis_dir, filename), 'rb'))
            gen_analysis = alt_data.get("generation_analysis", {})
            tokens = gen_analysis.get("tokens", [])
            input_word = gen_analysis.get("input_word", "")
            if data_type == "ipa":
                ipa_list = input_word.split()
            elif data_type == "audio":
                ipa_list = data["ipa_tokens"]
            if not ipa_list:
                print(f"[DEBUG] No valid IPA symbols for word {word}")
                continue
            word_stats = {ipa:{} for ipa in ipa_list}
            word_stats = {k: v for k, v in word_stats.items() if k != ""}
            for dim1, dim2 in tqdm(self.dim_pairs, total=len(self.dim_pairs), desc="Processing dim pairs"):
                attn_file_path = None
                for foldername in os.listdir(analysis_dir):
                    if foldername.endswith('.pkl') or foldername.endswith(".json"):
                        continue
                    semdim_dir = os.path.join(analysis_dir, foldername)
                    candidate = f"{word}_{dim1}_{dim2}.pkl"
                    if candidate in os.listdir(semdim_dir):
                        attn_file_path = os.path.join(semdim_dir, candidate)
                        break
                try:
                    data = pkl.load(open(attn_file_path, 'rb'))
                except Exception as e:
                    print(f"[ERROR] Failed to load data from {attn_file_path}: {e}")
                    breakpoint()
                alt_data = pkl.load(open(os.path.join(analysis_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl"), 'rb'))
                gen_analysis = alt_data.get("generation_analysis", {})
                gen_analysis, target_indices, input_word, wlen, d1len, d2len, response, answer, dim1_range, dim2_range, attention_matrices, tokens = self.get_attention_data(data, alt_data)
                
                first_dim1_from_tokens = tokens[target_indices['dim1'][0]:target_indices['dim1'][-1]+1]
                flag_index = 0
                for i, first_token in enumerate(first_dim1_from_tokens):
                    if first_token == "OPTIONS":
                        flag_index = i
                        break
                dim1_from_tokens = tokens[flag_index+3:target_indices['dim1'][-1]+1]
                dim2_from_tokens = input_word[target_indices['dim2'][0]:target_indices['dim2'][-1]+1]
                cleaned_dim1_from_tokens = [self._clean_token(token) for token in dim1_from_tokens]
                cleaned_dim2_from_tokens = [self._clean_token(token) for token in dim2_from_tokens]
                idx_to_remove_in_dim1 = []
                for i, dim1_token in enumerate(cleaned_dim1_from_tokens):
                    if dim1_token == "":
                        d1len -= 1
                        idx_to_remove_in_dim1.append(i)
                for i in reversed(idx_to_remove_in_dim1):
                    target_indices['dim1'].pop(i)
                try:
                    idx_to_remove_in_dim2 = []
                    for i, dim2_token in enumerate(cleaned_dim2_from_tokens):
                        if dim2_token == "":
                            d2len -= 1
                            idx_to_remove_in_dim2.append(i)
                    for i in reversed(idx_to_remove_in_dim2):
                        target_indices['dim2'].pop(i)
                except Exception as e:
                    print(f"[ERROR] Failed to load data from {attn_file_path}: {e}")
                    breakpoint()
                
                if ('1' in response and answer == dim1):
                    correct_dim = dim1
                    wrong_dim = dim2
                    correct_range = range(wlen, wlen+d1len)
                    wrong_range = range(wlen+d1len, wlen+d1len+d2len)
                elif ('2' in response and answer == dim2):
                    correct_dim = dim2
                    wrong_dim = dim1
                    correct_range = range(wlen+d1len, wlen+d1len+d2len)
                    wrong_range = range(wlen, wlen+d1len)
                else:
                    continue

                if isinstance(attention_matrices[0], (list, tuple)):
                    attn_layers = attention_matrices[0]
                else:
                    attn_layers = attention_matrices[0]
                    if attn_layers.ndim == 4:
                        attn_layers = attn_layers[0]
                n_layer = len(attn_layers)
                layer = start_layer
                attn = attn_layers[layer]
                n_head = attn.shape[1]
                if data_type == "ipa":
                    for ipa_idx, ipa in enumerate(ipa_list):
                        if ipa_idx >= wlen:
                            continue
                        correct_values = []
                        wrong_values = []
                        for layer in range(start_layer, min(end_layer+1, n_layer)):
                            for head in range(n_head):
                                layer_len = attn_layers[layer].shape[2]
                                if layer_len != wlen+d1len+d2len:
                                    print(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                                    continue
                                for d_idx in correct_range:
                                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                                        v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                                        correct_values.append(v)
                                for d_idx in wrong_range:
                                    if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                                        v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                                        wrong_values.append(v)
                        if len(correct_values) > 0 and len(wrong_values) > 0:
                            mean_correct = sum(correct_values) / len(correct_values)
                            mean_wrong = sum(wrong_values) / len(wrong_values)
                            word_stats[ipa][correct_dim] = mean_correct
                            word_stats[ipa][wrong_dim] = mean_wrong
                elif data_type == "audio":
                    ipa_runs = self.get_ipa_runs(ipa_list)
                    for ipa, start_idx, end_idx in ipa_runs:
                        if ipa == "":
                            continue
                        correct_values = []
                        wrong_values = []
                        for layer in range(start_layer, min(end_layer+1, n_layer)):
                            for head in range(n_head):
                                layer_len = attn_layers[layer].shape[2]
                                if layer_len != wlen+d1len+d2len:
                                    print(f"Layer length mismatch: {layer_len} != {wlen+d1len+d2len}")
                                    continue
                                sum_correct = 0.0
                                for d_idx in correct_range:
                                    for ipa_idx in range(start_idx, end_idx+1):
                                        if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                                            sum_correct += attn_layers[layer][0, head, d_idx, ipa_idx].item()
                                correct_values.append(sum_correct)
                                sum_wrong = 0.0
                                for d_idx in wrong_range:
                                    for ipa_idx in range(start_idx, end_idx+1):
                                        if d_idx < attn.shape[2] and ipa_idx < attn.shape[3]:
                                            sum_wrong += attn_layers[layer][0, head, d_idx, ipa_idx].item()
                                wrong_values.append(sum_wrong)

                        if len(correct_values) > 0 and len(wrong_values) > 0:
                            mean_correct = sum(correct_values) / len(correct_values)
                            mean_wrong = sum(wrong_values) / len(wrong_values)
                            word_stats[ipa][correct_dim] = mean_correct
                            word_stats[ipa][wrong_dim] = mean_wrong
            # Only add word if enough valid dims
            valid_dims = set()
            for ipa in word_stats:
                valid_dims.update(word_stats[ipa].keys())
            if len(valid_dims) > 0:
                sampled_words.append(word)
                all_word_stats[word] = word_stats
        return all_word_stats

    def sample_single_word_response_condition_v2(self, data_type:str, lang:str, start_layer:int=20, end_layer:int=27, num_samples:int=5):
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        analysis_dir = os.path.join(base_dir, "generation_attention")
        all_word_stats = {}
        if not os.path.exists(analysis_dir):
            print(f"Directory not found: {analysis_dir}")
            return None
        file_dim_pairs = []
        for foldername in os.listdir(analysis_dir):
            if foldername.endswith('.pkl') or foldername.endswith(".json"):
                continue
            semdim_dir = os.path.join(analysis_dir, foldername)
            for filename in os.listdir(semdim_dir):
                if not filename.endswith('.pkl'):
                    continue
                word, dim1, dim2 = filename.rsplit("_", 2)
                if dim2.endswith(".pkl"):
                    dim2 = dim2[:-4]
                gen_analysis_path = os.path.join(analysis_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl")
                if not os.path.exists(gen_analysis_path):
                    continue
                file_dim_pairs.append((word, dim1, dim2, filename, semdim_dir, [dim1, dim2]))
        all_words = list(set([f[0] for f in file_dim_pairs]))
        sampled_words = []
        word_idx = 0
        while len(sampled_words) < num_samples and word_idx < len(all_words):
            word = all_words[word_idx]
            word_idx += 1
            print(f"\n[DEBUG] Processing word: {word}")
            word_files = [f for f in file_dim_pairs if f[0] == word]
            if not word_files:
                continue
            _, dim1, dim2, filename, semdim_dir, _ = word_files[0]
            if f"{dim1}_{dim2}" in semdim_dir:
                semdim_dir = semdim_dir.rsplit("/", 1)[0]
                filename = filename[:-4] + "_generation_analysis.pkl"
            else:
                semdim_dir = os.path.join(semdim_dir, f"{dim1}_{dim2}")
            data = pkl.load(open(os.path.join(semdim_dir, filename), 'rb'))
            alt_data = pkl.load(open(os.path.join(analysis_dir, filename), 'rb'))
            gen_analysis = alt_data.get("generation_analysis", {})
            tokens = gen_analysis.get("tokens", [])
            input_word = gen_analysis.get("input_word", "")
            if data_type == "ipa":
                ipa_list = input_word.split()
            elif data_type == "audio":
                if "ipa_tokens" not in data.keys():
                    continue
                ipa_list = data["ipa_tokens"]
            if not ipa_list:
                print(f"[DEBUG] No valid IPA symbols for word {word}")
                continue
            word_stats = {ipa:{} for ipa in ipa_list}
            word_stats = {k: v for k, v in word_stats.items() if k != ""}
            for dim1, dim2 in self.dim_pairs:
                attn_file_path = None
                for foldername in os.listdir(analysis_dir):
                    if foldername.endswith('.pkl') or foldername.endswith(".json"):
                        continue
                    semdim_dir = os.path.join(analysis_dir, foldername)
                    candidate = f"{word}_{dim1}_{dim2}.pkl"
                    if candidate in os.listdir(semdim_dir):
                        attn_file_path = os.path.join(semdim_dir, candidate)
                        break
                if not attn_file_path:
                    continue
                data = pkl.load(open(attn_file_path, 'rb'))
                alt_data = pkl.load(open(os.path.join(analysis_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl"), 'rb'))
                gen_analysis, target_indices, input_word, wlen, d1len, d2len, response, answer, dim1_range, dim2_range, attention_matrices, tokens = self.get_attention_data(data, alt_data)
                dim1_from_tokens = tokens[target_indices['dim1'][0]:target_indices['dim1'][-1]+1]
                dim2_from_tokens = tokens[target_indices['dim2'][0]:target_indices['dim2'][-1]+1]
                cleaned_dim1_from_tokens = [self._clean_token(token) for token in dim1_from_tokens]
                cleaned_dim2_from_tokens = [self._clean_token(token) for token in dim2_from_tokens]
                idx_to_remove_in_dim1 = []
                for i, dim1_token in enumerate(cleaned_dim1_from_tokens):
                    print(f"i : {i}, dim1_token : '{dim1_token}', original token : '{dim1_from_tokens[i]}'")
                    if dim1_token == "" or dim1_token == "1":
                        idx_to_remove_in_dim1.append(i)
                print(idx_to_remove_in_dim1)
                print(len(target_indices['dim1']))
                for i in reversed(idx_to_remove_in_dim1):
                    print(i)
                    d1len -= 1
                    target_indices['dim1'].pop(i)
                
                idx_to_remove_in_dim2 = []
                for i, dim2_token in enumerate(cleaned_dim2_from_tokens):
                    print(f"i : {i}, dim2_token : {dim2_token}, original token : {dim2_from_tokens[i]}")
                    if dim2_token == "":
                        idx_to_remove_in_dim2.append(i)
                print(idx_to_remove_in_dim2)
                print(len(target_indices['dim2']))
                for i in reversed(idx_to_remove_in_dim2):
                    print(i)
                    d2len -= 1
                    target_indices['dim2'].pop(i)
                    
                dim1_range = range(wlen, wlen+d1len)
                dim2_range = range(wlen+d1len, wlen+d1len+d2len)
                if isinstance(attention_matrices[0], (list, tuple)):
                    attn_layers = attention_matrices[0]
                else:
                    attn_layers = attention_matrices[0]
                    if attn_layers.ndim == 4:
                        attn_layers = attn_layers[0]
                n_layer = len(attn_layers)
                n_head = attn_layers[0].shape[1] if n_layer > 0 else 0
                if data_type == "ipa":
                    for ipa_idx, ipa in enumerate(ipa_list):
                        for dim, dim_range in [(dim1, dim1_range), (dim2, dim2_range)]:
                            all_values = []
                            for layer in range(start_layer, min(end_layer+1, n_layer)):
                                for head in range(n_head):
                                    layer_len = attn_layers[layer].shape[2]
                                    if layer_len != wlen+d1len+d2len:
                                        continue
                                    for d_idx in dim_range:
                                        if d_idx < attn_layers[layer].shape[2] and ipa_idx < attn_layers[layer].shape[3]:
                                            v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                                            all_values.append(v)
                            mean_val = float(np.mean(all_values)) if all_values else np.nan
                            word_stats[ipa][dim] = mean_val
                            print(f"[DEBUG] Word: {word}, IPA: {ipa}, Dim: {dim}, Mean: {mean_val}, (answer: {answer}, response: {response})")
                            
                elif data_type == "audio":
                    ipa_runs = self.get_ipa_runs(ipa_list)
                    for ipa, start_idx, end_idx in ipa_runs:
                        if ipa == "":
                            continue
                        for dim, dim_range in [(dim1, dim1_range), (dim2, dim2_range)]:
                            all_values = []
                            for layer in range(start_layer, min(end_layer+1, n_layer)):
                                for head in range(n_head):
                                    layer_len = attn_layers[layer].shape[2]
                                    if layer_len != wlen+d1len+d2len:
                                        breakpoint()
                                        continue
                                    for d_idx in dim_range:
                                        for ipa_idx in range(start_idx, end_idx+1):
                                            breakpoint()
                                            if d_idx < attn_layers[layer].shape[2] and ipa_idx < attn_layers[layer].shape[3]:
                                                v = attn_layers[layer][0, head, d_idx, ipa_idx].item()
                                                all_values.append(v)
                            mean_val = float(np.mean(all_values)) if all_values else np.nan
                            word_stats[ipa][dim] = mean_val
                            print(f"[DEBUG] Word: {word}, IPA: {ipa}, Dim: {dim}, Mean: {mean_val}, (answer: {answer}, response: {response})")

            valid_dims = set()
            for ipa in word_stats:
                valid_dims.update(word_stats[ipa].keys())
            if len(valid_dims) > 0:
                sampled_words.append(word)
                all_word_stats[word] = word_stats
        return all_word_stats

    def get_attention_data(self, data, alt_data):
        gen_analysis = alt_data.get("generation_analysis", {})
        target_indices:dict[list[int]] = gen_analysis['step_analyses'][0]['target_indices']
        input_word:list[str] = alt_data["input_word"].split()
        wlen = len(target_indices['word'])
        d1len = len(target_indices['dim1'])
        d2len = len(target_indices['dim2'])
        response = gen_analysis.get('response', None)
        answer = gen_analysis.get('answer', None)
        dim1_range = range(wlen, wlen+d1len)
        dim2_range = range(wlen+d1len, wlen+d1len+d2len)
        attention_matrices = data['attention_matrices']
        tokens = data['tokens']
        return gen_analysis, target_indices, input_word, wlen, d1len, d2len, response, answer, dim1_range, dim2_range, attention_matrices, tokens
    
    def plot_sampled_words_heatmaps(self, word_stats, data_type, start_layer, end_layer, lang, save_path=None, suffix:str=None):
        """
        Plot heatmaps for each sampled word (X: ipa, Y: semantic dimension, value: mean attention for both dims in each pair)
        """
        if save_path is None:
            save_path = 'results/plots/attention/sampled_words/'
        os.makedirs(save_path, exist_ok=True)
        semdim_list = [d for pair in self.dim_pairs for d in pair]
        for word, ipa_dict in word_stats.items():
            ipa_list = list(ipa_dict.keys())
            semdim_pairs_to_plot = []
            for dim1, dim2 in self.dim_pairs:
                found_dim1 = any(not np.isnan(ipa_dict[ipa].get(dim1, np.nan)) for ipa in ipa_list)
                found_dim2 = any(not np.isnan(ipa_dict[ipa].get(dim2, np.nan)) for ipa in ipa_list)
                if found_dim1 and found_dim2:
                    semdim_pairs_to_plot.append((dim1, dim2))
            if not semdim_pairs_to_plot:
                continue
            valid_dims = [d for pair in semdim_pairs_to_plot for d in pair]
            matrix = np.full((len(valid_dims), len(ipa_list)), np.nan)
            for i, semdim in enumerate(valid_dims):
                for j, ipa in enumerate(ipa_list):
                    matrix[i, j] = ipa_dict[ipa].get(semdim, np.nan)
            fig, ax = plt.subplots(figsize=(max(10, len(ipa_list)*0.3), max(8, len(valid_dims)*0.3)))
            sns.heatmap(matrix, ax=ax, cmap='YlGnBu', cbar=True,
                        xticklabels=ipa_list, yticklabels=valid_dims, linewidths=0.2, linecolor='gray', square=False)
            # Add thick horizontal lines every 2 rows (for dim pairs)
            for i in range(2, len(valid_dims), 2):
                ax.axhline(i, color='black', linewidth=2)
            ax.set_xlabel('IPA Symbol', fontsize=12)
            ax.set_ylabel('Semantic Dimension', fontsize=12)
            ax.set_title(f'Sampled Word: {word}\nIPA-Semantic Dimension Attention Heatmap', fontsize=14, pad=15)
            plt.setp(ax.get_xticklabels(), ha='right')
            plt.tight_layout()
            clean_word = re.sub(r'[^\w\-]', '_', word)
            file_name = f"sampled_word_{clean_word}_{lang}_{data_type}_generation_attention_L{start_layer}_L{end_layer}{suffix if suffix else ''}.png"
            file_path = os.path.join(save_path, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Sampled word heatmap saved to {file_path}")
            plt.close()

    def aggregate_scores_multi_lang(
        self, data_type, langs, start_layer=20, end_layer=27, response_condition=True
    ):
        import numpy as np
        all_stats = {}
        for lang in langs:
            stats = self.aggregate_scores_with_response_condition(
                data_type, lang, start_layer, end_layer
            ) if response_condition else self.aggregate_scores_across_files_v2(
                data_type, lang, start_layer, end_layer
            )
            if stats:
                all_stats[lang] = stats
        merged = {}
        for lang, stats in all_stats.items():
            if response_condition:
                # Structure: stats[semdim][ipa] = float
                for semdim in stats:
                    for ipa in stats[semdim]:
                        merged.setdefault(ipa, {}).setdefault(semdim, []).append(stats[semdim][ipa])
            else:
                # Structure: stats[ipa][semdim]['all']['mean']
                for ipa in stats:
                    for semdim in stats[ipa]:
                        merged.setdefault(ipa, {}).setdefault(semdim, []).append(stats[ipa][semdim]['all']['mean'])
        stats_all = {}
        for ipa in merged:
            for semdim in merged[ipa]:
                mean = float(np.mean(merged[ipa][semdim]))
                stats_all.setdefault(ipa, {})[semdim] = {'all': {'mean': mean}}
        all_stats['all'] = stats_all
        return all_stats

    def plot_multi_lang_heatmaps(self, all_stats, data_type, start_layer, end_layer, condition_desc):
        for lang, stats in all_stats.items():
            print(f"[PLOT] Plotting for lang={lang}, stats keys={list(stats.keys())[:5]}")
            self.plot_ipa_semdim_heatmap_with_layers(
                stats, save_path='results/plots/attention/', lang=lang,
                data_type=data_type,
                start_layer=start_layer, end_layer=end_layer,
                condition_desc=condition_desc + (f' (all languages)' if lang == 'all' else '')
            )

    def get_ipa_runs(self, ipa_list):
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
    
    def run(self, data_type: str, lang: str, langs: list = None, start_layer: int = 0, end_layer: int = 27):
        print(f"Processing {data_type} data for language {lang}")
        if langs is None:
            langs = [lang]
        all_stats_std = {}
        # for l in langs:
        #     stats = self.aggregate_scores_across_files_v2(
        #         data_type, l, start_layer, end_layer
        #     )
        #     if stats:
        #         all_stats_std[l] = stats
                
        # all_stats_resp_condition = {}
        # for l in langs:
        #     stats = self.aggregate_scores_with_response_condition(
        #         data_type, l, start_layer, end_layer
        #     )
        #     if stats:
        #         all_stats_resp_condition[l] = stats
                
        # for l, stats in all_stats_std.items():
        #     print(f"[PLOT] Plotting standard mean for lang={l}")
        #     self.plot_ipa_semdim_heatmap_with_layers(
        #         stats, save_path='results/plots/attention/', lang=l,
        #         data_type=data_type,
        #         start_layer=start_layer, end_layer=end_layer,
        #         condition_desc="Standard Mean Attention"
        #     )
            
        # for l, stats in all_stats_resp_condition.items():
        #     print(f"[PLOT] Plotting response-condition mean for lang={l}")
        #     self.plot_ipa_semdim_heatmap_with_layers(
        #         stats, save_path='results/plots/attention/', lang=l,
        #         data_type=data_type,
        #         start_layer=start_layer, end_layer=end_layer,
        #         condition_desc="Response-Answer Match, Correct Only"
        #     )
        
        # Add single word sampling if requested
        for l in langs:
            print(f"[SAMPLING] Sampling single words for lang={l}")
            word_stats = self.sample_single_word_response_condition(
                data_type, l, start_layer, end_layer, num_samples=5
            )
            if word_stats:
                self.plot_sampled_words_heatmaps(
                    word_stats, data_type, start_layer, end_layer, l
                )
        
        for l in langs:
            print(f"[SAMPLING] Sampling single words for lang={l}")
            word_stats = self.sample_single_word_response_condition_v2(
                data_type, l, start_layer, end_layer, num_samples=5
            )
            if word_stats:
                self.plot_sampled_words_heatmaps(
                    word_stats, data_type, start_layer, end_layer, l, suffix="_v2"
                )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute IPA-Semantic Dimension Attention Scores")
    parser.add_argument('--data-type', type=str, default="ipa", choices=["audio", "original", "romanized", "ipa"], help="Data type to process")
    parser.add_argument('--lang', type=str, default=None, help="Language(s) to process, comma-separated (default: all ['en','fr','ja','ko','art'])")
    parser.add_argument('--model-path', type=str, default="Qwen/Qwen2.5-Omni-7B", help="Model path")
    parser.add_argument('--constructed', action='store_true', help='Use constructed words as dataset')
    parser.add_argument('--start-layer', type=int, default=20, help='Start layer index for attention score calculation (default: 20)')
    parser.add_argument('--end-layer', type=int, default=27, help='End layer index for attention score calculation (default: 27)')
    args = parser.parse_args()

    all_langs = ['en', 'fr', 'ja', 'ko', 'art']
    if args.lang:
        langs = [l.strip() for l in args.lang.split(',') if l.strip() in all_langs]
        if not langs:
            print(f"No valid languages specified in --lang. Using all: {all_langs}")
            langs = all_langs
    else:
        if args.constructed:
            langs = ['art']
        else:
            langs = ['en', 'fr', 'ja', 'ko']

    asc = AttentionScoreCalculator(
        model_path=args.model_path,
        data_type=args.data_type,
        lang=langs,
        layer_type="generation",
        head="all",
        layer="all",
        compute_type="heatmap",
        constructed=args.constructed
    )

    for lang in langs:
        print(f"\n=== Processing language: {lang} ===")
        asc.run(
            args.data_type, lang, langs=langs,
            start_layer=args.start_layer, end_layer=args.end_layer,
        )