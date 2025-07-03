import numpy as np
import json
import os
import re
import argparse
from typing import Union
import gc
import torch
from tqdm import tqdm
from semdim_heatmap import QwenOmniSemanticDimensionVisualizer as qwensemdim

ipa_to_feature_map = json.load(open("./data/constructed_words/ipa_to_feature.json"))
feature_to_score_map = json.load(open("./data/constructed_words/feature_to_score.json"))
'''
1. Input : combination of data type, language, attention type, computation type, layers and heads
data_type : audio, word, romanization, ipa
attention_type : self_attention, generation
computation_type : flow, heatmap
    - flow : 
    - heatmap : compute the attention score of a 
layers : 0-27, or all
heads : 0-7, or all


# Output
1. Matrix : Phonemes at X axis (around 50) and Semantic dimensions (50) at Y axis.
Score refers to the attention score it got with the pair of phonemes and semantic dimensions.
2. list[float] : Attention score of each phoneme.
'''

class AttentionScoreCalculator(qwensemdim):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        data_type: str,
        lang: str,
        layer_type: str,
        head: int,
        layer: int,
        compute_type: str,
    ):
        super().__init__()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.data_type = data_type
        self.lang = lang
        self.layer_type = layer_type
        self.compute_type = compute_type
        self.semantic_dimension_map = {
            
        }
    
    def load_matrix(self, layer_type: str, data_type: str, attention_type: str, word_tokens: str, dimension1: str, dimension2: str, lang: str):
        return super().read_matrix(layer_type, data_type, attention_type, word_tokens, dimension1, dimension2, lang)
    
    def convert_token_to_phoneme(self, tokens: list[str], alpha_to_ipa_map:dict):
        phoneme_list = []
        for token in tokens:
            if token in alpha_to_ipa_map:
                phoneme_list.append(alpha_to_ipa_map[token])
            else:
                phoneme_list.append(token)
        return phoneme_list
    
    def compute_layer_wise_matrix():
        pass
    
    def compute_head_wise_matrix():
        pass
    
    def compute_phoneme_semdim_relation():
        semantic_dimension_map = {
            "good": "",
            "bad": "",
            "beautiful": "",
            "ugly": "",
            "pleasant": "",
            "unpleasant": "",
            "strong": "",
            "weak": "",
            "big": "",
            "small": "",
            "rugged": "",
            "delicate": "",
            "active": "",
            "passive": "",
            "fast": "",
            "slow": "",
            "sharp": "",
            "round": "",
            "realistic": "",
            "fantastical": "",
            "structured": "",
            "disorganized": "",
            "orginary": "",
            "unique": "",
            "interesting": "",
            "uninteresting": "",
            "simple": "",
            "complex": "",
            "abrupt": "",
            "continuous": "",
            "exciting": "",
            "calming": "",
            "hard": "",
            "soft": "",
            "happy": "",
            "sad": "",
            "harsh": "",
            "mellow": "",
            "heavy": "",
            "light": "",
            "inhibited": "",
            "free": "",
            "masculine": "",
            "feminine": "",
            "solid": "",
            "nonsolid": "",
            "tense": "",
            "relaxed": "",
            "dangerous": "",
            "safe": ""
        }
        pass
    
    def run(self, dimension1, dimension2, computation_type, heads, layers, langs):
        
        self.load_matrix()
        self.convert_token_to_phoneme()
        if self.compute_type == "flow":
            self.compute_layer_wise_matrix()
        elif self.compute_type == "heatmap":
            self.compute_head_wise_matrix()
        
    def matrix_layer_level_computation(self, filtered_attention_matrix, computation_type, head:Union[int, str], layer:Union[int, str], phoneme_mean_map:dict):
        # Convert to tensor if it's not already
        if not hasattr(filtered_attention_matrix, 'mean'):
            filtered_attention_matrix = torch.tensor(filtered_attention_matrix)
        
        # Check tensor dimensions and handle accordingly
        tensor_shape = filtered_attention_matrix.shape
        
        if computation_type == "flow":
            computed_matrix = None
            if phoneme_mean_map:
                # Compute attention flow based on phoneme-meaning mapping
                for phoneme, meaning in phoneme_mean_map.items():
                    for meaning_idx, meaning_value in enumerate(meaning):
                        for head_idx, head_value in enumerate(filtered_attention_matrix[meaning_idx]):
                            for layer_idx, layer_value in enumerate(head_value):
                                pass
            else:
                # Handle different tensor dimensions
                if len(tensor_shape) == 4:  # [head, layer, seq_len, seq_len]
                    if isinstance(head, str) and isinstance(layer, str):
                        computed_matrix = torch.mean(filtered_attention_matrix, dim=(0, 1))
                    elif isinstance(head, int) and isinstance(layer, int):
                        computed_matrix = filtered_attention_matrix[head, layer]
                    else:
                        computed_matrix = torch.mean(filtered_attention_matrix, dim=0)
                elif len(tensor_shape) == 3:  # [seq_len, seq_len, seq_len] or similar
                    # For 3D tensor, just take the mean across the first dimension
                    computed_matrix = torch.mean(filtered_attention_matrix, dim=0)
                elif len(tensor_shape) == 2:  # [seq_len, seq_len]
                    computed_matrix = filtered_attention_matrix
                else:
                    # For 1D or other dimensions, just use as is
                    computed_matrix = filtered_attention_matrix
                    
        elif computation_type == "heatmap":
            if len(tensor_shape) == 4:  # [head, layer, seq_len, seq_len]
                if isinstance(head, str) and isinstance(layer, str):
                    computed_matrix = torch.mean(filtered_attention_matrix, dim=(0, 1))
                elif isinstance(head, int) and isinstance(layer, int):
                    computed_matrix = filtered_attention_matrix[head, layer]
                elif isinstance(head, str) and isinstance(layer, int):
                    computed_matrix = torch.mean(filtered_attention_matrix[:, layer], dim=0)
                elif isinstance(head, int) and isinstance(layer, str):
                    computed_matrix = torch.mean(filtered_attention_matrix[head], dim=0)
                else:
                    raise ValueError("head and layer must be either int or str")
            elif len(tensor_shape) == 3:  # [seq_len, seq_len, seq_len] or similar
                # For 3D tensor, just take the mean across the first dimension
                computed_matrix = torch.mean(filtered_attention_matrix, dim=0)
            elif len(tensor_shape) == 2:  # [seq_len, seq_len]
                computed_matrix = filtered_attention_matrix
            else:
                # For 1D or other dimensions, just use as is
                computed_matrix = filtered_attention_matrix
        return computed_matrix