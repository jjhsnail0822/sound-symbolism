import numpy as np
import json
import os
import re
import argparse
import pickle as pkl
from typing import Union, Dict, List, Tuple
import gc
import torch
from tqdm import tqdm
from semdim_heatmap import QwenOmniSemanticDimensionVisualizer as qwensemdim
import matplotlib.pyplot as plt
import seaborn as sns

# python src/analysis/heatmap/compute_attention_score.py --data-type audio --attention-type generation_attention --start-layer 0 --end-layer 27 --constructed
# python src/analysis/heatmap/compute_attention_score.py --data-type audio --attention-type generation_attention --start-layer 21 --end-layer 27 --constructed
# python src/analysis/heatmap/compute_attention_score.py --data-type ipa --attention-type self_attention
# ipa_to_feature_map = json.load(open("./data/constructed_words/ipa_to_feature.json"))
# feature_to_score_map = json.load(open("./data/constructed_words/feature_to_score.json"))
data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"

class AttentionScoreCalculator:
    def __init__(
        self,
        model_path: str,
        data_path: str,
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
        
        self.ipa_symbols = [
            'a', 'ɑ', 'æ', 'ɐ', 'ə', 'ɚ', 'ɝ', 'ɛ', 'ɜ', 'e', 'ɪ', 'i', 'ɨ', 'ɯ', 'o', 'ɔ', 'ʊ', 'u', 'ʌ', 'ʉ',
            'b', 'β', 'c', 'ç', 'd', 'ð', 'f', 'ɡ', 'ɣ', 'h', 'ɦ', 'j', 'k', 'l', 'ɭ', 'ʟ', 'm', 'ɱ', 'n', 'ŋ',
            'ɲ', 'p', 'ɸ', 'q', 'r', 'ɾ', 'ɹ', 'ʁ', 's', 'ʃ', 't', 'θ', 'v', 'w', 'x', 'χ', 'z', 'ʒ', 'ʔ', 'ʕ',
            'ʡ', 'ʢ', 'ʘ', 'ǀ', 'ǃ', 'ǂ', 'ǁ', 'ɓ', 'ɗ', 'ʄ', 'ɠ', 'ʛ', 'ɦ', 'ʍ', 'ɥ', 'ʜ', 'ʢ', 'ʎ', 'ʟ',
            'ɺ', 'ɻ', 'ɽ', 'ʀ', 'ʂ', 'ʈ', 'ʋ', 'ʐ', 'ʑ', 'ʝ', 'ʞ', 'ʟ', 'ʠ', 'ʡ', 'ʢ', 'ʣ', 'ʤ', 'ʥ', 'ʦ',
            'ʧ', 'ʨ', 'ʩ', 'ʪ', 'ʫ', 'ʬ', 'ʭ', 'ʮ', 'ʯ',
            'ɴ', 'ɕ', 'd͡ʑ', 't͡ɕ', 'ʑ', 'ɰ', 'ã', 'õ', 'ɯ̃', 'ĩ', 'ẽ', 'ɯː', 'aː', 'oː', 'iː', 'eː'
        ]
    
    def _clean_token(self, token:str):
        return re.sub(r'^[ĠĊ\[\],.:;!?\n\r\t]+|[ĠĊ\[\],.:;!?\n\r\t]+$', '', token)
    
    def extract_ipa_tokens_from_word(self, tokens:list[str], word_tokens:str):
        ipa_tokens = []
        word_subtokens = []
        if isinstance(word_tokens, str):
            from transformers import Qwen2_5OmniProcessor
            processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
            word_subtokens = processor.tokenizer.tokenize(word_tokens)
        else:
            word_subtokens = word_tokens
        
        for i, token in enumerate(tokens):
            clean_token = self._clean_token(token)
            
            if not clean_token or clean_token.strip() == '':
                continue
            
            if clean_token in self.ipa_symbols:
                ipa_tokens.append(clean_token)
        
        return ipa_tokens
    
    def extract_ipa_from_tokens(self, tokens:list[str]):
        """Extract IPA symbols from tokens (works for both audio and IPA data types)"""
        ipa_tokens = []
        for token in tokens:
            clean_token = self._clean_token(token)
            if clean_token and clean_token in self.ipa_symbols:
                ipa_tokens.append(clean_token)
        return ipa_tokens
    
    def load_matrix(self, layer_type: str, data_type: str, attention_type: str, word_tokens: str, dimension1: str, dimension2: str, lang: str):
        """Load attention matrix from pickle file"""
        try:
            if attention_type == "generation_attention":
                # Load generation attention analysis
                file_path = os.path.join(
                    self.output_dir, 
                    "semantic_dimension", 
                    data_type, 
                    lang, 
                    "generation_attention",
                    f"{word_tokens}_{dimension1}_{dimension2}_generation_analysis.pkl"
                )
            else:
                # Load self attention matrix
                file_path = os.path.join(
                    self.output_dir,
                    "semantic_dimension",
                    data_type,
                    lang,
                    "self_attention",
                    f"{dimension1}_{dimension2}",
                    f"{word_tokens}_{dimension1}_{dimension2}_{layer_type}.pkl"
                )
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
            
            with open(file_path, "rb") as f:
                data = pkl.load(f)
            if attention_type == "generation_attention":
                return data["generation_analysis"], dimension1, dimension2, data.get("answer"), word_tokens, [dimension1, dimension2]
            else:
                return data["attention_matrix"], dimension1, dimension2, data.get("answer"), word_tokens, [dimension1, dimension2]
                
        except Exception as e:
            print(f"Error loading matrix: {e}")
            return None
    
    def extract_ipa_attention_scores(self, attention_matrix, tokens, relevant_indices, dimension1, dimension2, answer=None):
        """Extract attention scores between IPA symbols and semantic dimensions"""
        if attention_matrix is None:
            return None
        
        # Convert to tensor if needed
        if not isinstance(attention_matrix, torch.Tensor):
            attention_matrix = torch.tensor(attention_matrix)
        
        # Get dimension indices
        dim1_indices = []
        dim2_indices = []
        
        # Find dimension tokens in the sequence
        for i, token in enumerate(tokens):
            clean_token = self._clean_token(token)
            if clean_token == dimension1:
                dim1_indices.append(i)
            elif clean_token == dimension2:
                dim2_indices.append(i)
            # Try partial match for cases where tokenization might split the word
            elif dimension1 in clean_token:
                dim1_indices.append(i)
            elif dimension2 in clean_token:
                dim2_indices.append(i)
        
        # Debug print to see what dimensions were found (only for interesting-uninteresting pairs)
        if dimension1 == "interesting" or dimension2 == "interesting" or dimension1 == "uninteresting" or dimension2 == "uninteresting":
            print(f"Found {len(dim1_indices)} tokens for '{dimension1}': {[tokens[i] for i in dim1_indices]}")
            print(f"Found {len(dim2_indices)} tokens for '{dimension2}': {[tokens[i] for i in dim2_indices]}")
        
        # Initialize results
        ipa_dim1_scores = {}
        ipa_dim2_scores = {}
        
        # Process each IPA symbol
        for i, token in enumerate(tokens):
            clean_token = self._clean_token(token)
            
            # Skip if token is empty, whitespace only, or not a valid IPA symbol
            if not clean_token or clean_token.strip() == '' or clean_token not in self.ipa_symbols:
                continue
            
            # Calculate attention scores from this IPA token to dimensions
            dim1_score = 0.0
            dim2_score = 0.0
            valid_dim1_pairs = 0
            valid_dim2_pairs = 0
            
            # Calculate attention to dimension1
            for dim_idx in dim1_indices:
                if dim_idx < attention_matrix.shape[0] and i < attention_matrix.shape[1]:
                    # Remove causal constraint - allow attention in both directions
                    score = attention_matrix[i, dim_idx].item()
                    dim1_score += score
                    valid_dim1_pairs += 1
                    
                    # Also consider reverse direction (dimension to IPA)
                    if i < attention_matrix.shape[0] and dim_idx < attention_matrix.shape[1]:
                        reverse_score = attention_matrix[dim_idx, i].item()
                        dim1_score += reverse_score
                        valid_dim1_pairs += 1
            
            # Calculate attention to dimension2 (bidirectional)
            for dim_idx in dim2_indices:
                if dim_idx < attention_matrix.shape[0] and i < attention_matrix.shape[1]:
                    # Remove causal constraint - allow attention in both directions
                    score = attention_matrix[i, dim_idx].item()
                    dim2_score += score
                    valid_dim2_pairs += 1
                    
                    # Also consider reverse direction (dimension to IPA)
                    if i < attention_matrix.shape[0] and dim_idx < attention_matrix.shape[1]:
                        reverse_score = attention_matrix[dim_idx, i].item()
                        dim2_score += reverse_score
                        valid_dim2_pairs += 1
            
            # Average the scores
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
        
        # If answer is provided, only return scores for the matching dimension
        if answer is not None:
            if answer == dimension1:
                return {
                    'dim1_scores': ipa_dim1_scores,
                    'dimension1': dimension1,
                    'dimension2': dimension2
                }
            elif answer == dimension2:
                return {
                    'dim2_scores': ipa_dim2_scores,
                    'dimension1': dimension1,
                    'dimension2': dimension2
                }
            else:
                # If answer doesn't match either dimension, return empty results
                return {
                    'dim1_scores': {},
                    'dim2_scores': {},
                    'dimension1': dimension1,
                    'dimension2': dimension2
                }
        
        # If no answer provided, return both dimensions (original behavior)
        return {
            'dim1_scores': ipa_dim1_scores,
            'dim2_scores': ipa_dim2_scores,
            'dimension1': dimension1,
            'dimension2': dimension2
        }
    
    def aggregate_scores_across_files(self, data_type: str, lang: str, attention_type: str = "generation_attention"):
        """Aggregate attention scores across all available files for all (ipa, semantic dimension) pairs"""
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        if attention_type == "generation_attention":
            analysis_dir = os.path.join(base_dir, "generation_attention")
        else:
            analysis_dir = os.path.join(base_dir, "self_attention")
        if not os.path.exists(analysis_dir):
            print(f"Directory not found: {analysis_dir}")
            return None
        all_ipa_semdim_scores = {}  # (ipa, semdim): [score, ...]
        file_count = 0
        heatmap_plot_count = 0  # Counter for heatmap generation
        
        for filename in os.listdir(analysis_dir):
            if not filename.endswith('.pkl'):
                continue
            try:
                file_path = os.path.join(analysis_dir, filename)
                with open(file_path, "rb") as f:
                    data = pkl.load(f)
                # Only process new format files
                if not isinstance(data, dict):
                    print(f"Skipping old format file: {filename}")
                    continue
                if attention_type == "generation_attention" and "generation_analysis" in data:
                    try:
                        generation_analysis = data["generation_analysis"]
                        dimension1 = data["dimension1"]
                        dimension2 = data["dimension2"]
                        answer = data.get("answer", "")  # Get the answer from the data
                        input_word = data.get("input_word", "") or generation_analysis.get("input_word", "")
                        word_dim1_raw_all = generation_analysis.get('word_dim1_raw_all', None)
                        word_dim2_raw_all = generation_analysis.get('word_dim2_raw_all', None)
                        
                        # Determine which dimension matches the answer
                        target_dimension = None
                        target_score = None
                        if answer == dimension1:
                            target_dimension = dimension1
                            target_score = word_dim1_raw_all
                        elif answer == dimension2:
                            target_dimension = dimension2
                            target_score = word_dim2_raw_all
                        
                        # Only process if we have a valid answer and target dimension
                        if target_dimension is not None and target_score is not None:
                            avg_target_score = float(target_score)
                            if input_word:
                                ipa_symbols = []
                                tokens = generation_analysis.get("tokens", [])
                                if tokens:
                                    ipa_symbols = self.extract_ipa_from_tokens(tokens)
                                else:
                                    # Fallback to input_word parsing if tokens not available
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
                                    file_count += 1
                        else:
                            # Fallback to step analysis if raw_all values are not available
                            step_analyses = generation_analysis.get('step_analyses', [])
                            if step_analyses and len(step_analyses) > 0:
                                step_analysis = step_analyses[0]
                                word_dim1_raw_matrix = step_analysis.get('word_dim1_raw_matrix', None)
                                word_dim2_raw_matrix = step_analysis.get('word_dim2_raw_matrix', None)
                                
                                # Determine which dimension matches the answer
                                target_dimension = None
                                target_matrix = None
                                if answer == dimension1 and word_dim1_raw_matrix is not None:
                                    target_dimension = dimension1
                                    target_matrix = word_dim1_raw_matrix
                                elif answer == dimension2 and word_dim2_raw_matrix is not None:
                                    target_dimension = dimension2
                                    target_matrix = word_dim2_raw_matrix
                                
                                # Only process if we have a valid answer and target matrix
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
                                            # Fallback to input_word parsing if tokens not available
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
                                            file_count += 1
                    except Exception as e:
                        print(f"Error processing generation analysis file {filename}: {e}")
                        continue
                elif attention_type == "self_attention" and "attention_matrix" in data:
                    try:
                        attention_matrix = data["attention_matrix"]
                        tokens = data.get("tokens", [])
                        dimension1 = data["dimension1"]
                        dimension2 = data["dimension2"]
                        answer = data.get("answer", "")  # Get the answer from the data
                        relevant_indices = data.get("relevant_indices", None)
                        input_word = data.get("input_word", "")
                        if not input_word and "word_tokens" in data:
                            input_word = data["word_tokens"]
                        
                        # Only process if we have a valid answer
                        if answer in [dimension1, dimension2]:
                            # Debug info for interesting-uninteresting pairs
                            if dimension1 == "interesting" or dimension2 == "interesting" or dimension1 == "uninteresting" or dimension2 == "uninteresting":
                                print(f"Processing file {filename}: {dimension1} vs {dimension2}, answer: {answer}")
                            
                            # Audio and IPA data types now both contain IPA symbols directly in tokens
                            # No special processing needed for audio data type
                            
                            ipa_scores = self.extract_ipa_attention_scores(
                                attention_matrix, tokens, relevant_indices, dimension1, dimension2, answer
                            )
                            if ipa_scores:
                                # Get the scores for the matching dimension
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
                                    
                                    # Debug info for interesting-uninteresting pairs
                                    if answer in ["interesting", "uninteresting"]:
                                        print(f"  Added scores for {ipa} -> {answer}: {scores}")
                                
                                file_count += 1
                                
                                # Generate word-level attention heatmap every 50 files
                                if file_count % 50 == 0:
                                    try:
                                        # Get target indices from the data
                                        target_indices = data.get('target_indices', {})
                                        word_indices = target_indices.get('word', [])
                                        dim1_indices = target_indices.get('dim1', [])
                                        dim2_indices = target_indices.get('dim2', [])
                                        
                                        if word_indices and (dim1_indices or dim2_indices):
                                            # Generate heatmap for first layer and head
                                            self.plot_single_word_attention_heatmap(
                                                attention_matrix=attention_matrix,
                                                tokens=tokens,
                                                word_indices=word_indices,
                                                dim1_indices=dim1_indices,
                                                dim2_indices=dim2_indices,
                                                dimension1=dimension1,
                                                dimension2=dimension2,
                                                word_tokens=data.get('word_tokens', 'unknown'),
                                                save_path='results/plots/attention/',
                                                lang=lang,
                                                data_type=data_type,
                                                attention_type=attention_type,
                                                layer_idx=0,
                                                head_idx=0
                                            )
                                            heatmap_plot_count += 1
                                            print(f"Generated word attention heatmap #{heatmap_plot_count} (file #{file_count})")
                                    except Exception as e:
                                        print(f"Warning: Could not generate word attention heatmap for {filename}: {e}")
                    except Exception as e:
                        print(f"Error processing self attention file {filename}: {e}")
                        continue
                else:
                    print(f"Skipping unrecognized file format: {filename}")
                    continue
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue
        print(f"Processed {file_count} files, generated {heatmap_plot_count} word attention heatmaps")
        # Aggregate statistics for each (ipa, semdim)
        stats = {}
        for (ipa, semdim), scores in all_ipa_semdim_scores.items():
            # Filter out 0.0 scores that might be due to missing data
            filtered_scores = [score for score in scores if score > 0.0]
            
            # Only include if we have valid scores
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
        # # Print summary statistics for each semantic dimension
        # for semdim in sorted(stats.keys()):
        #     print(f"\n=== Semantic Dimension: {semdim} ===")
        #     ipa_means = [(ipa, v['mean']) for ipa, v in stats[semdim].items()]
        #     ipa_means.sort(key=lambda x: x[1], reverse=True)
        #     for ipa, mean in ipa_means:
        #         print(f"{ipa}: {mean:.4f}")
        try:
            import pandas as pd
            all_semdims = sorted(stats.keys())
            all_ipas = sorted(set(ipa for semdim in stats for ipa in stats[semdim]))
            data = []
            for ipa in all_ipas:
                row = []
                for semdim in all_semdims:
                    if ipa in stats[semdim]:
                        row.append(stats[semdim][ipa]['mean'])
                    else:
                        row.append(float('nan'))
                data.append(row)
            df = pd.DataFrame(data, index=all_ipas, columns=all_semdims)
        except ImportError:
            print("[WARN]")
        try:
            from termcolor import colored
        except ImportError:
            def colored(text, color=None, attrs=None):
                return text
        
        dim_pairs = [
            ("good", "bad"), ("beautiful", "ugly"), ("pleasant", "unpleasant"), ("strong", "weak"),
            ("big", "small"), ("rugged", "delicate"), ("active", "passive"), ("fast", "slow"),
            ("sharp", "round"), ("realistic", "fantastical"), ("structured", "disorganized"), ("orginary", "unique"),
            ("interesting", "uninteresting"), ("simple", "complex"), ("abrupt", "continuous"), ("exciting", "calming"),
            ("hard", "soft"), ("happy", "sad"), ("harsh", "mellow"), ("heavy", "light"),
            ("inhibited", "free"), ("masculine", "feminine"), ("solid", "nonsolid"), ("tense", "relaxed"),
            ("dangerous", "safe")
        ]
        
        # 실제 stats에 존재하는 dimension만 사용 (dim_pairs 순서)
        all_semdims = []
        for d1, d2 in dim_pairs:
            if d1 in stats:
                all_semdims.append(d1)
            if d2 in stats:
                all_semdims.append(d2)
        
        topk = 5
        top_ipa_per_semdim = []
        for semdim in all_semdims:
            ipa_means = [(ipa, v['mean']) for ipa, v in stats[semdim].items()]
            ipa_means.sort(key=lambda x: x[1], reverse=True)
            top_ipa_per_semdim.append(ipa_means[:topk])
        
        print("\n=== Top 5 IPA per Semantic Dimension (columns: semantic dimension, rows: (IPA, score)) ===")
        for block_start in range(0, len(all_semdims), 8):
            block_dims = all_semdims[block_start:block_start+8]
            block_top_ipa = top_ipa_per_semdim[block_start:block_start+8]
            header = "".join(f"{semdim:^18}" for semdim in block_dims)
            print(header)
            print("=" * (len(block_dims) * 18))
            for row in range(topk):
                row_str = ""
                for col, semdim in enumerate(block_dims):
                    ipa_score_list = block_top_ipa[col]
                    if row < len(ipa_score_list):
                        ipa, score = ipa_score_list[row]
                        if row == 0:
                            ipa_str = colored(f"{ipa:>7}", 'blue', attrs=['bold'])
                            score_str = colored(f"{score:<10.4f}", 'blue', attrs=['bold'])
                        else:
                            ipa_str = f"{ipa:>3}"
                            score_str = f"{score:6.4f}"
                        cell = f"{ipa_str}:{score_str}"
                    else:
                        cell = " " * 12
                    row_str += f"{cell:^18}"
                print(row_str)
            print()
        return {
            'ipa_semdim_stats': stats,
            'file_count': file_count
        }
    
    def aggregate_scores_across_files_v2(self, data_type: str, lang: str, attention_type: str = "self_attention", start_layer:int=0, end_layer:int=27):
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        if attention_type == "generation_attention":
            analysis_dir = os.path.join(base_dir, "generation_attention")
        else:
            analysis_dir = os.path.join(base_dir, "self_attention")
        if not os.path.exists(analysis_dir):
            print(f"Directory not found: {analysis_dir}")
            return None
        all_scores = {}  # (ipa, semdim, layer, head): [score, ...]
        num_of_files = 0
        for foldername in os.listdir(analysis_dir):
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
                target_indices = alt_data['generation_analysis']['step_analyses'][0]['target_indices']
                wlen = len(target_indices['word'])
                d1len = len(target_indices['dim1'])
                d2len = len(target_indices['dim2'])
                input_word = alt_data['generation_analysis']['input_word']
                
                # Extract IPA symbols properly
                tokens = alt_data['generation_analysis'].get("tokens", [])
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
                
                attention_matrices = data['attention_matrices']  # [step][layer][head][N,N] or [step][batch][layer][head][N,N]
                # print(f"Word: {word}, dim1: {dim1}, dim2: {dim2}, wlen: {wlen}, d1len: {d1len}, d2len: {d2len}, total length: {wlen+d1len+d2len}, input word: {input_word}")
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
                            elif attn.ndim == 3:
                                for head in range(n_head):
                                    if len(dim_range) == 0:
                                        continue
                                    valid_scores = []
                                    for d_idx in dim_range:
                                        if d_idx < attn.shape[1] and ipa_idx < attn.shape[2]:
                                            valid_scores.append(attn[head, d_idx, ipa_idx].item())
                                    score = float(np.mean(valid_scores)) if valid_scores else 0.0
                                    key = (ipa, semdim, layer, head)
                                    all_scores.setdefault(key, []).append(score)
                            elif attn.ndim == 2:
                                if len(dim_range) == 0:
                                    continue
                                valid_scores = []
                                for d_idx in dim_range:
                                    if d_idx < attn.shape[0] and ipa_idx < attn.shape[1]:
                                        valid_scores.append(attn[d_idx, ipa_idx].item())
                                score = float(np.mean(valid_scores)) if valid_scores else 0.0
                                key = (ipa, semdim, layer, 0)
                                all_scores.setdefault(key, []).append(score)
                            else:
                                print(f"[WARN] Unexpected attn shape: {attn.shape}")
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
        # breakpoint()
        return stats
    
    def aggregate_scores_across_files_multi(self, data_type: str, langs: list, attention_type: str = "generation_attention"):
        """Aggregate attention scores across all available files for all (ipa, semantic dimension) pairs for multiple languages."""
        all_ipa_semdim_scores = {}
        total_file_count = 0
        for lang in langs:
            result = self.aggregate_scores_across_files(data_type, lang, attention_type)
            if not result:
                continue
            stats = result['ipa_semdim_stats']
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

    def create_phoneme_semdim_matrix(self, aggregated_scores):
        """Create a matrix of phoneme-semantic dimension attention scores"""
        if not aggregated_scores:
            return None, [], []
        
        # Get all unique IPA symbols and semantic dimensions from ipa_semdim_stats
        ipa_semdim_stats = aggregated_scores.get('ipa_semdim_stats', {})
        if not ipa_semdim_stats:
            return None, [], []
        
        # Get all unique IPA symbols and semantic dimensions
        all_ipa_symbols = set()
        all_semantic_dimensions = set()
        
        for semdim in ipa_semdim_stats:
            all_semantic_dimensions.add(semdim)
            for ipa in ipa_semdim_stats[semdim]:
                all_ipa_symbols.add(ipa)
        
        # Create matrix: rows = IPA symbols, columns = semantic dimensions
        ipa_list = sorted(list(all_ipa_symbols))
        semdim_list = sorted(list(all_semantic_dimensions))
        
        matrix = np.zeros((len(ipa_list), len(semdim_list)))
        
        # Fill the matrix with scores
        for i, ipa in enumerate(ipa_list):
            for j, semdim in enumerate(semdim_list):
                if semdim in ipa_semdim_stats and ipa in ipa_semdim_stats[semdim]:
                    matrix[i, j] = ipa_semdim_stats[semdim][ipa]['mean']
        
        return matrix, ipa_list, semdim_list
    
    def plot_ipa_semdim_heatmap(self, ipa_semdim_stats, save_path=None, lang=None, data_type=None, attention_type=None):
        """
        Plot and save a heatmap: X-axis=IPA symbols, Y-axis=semantic dimensions, values=mean attention scores.
        """
        # Define dimension pairs order (same as in aggregate_scores_across_files)
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
            if d1 in ipa_semdim_stats:
                ordered_semdim_list.append(d1)
            if d2 in ipa_semdim_stats:
                ordered_semdim_list.append(d2)
        
        # Debug: Check if interesting and uninteresting are in the stats
        if "interesting" in ipa_semdim_stats:
            print(f"Found 'interesting' in stats with {len(ipa_semdim_stats['interesting'])} IPA symbols")
        else:
            print("'interesting' NOT found in stats")
        
        if "uninteresting" in ipa_semdim_stats:
            print(f"Found 'uninteresting' in stats with {len(ipa_semdim_stats['uninteresting'])} IPA symbols")
        else:
            print("'uninteresting' NOT found in stats")
        
        # Collect all unique IPA symbols and sort them by phonetic features
        ipa_set = set()
        for semdim in ipa_semdim_stats:
            ipa_set.update(ipa_semdim_stats[semdim].keys())
        
        # Define IPA sorting order: vowels first, then consonants by place of articulation
        # Vowels (front to back, high to low)
        vowels = [
            'i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ɛ', 'œ', 'æ', 'a', 'ɶ',  # Front vowels
            'ɨ', 'ʉ', 'ɯ', 'u', 'ɤ', 'o', 'ɜ', 'ɞ', 'ʌ', 'ɔ', 'ɑ', 'ɒ'  # Central/Back vowels
        ]
        
        # Consonants by place of articulation (bilabial to glottal)
        consonants = [
            # Bilabial
            'p', 'b', 'ɸ', 'β', 'm', 'ɱ',
            # Labiodental
            'f', 'v',
            # Dental
            'θ', 'ð',
            # Alveolar
            't', 'd', 's', 'z', 'n', 'r', 'ɾ', 'ɹ', 'l', 'ɬ', 'ɮ',
            # Post-alveolar
            'ʃ', 'ʒ', 'ɻ',
            # Retroflex
            'ʈ', 'ɖ', 'ʂ', 'ʐ', 'ɳ', 'ɽ', 'ɭ',
            # Palatal
            'c', 'ɟ', 'ç', 'ʝ', 'ɲ', 'j', 'ʎ',
            # Velar
            'k', 'ɡ', 'x', 'ɣ', 'ŋ', 'ɰ', 'ʟ',
            # Uvular
            'q', 'ɢ', 'χ', 'ʁ', 'ɴ',
            # Pharyngeal
            'ħ', 'ʕ',
            # Glottal
            'h', 'ɦ', 'ʔ'
        ]
        
        # Sort IPA symbols according to our defined order
        sorted_ipa_list = []
        for ipa in vowels + consonants:
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
        vowel_count = sum(1 for ipa in sorted_ipa_list if ipa in vowels)
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
        ax.set_title(f'IPA-Semantic Dimension Attention Heatmap ({lang}, {data_type}, {attention_type})', fontsize=16, pad=15)
        
        plt.setp(ax.get_xticklabels(), ha='right')
        
        plt.tight_layout()
        if save_path is None:
            save_path = 'results/plots/attention/'
        os.makedirs(save_path, exist_ok=True)
        file_name = f"ipa_semdim_attention_heatmap_{lang}_{data_type}_{attention_type}.png"
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"IPA-Semantic Dimension heatmap saved to {file_path}")
        plt.close()
    
    def plot_single_word_attention_heatmap(self, attention_matrix, tokens, word_indices, dim1_indices, dim2_indices, 
                                         dimension1, dimension2, word_tokens, save_path=None, lang=None, 
                                         data_type=None, attention_type=None, layer_idx=0, head_idx=0):
        """
        Plot attention heatmap for a single word showing attention from word tokens to semantic dimensions.
        X-axis: tokens from min(word_indices) to max(dim2_indices)
        Y-axis: attention heads (if multiple heads) or layers (if multiple layers)
        """
        if attention_matrix is None:
            print("No attention matrix provided")
            return
        
        # Convert to tensor if needed
        if not isinstance(attention_matrix, torch.Tensor):
            attention_matrix = torch.tensor(attention_matrix)
        
        # Get the attention matrix for specified layer and head
        if len(attention_matrix.shape) == 4:  # [layer, head, seq, seq]
            attn = attention_matrix[layer_idx, head_idx]
        elif len(attention_matrix.shape) == 3:  # [layer, seq, seq]
            attn = attention_matrix[layer_idx]
        else:  # [seq, seq]
            attn = attention_matrix
        
        # Define the region to plot
        min_word_idx = min(word_indices) if word_indices else 0
        max_dim_idx = max(dim1_indices + dim2_indices) if (dim1_indices or dim2_indices) else attn.shape[0]
        
        # Extract the relevant region
        start_idx = max(0, min_word_idx - 2)  # Include some context
        end_idx = min(attn.shape[0], max_dim_idx + 3)  # Include some context
        
        region_attn = attn[start_idx:end_idx, start_idx:end_idx].numpy()
        region_tokens = tokens[start_idx:end_idx] if tokens else [f"token_{i}" for i in range(start_idx, end_idx)]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(region_attn, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(region_tokens)))
        ax.set_yticks(range(len(region_tokens)))
        ax.set_xticklabels(region_tokens, ha='right', fontsize=8)
        ax.set_yticklabels(region_tokens, fontsize=8)
        
        # Highlight word and dimension regions
        word_region = [i - start_idx for i in word_indices if start_idx <= i < end_idx]
        dim1_region = [i - start_idx for i in dim1_indices if start_idx <= i < end_idx]
        dim2_region = [i - start_idx for i in dim2_indices if start_idx <= i < end_idx]
        
        # Add rectangles to highlight regions
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
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Score', fontsize=12)
        
        # Set title and labels
        title = f'Word Attention Heatmap: {word_tokens}\n{dimension1} vs {dimension2}'
        if len(attention_matrix.shape) >= 3:
            title += f' (Layer {layer_idx}, Head {head_idx})'
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Query Tokens', fontsize=12)
        ax.set_ylabel('Key Tokens', fontsize=12)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='blue', label='Word Tokens'),
            Patch(facecolor='none', edgecolor='red', label=f'{dimension1}'),
            Patch(facecolor='none', edgecolor='green', label=f'{dimension2}')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = 'results/plots/attention/'
        os.makedirs(save_path, exist_ok=True)
        
        # Clean word_tokens for filename
        clean_word = re.sub(r'[^\w\-]', '_', word_tokens)
        file_name = f"word_attention_heatmap_{clean_word}_{dimension1}_{dimension2}_{lang}_{data_type}_{attention_type}_L{layer_idx}H{head_idx}.png"
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Word attention heatmap saved to {file_path}")
        plt.close()
        
        return file_path
    
    def print_top_ipa_statistics(self, stats, title_prefix=""):
        """Print top 5 IPA per semantic dimension statistics"""
        try:
            from termcolor import colored
        except ImportError:
            def colored(text, color=None, attrs=None):
                return text
        
        dim_pairs = [
            ("good", "bad"), ("beautiful", "ugly"), ("pleasant", "unpleasant"), ("strong", "weak"),
            ("big", "small"), ("rugged", "delicate"), ("active", "passive"), ("fast", "slow"),
            ("sharp", "round"), ("realistic", "fantastical"), ("structured", "disorganized"), ("ordinary", "unique"),
            ("interesting", "uninteresting"), ("simple", "complex"), ("abrupt", "continuous"), ("exciting", "calming"),
            ("hard", "soft"), ("happy", "sad"), ("harsh", "mellow"), ("heavy", "light"),
            ("inhibited", "free"), ("masculine", "feminine"), ("solid", "nonsolid"), ("tense", "relaxed"),
            ("dangerous", "safe")
        ]
        
        all_semdims = []
        for d1, d2 in dim_pairs:
            if d1 in stats:
                all_semdims.append(d1)
            if d2 in stats:
                all_semdims.append(d2)
        
        topk = 5
        top_ipa_per_semdim = []
        for semdim in all_semdims:
            ipa_means = [(ipa, v['mean']) for ipa, v in stats[semdim].items()]
            ipa_means.sort(key=lambda x: x[1], reverse=True)
            top_ipa_per_semdim.append(ipa_means[:topk])
        
        print(f"\n=== {title_prefix}Top 5 IPA per Semantic Dimension (columns: semantic dimension, rows: (IPA, score)) ===")
        for block_start in range(0, len(all_semdims), 8):
            block_dims = all_semdims[block_start:block_start+8]
            block_top_ipa = top_ipa_per_semdim[block_start:block_start+8]
            header = "".join(f"{semdim:^18}" for semdim in block_dims)
            print(header)
            print("=" * (len(block_dims) * 18))
            for row in range(topk):
                row_str = ""
                for col, semdim in enumerate(block_dims):
                    ipa_score_list = block_top_ipa[col]
                    if row < len(ipa_score_list):
                        ipa, score = ipa_score_list[row]
                        if row == 0:
                            ipa_str = colored(f"{ipa:>7}", 'blue', attrs=['bold'])
                            score_str = colored(f"{score:<10.4f}", 'blue', attrs=['bold'])
                        else:
                            ipa_str = f"{ipa:>7}"
                            score_str = f"{score:<10.4f}"
                        cell = f"{ipa_str}:{score_str}"
                    else:
                        cell = " " * 12
                    row_str += f"{cell:^18}"
                print(row_str)
            print()

    def plot_ipa_semdim_heatmap_with_layers(
        self, stats, save_path, lang, data_type, attention_type, start_layer, end_layer, condition_desc=""
    ):
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
        
        # Define IPA sorting order: vowels first, then consonants by place of articulation
        # Vowels (front to back, high to low)
        vowels = [
            'i', 'y', 'ɪ', 'ʏ', 'e', 'ø', 'ɛ', 'œ', 'æ', 'a', 'ɶ',  # Front vowels
            'ɨ', 'ʉ', 'ɯ', 'u', 'ɤ', 'o', 'ɜ', 'ɞ', 'ʌ', 'ɔ', 'ɑ', 'ɒ'  # Central/Back vowels
        ]
        
        # Consonants by place of articulation (bilabial to glottal)
        consonants = [
            # Bilabial
            'p', 'b', 'ɸ', 'β', 'm', 'ɱ',
            # Labiodental
            'f', 'v',
            # Dental
            'θ', 'ð',
            # Alveolar
            't', 'd', 's', 'z', 'n', 'r', 'ɾ', 'ɹ', 'l', 'ɬ', 'ɮ',
            # Post-alveolar
            'ʃ', 'ʒ', 'ɻ',
            # Retroflex
            'ʈ', 'ɖ', 'ʂ', 'ʐ', 'ɳ', 'ɽ', 'ɭ',
            # Palatal
            'c', 'ɟ', 'ç', 'ʝ', 'ɲ', 'j', 'ʎ',
            # Velar
            'k', 'ɡ', 'x', 'ɣ', 'ŋ', 'ɰ', 'ʟ',
            # Uvular
            'q', 'ɢ', 'χ', 'ʁ', 'ɴ',
            # Pharyngeal
            'ħ', 'ʕ',
            # Glottal
            'h', 'ɦ', 'ʔ'
        ]
        
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
        for ipa in vowels + consonants:
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
        
        import matplotlib.pyplot as plt
        import seaborn as sns
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
        vowel_count = sum(1 for ipa in sorted_ipa_list if ipa in vowels)
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
        file_name = f"ipa_semdim_attention_heatmap_{lang}_{data_type}_{attention_type}_L{start_layer}_L{end_layer}.png"
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"IPA-Semantic Dimension heatmap saved to {file_path}")
        plt.close()

    def aggregate_scores_with_response_condition(
        self, data_type, lang, attention_type="generation_attention", start_layer=20, end_layer=27
    ):
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        if attention_type == "generation_attention":
            analysis_dir = os.path.join(base_dir, "generation_attention")
        else:
            analysis_dir = os.path.join(base_dir, "self_attention")
        if not os.path.exists(analysis_dir):
            print(f"Directory not found: {analysis_dir}")
            return None
        all_scores = {}  # (ipa, semdim, layer, head): [score, ...]
        num_of_files = 0
        for foldername in os.listdir(analysis_dir):
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
                            attn = attn_layers[layer]
                        else:
                            attn = attn_layers[layer]
                        
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
                                
                        elif attn.ndim == 3:
                            for head in range(n_head):
                                if len(correct_range) == 0:
                                    continue
                                valid_correct_scores = []
                                valid_wrong_scores = []
                                for d_idx in correct_range:
                                    if d_idx < attn.shape[1] and ipa_idx < attn.shape[2]:
                                        valid_correct_scores.append(attn[head, d_idx, ipa_idx].item())
                                for d_idx in wrong_range:
                                    if d_idx < attn.shape[1] and ipa_idx < attn.shape[2]:
                                        valid_wrong_scores.append(attn[head, d_idx, ipa_idx].item())
                                
                                correct_score = float(np.mean(valid_correct_scores)) if valid_correct_scores else 0.0
                                wrong_score = float(np.mean(valid_wrong_scores)) if valid_wrong_scores else 0.0
                                denom = correct_score + wrong_score
                                score = correct_score / denom if denom > 0 else 0.0
                                key = (ipa, correct_dim, layer, head)
                                all_scores.setdefault(key, []).append(score)
                                
                        elif attn.ndim == 2:
                            if len(correct_range) == 0:
                                continue
                            valid_correct_scores = []
                            valid_wrong_scores = []
                            for d_idx in correct_range:
                                if d_idx < attn.shape[0] and ipa_idx < attn.shape[1]:
                                    valid_correct_scores.append(attn[d_idx, ipa_idx].item())
                            for d_idx in wrong_range:
                                if d_idx < attn.shape[0] and ipa_idx < attn.shape[1]:
                                    valid_wrong_scores.append(attn[d_idx, ipa_idx].item())
                            
                            correct_score = float(np.mean(valid_correct_scores)) if valid_correct_scores else 0.0
                            wrong_score = float(np.mean(valid_wrong_scores)) if valid_wrong_scores else 0.0
                            denom = correct_score + wrong_score
                            score = correct_score / denom if denom > 0 else 0.0
                            key = (ipa, correct_dim, layer, 0)
                            all_scores.setdefault(key, []).append(score)
                        else:
                            print(f"[WARN] Unexpected attn shape: {attn.shape}")
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

    def sample_single_word_response_condition(
        self, data_type, lang, attention_type="generation_attention", start_layer=20, end_layer=27, num_samples=5
    ):
        """
        Sample single words and compute response condition attention scores for each.
        Returns attention scores for randomly sampled words that meet response condition.
        """
        import random
        
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        if attention_type == "generation_attention":
            analysis_dir = os.path.join(base_dir, "generation_attention")
        else:
            analysis_dir = os.path.join(base_dir, "self_attention")
        if not os.path.exists(analysis_dir):
            print(f"Directory not found: {analysis_dir}")
            return None
        
        # Collect all available files that meet response condition
        valid_files = []
        for foldername in os.listdir(analysis_dir):
            if foldername.endswith('.pkl') or foldername.endswith(".json"):
                continue
            semdim_dir = os.path.join(analysis_dir, foldername)
            for filename in os.listdir(semdim_dir):
                if not filename.endswith('.pkl'):
                    continue
                try:
                    word, dim1, dim2 = filename.rsplit("_", 2)
                    if dim2.endswith(".pkl"):
                        dim2 = dim2[:-4]
                    
                    # Check if corresponding generation analysis exists
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
                        valid_files.append((word, dim1, dim2, filename, semdim_dir))
                except Exception as e:
                    print(f"Error checking file {filename}: {e}")
                    continue
        
        if not valid_files:
            print(f"No valid files found for response condition in {lang}")
            return None
        
        # Randomly sample words
        sampled_files = random.sample(valid_files, min(num_samples, len(valid_files)))
        
        all_word_stats = {}
        
        for i, (word, dim1, dim2, filename, semdim_dir) in enumerate(sampled_files):
            print(f"Processing sampled word {i+1}/{len(sampled_files)}: {word} ({dim1} vs {dim2})")
            
            try:
                data = pkl.load(open(os.path.join(semdim_dir, filename), 'rb'))
                alt_data = pkl.load(open(os.path.join(analysis_dir, f"{word}_{dim1}_{dim2}_generation_analysis.pkl"), 'rb'))
                
                gen_analysis = alt_data.get("generation_analysis", {})
                answer = gen_analysis.get("answer", None)
                response = gen_analysis.get("response", None)
                
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
                
                word_scores = {}  # (ipa, semdim, layer, head): score
                
                for ipa_idx, ipa in enumerate(input_word_list):
                    if ipa_idx >= wlen:
                        continue
                    for layer in range(start_layer, min(end_layer+1, n_layer)):
                        if isinstance(attn_layers, (list, tuple)):
                            attn = attn_layers[layer]
                        else:
                            attn = attn_layers[layer]
                        
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
                                word_scores[(ipa, correct_dim, layer, head)] = score
                                
                        elif attn.ndim == 3:
                            for head in range(n_head):
                                if len(correct_range) == 0:
                                    continue
                                valid_correct_scores = []
                                valid_wrong_scores = []
                                for d_idx in correct_range:
                                    if d_idx < attn.shape[1] and ipa_idx < attn.shape[2]:
                                        valid_correct_scores.append(attn[head, d_idx, ipa_idx].item())
                                for d_idx in wrong_range:
                                    if d_idx < attn.shape[1] and ipa_idx < attn.shape[2]:
                                        valid_wrong_scores.append(attn[head, d_idx, ipa_idx].item())
                                
                                correct_score = float(np.mean(valid_correct_scores)) if valid_correct_scores else 0.0
                                wrong_score = float(np.mean(valid_wrong_scores)) if valid_wrong_scores else 0.0
                                denom = correct_score + wrong_score
                                score = correct_score / denom if denom > 0 else 0.0
                                word_scores[(ipa, correct_dim, layer, head)] = score
                                
                        elif attn.ndim == 2:
                            if len(correct_range) == 0:
                                continue
                            valid_correct_scores = []
                            valid_wrong_scores = []
                            for d_idx in correct_range:
                                if d_idx < attn.shape[0] and ipa_idx < attn.shape[1]:
                                    valid_correct_scores.append(attn[d_idx, ipa_idx].item())
                            for d_idx in wrong_range:
                                if d_idx < attn.shape[0] and ipa_idx < attn.shape[1]:
                                    valid_wrong_scores.append(attn[d_idx, ipa_idx].item())
                            
                            correct_score = float(np.mean(valid_correct_scores)) if valid_correct_scores else 0.0
                            wrong_score = float(np.mean(valid_wrong_scores)) if valid_wrong_scores else 0.0
                            denom = correct_score + wrong_score
                            score = correct_score / denom if denom > 0 else 0.0
                            word_scores[(ipa, correct_dim, layer, 0)] = score
                        else:
                            print(f"[WARN] Unexpected attn shape: {attn.shape}")
                            continue
                
                # Convert word_scores to stats format
                word_stats = {}
                for (ipa, semdim, layer, head), score in word_scores.items():
                    word_stats.setdefault(ipa, {}).setdefault(semdim, {}).setdefault('layerwise', {})[(layer, head)] = {
                        'mean': score,
                        'std': 0.0,
                        'min': score,
                        'max': score,
                        'median': score,
                        'count': 1,
                        'q25': score,
                        'q75': score,
                    }
                
                # Calculate 'all' statistics for each IPA-semdim pair
                for ipa in word_stats:
                    for semdim in word_stats[ipa]:
                        all_means = [v['mean'] for v in word_stats[ipa][semdim]['layerwise'].values()]
                        if all_means:
                            arr = np.array(all_means)
                            word_stats[ipa][semdim]['all'] = {
                                'mean': float(np.mean(arr)),
                                'std': float(np.std(arr)),
                                'min': float(np.min(arr)),
                                'max': float(np.max(arr)),
                                'median': float(np.median(arr)),
                                'count': int(len(arr)),
                                'q25': float(np.percentile(arr, 25)),
                                'q75': float(np.percentile(arr, 75)),
                            }
                
                all_word_stats[word] = {
                    'stats': word_stats,
                    'dimensions': (dim1, dim2),
                    'answer': answer,
                    'input_word': input_word
                }
                
            except Exception as e:
                print(f"Error processing sampled word {word}: {e}")
                continue
        
        return all_word_stats

    def plot_sampled_words_heatmaps(
        self, word_stats, data_type, attention_type, start_layer, end_layer, lang, save_path=None
    ):
        """
        Plot heatmaps for each sampled word
        """
        if save_path is None:
            save_path = 'results/plots/attention/sampled_words/'
        os.makedirs(save_path, exist_ok=True)
        
        for word, word_data in word_stats.items():
            stats = word_data['stats']
            dim1, dim2 = word_data['dimensions']
            answer = word_data['answer']
            input_word = word_data['input_word']
            
            condition_desc = f"Word: {word} ({input_word}) - {dim1} vs {dim2}, Answer: {answer}"
            
            self.plot_ipa_semdim_heatmap_with_layers(
                stats, save_path=save_path, lang=lang,
                data_type=data_type, attention_type=attention_type,
                start_layer=start_layer, end_layer=end_layer,
                condition_desc=condition_desc
            )
            
            # Create custom filename for this word
            clean_word = re.sub(r'[^\w\-]', '_', word)
            file_name = f"sampled_word_{clean_word}_{lang}_{data_type}_{attention_type}_L{start_layer}_L{end_layer}.png"
            old_file_path = os.path.join(save_path, f"ipa_semdim_attention_heatmap_{lang}_{data_type}_{attention_type}_L{start_layer}_L{end_layer}.png")
            new_file_path = os.path.join(save_path, file_name)
            
            if os.path.exists(old_file_path):
                os.rename(old_file_path, new_file_path)
                print(f"Sampled word heatmap saved to {new_file_path}")

    def aggregate_scores_multi_lang(
        self, data_type, langs, attention_type="generation_attention", start_layer=20, end_layer=27, response_condition=True
    ):
        import numpy as np
        all_stats = {}
        for lang in langs:
            stats = self.aggregate_scores_with_response_condition(
                data_type, lang, attention_type, start_layer, end_layer
            ) if response_condition else self.aggregate_scores_across_files_v2(
                data_type, lang, attention_type, start_layer, end_layer
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

    def plot_multi_lang_heatmaps(self, all_stats, data_type, attention_type, start_layer, end_layer, condition_desc):
        for lang, stats in all_stats.items():
            print(f"[PLOT] Plotting for lang={lang}, stats keys={list(stats.keys())[:5]}")
            self.plot_ipa_semdim_heatmap_with_layers(
                stats, save_path='results/plots/attention/', lang=lang,
                data_type=data_type, attention_type=attention_type,
                start_layer=start_layer, end_layer=end_layer,
                condition_desc=condition_desc + (f' (all languages)' if lang == 'all' else '')
            )

    def run(self, data_type: str, lang: str, attention_type: str = "generation_attention", langs: list = None, start_layer: int = 0, end_layer: int = 27, include_sampling: bool = False):
        print(f"Processing {data_type} data for language {lang} with {attention_type}")
        if langs is None:
            langs = [lang]
        all_stats_std = {}
        # for l in langs:
        #     stats = self.aggregate_scores_across_files_v2(
        #         data_type, l, attention_type, start_layer, end_layer
        #     )
        #     if stats:
        #         all_stats_std[l] = stats
        # all_stats_resp_condition = {}
        # for l in langs:
        #     stats = self.aggregate_scores_with_response_condition(
        #         data_type, l, attention_type, start_layer, end_layer
        #     )
        #     if stats:
        #         all_stats_resp_condition[l] = stats
        # for l, stats in all_stats_std.items():
        #     print(f"[PLOT] Plotting standard mean for lang={l}")
        #     self.plot_ipa_semdim_heatmap_with_layers(
        #         stats, save_path='results/plots/attention/', lang=l,
        #         data_type=data_type, attention_type=attention_type,
        #         start_layer=start_layer, end_layer=end_layer,
        #         condition_desc="Standard Mean Attention"
        #     )
        # for l, stats in all_stats_resp_condition.items():
        #     print(f"[PLOT] Plotting response-condition mean for lang={l}")
        #     self.plot_ipa_semdim_heatmap_with_layers(
        #         stats, save_path='results/plots/attention/', lang=l,
        #         data_type=data_type, attention_type=attention_type,
        #         start_layer=start_layer, end_layer=end_layer,
        #         condition_desc="Response-Answer Match, Correct Only"
        #     )
        
        # Add single word sampling if requested
        if include_sampling:
            for l in langs:
                print(f"[SAMPLING] Sampling single words for lang={l}")
                word_stats = self.sample_single_word_response_condition(
                    data_type, l, attention_type, start_layer, end_layer, num_samples=5
                )
                if word_stats:
                    self.plot_sampled_words_heatmaps(
                        word_stats, data_type, attention_type, start_layer, end_layer, l
                    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute IPA-Semantic Dimension Attention Scores")
    parser.add_argument('--data-type', type=str, default="ipa", choices=["audio", "original", "romanized", "ipa"],
                       help="Data type to process")
    parser.add_argument('--lang', type=str, default=None,
                       help="Language(s) to process, comma-separated (default: all ['en','fr','ja','ko','art'])")
    parser.add_argument('--attention-type', type=str, default="generation_attention", 
                       choices=["generation_attention", "self_attention"],
                       help="Type of attention to analyze")
    parser.add_argument('--model-path', type=str, default="Qwen/Qwen2.5-Omni-7B",
                       help="Model path")
    parser.add_argument('--constructed', action='store_true', help='Use constructed words as dataset')
    parser.add_argument('--start-layer', type=int, default=20, help='Start layer index for attention score calculation (default: 20)')
    parser.add_argument('--end-layer', type=int, default=27, help='End layer index for attention score calculation (default: 27)')
    parser.add_argument('--include-sampling', action='store_true', help='Include single word sampling and plotting')
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
        data_path=None,  # will be set internally
        data_type=args.data_type,
        lang=langs[0],  # dummy, will be overridden per language
        layer_type="generation",
        head="all",
        layer="all",
        compute_type="heatmap",
        constructed=args.constructed
    )

    for lang in langs:
        print(f"\n=== Processing language: {lang} ===")
        asc.run(
            args.data_type, lang, args.attention_type, langs=langs,
            start_layer=args.start_layer, end_layer=args.end_layer,
            include_sampling=args.include_sampling
        )

    