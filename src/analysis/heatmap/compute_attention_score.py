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

# python src/analysis/heatmap/compute_attention_score.py --data-type ipa --attention-type generation_attention
# python src/analysis/heatmap/compute_attention_score.py --data-type audio --attention-type generation_attention
# python src/analysis/heatmap/compute_attention_score.py --data-type ipa --attention-type self_attention
# ipa_to_feature_map = json.load(open("./data/constructed_words/ipa_to_feature.json"))
# feature_to_score_map = json.load(open("./data/constructed_words/feature_to_score.json"))
data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"

'''
1. Input : combination of data type, language, attention type, computation type, layers and heads
data_type : audio, word, romanization, ipa
attention_type : self_attention, generation
computation_type : flow, heatmap
    - flow : 
    - heatmap : compute the attention score of a 
layers : 0-27, or all
heads : 0-27, or all

# Output
1. Matrix : Phonemes at X axis (around 50) and Semantic dimensions (50) at Y axis.
Score refers to the attention score it got with the pair of phonemes and semantic dimensions.
2. list[float] : Attention score of each phoneme.
'''

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
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.data_type = data_type
        self.lang = lang
        self.layer_type = layer_type
        self.head = head
        self.layer = layer
        self.compute_type = compute_type
        self.output_dir = "results/experiments/understanding/attention_heatmap"
        
        # Semantic dimensions list
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
    
    def _clean_token(self, token):
        """Clean token by removing special characters, same as in semdim_heatmap.py"""
        return re.sub(r'^[ĠĊ\[\],.:;!?\n\r\t]+|[ĠĊ\[\],.:;!?\n\r\t]+$', '', token)
    
    def extract_ipa_tokens_from_word(self, tokens, word_tokens):
        """Extract IPA tokens from the word tokens in the sequence"""
        ipa_tokens = []
        
        # Find the word tokens in the sequence
        word_subtokens = []
        if isinstance(word_tokens, str):
            # If word_tokens is a string, tokenize it
            from transformers import Qwen2_5OmniProcessor
            processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
            word_subtokens = processor.tokenizer.tokenize(word_tokens)
        else:
            word_subtokens = word_tokens
        
        # Find the word in the tokens sequence
        for i, token in enumerate(tokens):
            clean_token = self._clean_token(token)
            
            # Skip if token is empty, whitespace only, or not a valid IPA symbol
            if not clean_token or clean_token.strip() == '':
                continue
            
            # Check if this is an IPA symbol
            if clean_token in self.ipa_symbols:
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

        # ---- 추가: 전체 표 형태로 출력 ----
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
            # print("\n=== IPA × Semantic Dimension Mean Attention Score Table ===")
            # with pd.option_context('display.max_rows', 100, 'display.max_columns', 100, 'display.width', 200):
            #     print(df.round(4))
        except ImportError:
            print("[WARN] pandas가 설치되어 있지 않아 표 형태로 출력하지 않습니다.")
        # ---- 추가: semantic dimension별 상위 5개 IPA를 column별로 나란히 출력 (dim1-dim2 쌍 순서, 8개마다 줄 띄움) ----
        try:
            from termcolor import colored
        except ImportError:
            def colored(text, color=None, attrs=None):
                return text
        
        # 대립쌍 정의 (dim1, dim2)
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
        
        # 헤더 출력 (8개씩)
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
            print()  # 8개 dimension마다 한 줄 띄움
        return {
            'ipa_semdim_stats': stats,
            'file_count': file_count
        }
    
    def aggregate_scores_across_files_v2(self, data_type: str, lang: str, attention_type: str = "self_attention"):
        """Aggregate attention scores for each (ipa, semantic dimension, layer, head) and compute statistics."""
        import numpy as np
        import json
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        if attention_type == "generation_attention":
            analysis_dir = os.path.join(base_dir, "generation_attention")
        else:
            analysis_dir = os.path.join(base_dir, "self_attention")
        if not os.path.exists(analysis_dir):
            print(f"Directory not found: {analysis_dir}")
            return None
        all_scores = {}  # (ipa, semdim, layer, head): [score, ...]
        file_count = 0
        for filename in os.listdir(analysis_dir):
            if not filename.endswith('.pkl'):
                continue
            file_path = os.path.join(analysis_dir, filename)
            try:
                with open(file_path, "rb") as f:
                    data = pkl.load(f)
                
                # Try to get tokens, target_indices, attention_matrix
                tokens = data.get('tokens')
                target_indices = data.get('target_indices')
                attn = data.get('attention_matrix')
                input_word = data.get('input_word', "")
                
                # Try to get input_word from different possible locations
                if not input_word:
                    if "generation_analysis" in data:
                        input_word = data["generation_analysis"].get("input_word", "")
                    elif "word_tokens" in data:
                        input_word = data["word_tokens"]
                
                if tokens is None or target_indices is None or attn is None:
                    print(f"Skipping file (missing keys): {filename}")
                    continue
                
                word_indices = target_indices.get('word', [])
                dim1_indices = target_indices.get('dim1', [])
                dim2_indices = target_indices.get('dim2', [])
                
                if not word_indices or (not dim1_indices and not dim2_indices):
                    print(f"Skipping file (missing indices): {filename}")
                    continue
                
                # Extract individual IPA symbols from input_word (space-separated)
                ipa_symbols = []
                if input_word:
                    for ipa_part in input_word.split():
                        clean_ipa = self._clean_token(ipa_part)
                        if clean_ipa and clean_ipa in self.ipa_symbols:
                            ipa_symbols.append(clean_ipa)
                
                if not ipa_symbols:
                    print(f"Skipping file (no valid IPA symbols found): {filename}")
                    continue
                
                # attn: [layer, head, seq, seq]
                n_layer, n_head, seq_len, _ = attn.shape
                
                # Get semantic dimension names from file or data
                # Try to parse from filename: ..._{dimension1}_{dimension2}_...
                import re
                m = re.search(r'_([^_]+)_([^_]+)_(self|generation)_?analysis?\\.pkl$', filename)
                if m:
                    semdim1, semdim2 = m.group(1), m.group(2)
                else:
                    semdim1 = data.get('dimension1', 'dim1')
                    semdim2 = data.get('dimension2', 'dim2')
                
                # For each layer, head, word_idx, dim_idx, extract score
                # Map word_indices to individual IPA symbols
                for layer in range(n_layer):
                    for head in range(n_head):
                        # For each IPA symbol, calculate attention scores
                        for ipa_idx, ipa in enumerate(ipa_symbols):
                            # Find corresponding word index (if available)
                            word_idx = word_indices[ipa_idx] if ipa_idx < len(word_indices) else None
                            
                            # Calculate attention from semantic dimensions to this IPA symbol
                            for d_idx in dim1_indices:
                                if d_idx < seq_len and word_idx is not None and word_idx < seq_len:
                                    score = float(attn[layer, head, d_idx, word_idx])
                                    key = (ipa, semdim1, layer, head)
                                    all_scores.setdefault(key, []).append(score)
                            
                            for d_idx in dim2_indices:
                                if d_idx < seq_len and word_idx is not None and word_idx < seq_len:
                                    score = float(attn[layer, head, d_idx, word_idx])
                                    key = (ipa, semdim2, layer, head)
                                    all_scores.setdefault(key, []).append(score)
                
                file_count += 1
                print(f"Processed {filename}: {len(ipa_symbols)} IPA symbols")
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue
        
        # Aggregate statistics
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
        
        # Also compute all-wise (across all layers/heads)
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
        
        # Save as JSON
        output_path = os.path.join(self.output_dir, "semantic_dimension", data_type, lang, attention_type)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"ipa_semdim_attention_stats_{data_type}_{lang}_{attention_type}.json")
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        # print(f"Saved stats to {output_file}")
        
        # Print sorted IPA scores for each semantic dimension
        for semdim in sorted({k[1] for k in all_scores.keys()}):
            # print(f"\n=== Semantic Dimension: {semdim} ===")
            ipa_means = []
            for ipa in stats:
                if semdim in stats[ipa] and 'all' in stats[ipa][semdim]:
                    ipa_means.append((ipa, stats[ipa][semdim]['all']['mean']))
            ipa_means.sort(key=lambda x: x[1], reverse=True)
            # for ipa, mean in ipa_means:
            #     print(f"{ipa}: {mean:.4f}")
        
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
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
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
        ax.set_xticklabels(region_tokens, rotation=45, ha='right', fontsize=8)
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
        
        # 대립쌍 정의 (dim1, dim2)
        dim_pairs = [
            ("good", "bad"), ("beautiful", "ugly"), ("pleasant", "unpleasant"), ("strong", "weak"),
            ("big", "small"), ("rugged", "delicate"), ("active", "passive"), ("fast", "slow"),
            ("sharp", "round"), ("realistic", "fantastical"), ("structured", "disorganized"), ("ordinary", "unique"),
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
        
        # 헤더 출력 (8개씩)
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
            print()  # 8개 dimension마다 한 줄 띄움

    def run(self, data_type: str, lang: str, attention_type: str = "generation_attention", langs: list = None):
        """Main execution function (now supports all-language aggregation)"""
        print(f"Processing {data_type} data for language {lang} with {attention_type}")
        # Aggregate scores across all files (per language)
        aggregated_scores = self.aggregate_scores_across_files(data_type, lang, attention_type)
        if not aggregated_scores:
            print(f"[WARN] No scores found for lang={lang}, data_type={data_type}, attention_type={attention_type}")
            return None
        # Create phoneme-semantic dimension matrix
        matrix, ipa_list, semdim_list = self.create_phoneme_semdim_matrix(aggregated_scores)
        # Save results (per language)
        output_path = os.path.join(self.output_dir, "semantic_dimension", data_type, lang, attention_type)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"ipa_semdim_attention_scores_{data_type}_{lang}_{attention_type}.json")
        results = {
            'matrix': matrix.tolist() if matrix is not None else [],
            'ipa_symbols': ipa_list,
            'semantic_dimensions': semdim_list,
            'detailed_stats': aggregated_scores.get('detailed_stats', {}),
            'file_count': aggregated_scores['file_count'],
            'data_type': data_type,
            'language': lang,
            'attention_type': attention_type,
            'summary': {
                'total_ipa_symbols': len(ipa_list),
                'total_semantic_dimensions': len(semdim_list),
                'matrix_shape': matrix.shape if matrix is not None else (0, 0),
                'processing_info': {
                    'data_type': data_type,
                    'language': lang,
                    'attention_type': attention_type,
                    'files_processed': aggregated_scores['file_count']
                }
            }
        }
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n=== RESULTS SUMMARY for lang={lang} ===")
        print(f"Results saved to {output_file}")
        print(f"Matrix shape: {matrix.shape if matrix is not None else (0, 0)}")
        print(f"Number of IPA symbols: {len(ipa_list)}")
        print(f"Number of semantic dimensions: {len(semdim_list)}")
        print(f"Files processed: {aggregated_scores['file_count']}")
        # Save heatmap (per language)
        if aggregated_scores.get('ipa_semdim_stats'):
            self.plot_ipa_semdim_heatmap(
                aggregated_scores['ipa_semdim_stats'],
                save_path='results/plots/attention/',
                lang=lang,
                data_type=data_type,
                attention_type=attention_type
            )
        else:
            print(f"[WARN] ipa_semdim_stats is empty for lang={lang}, data_type={data_type}, attention_type={attention_type}. No heatmap will be saved.")
        # Save csv/png (per language)
        # ---- All-language (multi) aggregation ----
        if langs is not None and len(langs) > 1:
            print(f"\n=== Aggregating across all languages: {langs} ===")
            aggregated_multi = self.aggregate_scores_across_files_multi(data_type, langs, attention_type)
            matrix_multi, ipa_list_multi, semdim_list_multi = self.create_phoneme_semdim_matrix({'ipa_semdim_stats': aggregated_multi['ipa_semdim_stats']})
            output_path_multi = os.path.join(self.output_dir, "semantic_dimension", data_type, "all", attention_type)
            os.makedirs(output_path_multi, exist_ok=True)
            output_file_multi = os.path.join(output_path_multi, f"ipa_semdim_attention_scores_{data_type}_all_{attention_type}.json")
            results_multi = {
                'matrix': matrix_multi.tolist() if matrix_multi is not None else [],
                'ipa_symbols': ipa_list_multi,
                'semantic_dimensions': semdim_list_multi,
                'file_count': aggregated_multi['file_count'],
                'data_type': data_type,
                'language': 'all',
                'attention_type': attention_type,
                'summary': {
                    'total_ipa_symbols': len(ipa_list_multi),
                    'total_semantic_dimensions': len(semdim_list_multi),
                    'matrix_shape': matrix_multi.shape if matrix_multi is not None else (0, 0),
                    'processing_info': {
                        'data_type': data_type,
                        'language': 'all',
                        'attention_type': attention_type,
                        'files_processed': aggregated_multi['file_count']
                    }
                }
            }
            with open(output_file_multi, 'w') as f:
                json.dump(results_multi, f, indent=2)
            print(f"\n=== RESULTS SUMMARY for ALL LANGUAGES ===")
            print(f"Results saved to {output_file_multi}")
            print(f"Matrix shape: {matrix_multi.shape if matrix_multi is not None else (0, 0)}")
            print(f"Number of IPA symbols: {len(ipa_list_multi)}")
            print(f"Number of semantic dimensions: {len(semdim_list_multi)}")
            print(f"Files processed: {aggregated_multi['file_count']}")
            
            # Print all-language statistics
            if aggregated_multi.get('ipa_semdim_stats'):
                self.print_top_ipa_statistics(aggregated_multi['ipa_semdim_stats'], "ALL LANGUAGES - ")
            
            # Save heatmap/csv/png for all languages
            if aggregated_multi.get('ipa_semdim_stats'):
                self.plot_ipa_semdim_heatmap(
                    aggregated_multi['ipa_semdim_stats'],
                    save_path='results/plots/attention/',
                    lang='all',
                    data_type=data_type,
                    attention_type=attention_type
                )
            # Save csv (dim_pairs 순서)
            try:
                import pandas as pd
                import numpy as np
                stats = aggregated_multi['ipa_semdim_stats']
                dim_pairs = [
                    ("good", "bad"), ("beautiful", "ugly"), ("pleasant", "unpleasant"), ("strong", "weak"),
                    ("big", "small"), ("rugged", "delicate"), ("active", "passive"), ("fast", "slow"),
                    ("sharp", "round"), ("realistic", "fantastical"), ("structured", "disorganized"), ("ordinary", "unique"),
                    ("interesting", "uninteresting"), ("simple", "complex"), ("abrupt", "continuous"), ("exciting", "calming"),
                    ("hard", "soft"), ("happy", "sad"), ("harsh", "mellow"), ("heavy", "light"),
                    ("inhibited", "free"), ("masculine", "feminine"), ("solid", "nonsolid"), ("tense", "relaxed"),
                    ("dangerous", "safe")
                ]
                ordered_semdims = []
                for d1, d2 in dim_pairs:
                    if d1 in stats:
                        ordered_semdims.append(d1)
                    if d2 in stats:
                        ordered_semdims.append(d2)
                all_ipas = sorted(set(ipa for semdim in stats for ipa in stats[semdim]))
                data = []
                for semdim in ordered_semdims:
                    row = []
                    for ipa in all_ipas:
                        if ipa in stats[semdim]:
                            row.append(stats[semdim][ipa]['mean'])
                        else:
                            row.append(np.nan)
                    data.append(row)
                df = pd.DataFrame(data, index=ordered_semdims, columns=all_ipas)
                save_dir = 'results/plots/attention/'
                os.makedirs(save_dir, exist_ok=True)
                csv_path = os.path.join(save_dir, f"ipa_semdim_table_all_{data_type}_{attention_type}.csv")
                df.to_csv(csv_path)
                print(f"IPA-Semantic Dimension table (ALL) saved to {csv_path}")
            except Exception as e:
                print(f"[WARN] Could not save all-language csv: {e}")
        return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute IPA-Semantic Dimension Attention Scores")
    parser.add_argument('--data-type', type=str, default="ipa", choices=["audio", "original", "romanized", "ipa"],
                       help="Data type to process")
    parser.add_argument('--lang', type=str, default=None,
                       help="Language(s) to process, comma-separated (default: all ['en','fr','ja','ko'])")
    parser.add_argument('--attention-type', type=str, default="generation_attention", 
                       choices=["generation_attention", "self_attention"],
                       help="Type of attention to analyze")
    parser.add_argument('--model-path', type=str, default="Qwen/Qwen2.5-Omni-7B",
                       help="Model path")
    args = parser.parse_args()

    all_langs = ['en', 'fr', 'ja', 'ko']
    if args.lang:
        langs = [l.strip() for l in args.lang.split(',') if l.strip() in all_langs]
        if not langs:
            print(f"No valid languages specified in --lang. Using all: {all_langs}")
            langs = all_langs
    else:
        langs = all_langs

    asc = AttentionScoreCalculator(
        model_path=args.model_path,
        data_path=data_path,
        data_type=args.data_type,
        lang=langs[0],  # dummy, will be overridden per language
        layer_type="generation",
        head="all",
        layer="all",
        compute_type="heatmap"
    )

    for lang in langs:
        print(f"\n=== Processing language: {lang} ===")
        asc.run(args.data_type, lang, args.attention_type, langs=langs)

    