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

# python src/analysis/heatmap/compute_attention_score.py --data-type ipa --lang en --attention-type generation_attention
# python src/analysis/heatmap/compute_attention_score.py --data-type ipa --lang en --attention-type self_attention
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
heads : 0-7, or all

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
        
        # IPA symbols list (common IPA symbols)
        self.ipa_symbols = [
            'a', 'ɑ', 'æ', 'ɐ', 'ə', 'ɚ', 'ɝ', 'ɛ', 'ɜ', 'e', 'ɪ', 'i', 'ɨ', 'ɯ', 'o', 'ɔ', 'ʊ', 'u', 'ʌ', 'ʉ',
            'b', 'β', 'c', 'ç', 'd', 'ð', 'f', 'ɡ', 'ɣ', 'h', 'ɦ', 'j', 'k', 'l', 'ɭ', 'ʟ', 'm', 'ɱ', 'n', 'ŋ',
            'ɲ', 'p', 'ɸ', 'q', 'r', 'ɾ', 'ɹ', 'ʁ', 's', 'ʃ', 't', 'θ', 'v', 'w', 'x', 'χ', 'z', 'ʒ', 'ʔ', 'ʕ',
            'ʡ', 'ʢ', 'ʘ', 'ǀ', 'ǃ', 'ǂ', 'ǁ', 'ɓ', 'ɗ', 'ʄ', 'ɠ', 'ʛ', 'ɦ', 'ʍ', 'ɥ', 'ʜ', 'ʢ', 'ʎ', 'ʟ',
            'ɺ', 'ɻ', 'ɽ', 'ʀ', 'ʂ', 'ʈ', 'ʋ', 'ʐ', 'ʑ', 'ʝ', 'ʞ', 'ʟ', 'ʠ', 'ʡ', 'ʢ', 'ʣ', 'ʤ', 'ʥ', 'ʦ',
            'ʧ', 'ʨ', 'ʩ', 'ʪ', 'ʫ', 'ʬ', 'ʭ', 'ʮ', 'ʯ'
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
            breakpoint()
            if attention_type == "generation_attention":
                return data["generation_analysis"], dimension1, dimension2, data.get("answer"), word_tokens, [dimension1, dimension2]
            else:
                return data["attention_matrix"], dimension1, dimension2, data.get("answer"), word_tokens, [dimension1, dimension2]
                
        except Exception as e:
            print(f"Error loading matrix: {e}")
            return None
    
    def extract_ipa_attention_scores(self, attention_matrix, tokens, relevant_indices, dimension1, dimension2):
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
                    # Check causal attention constraint
                    if i <= dim_idx:  # IPA token can attend to dimension token
                        score = attention_matrix[i, dim_idx].item()
                        dim1_score += score
                        valid_dim1_pairs += 1
            
            # Calculate attention to dimension2
            for dim_idx in dim2_indices:
                if dim_idx < attention_matrix.shape[0] and i < attention_matrix.shape[1]:
                    # Check causal attention constraint
                    if i <= dim_idx:  # IPA token can attend to dimension token
                        score = attention_matrix[i, dim_idx].item()
                        dim2_score += score
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
        
        return {
            'dim1_scores': ipa_dim1_scores,
            'dim2_scores': ipa_dim2_scores,
            'dimension1': dimension1,
            'dimension2': dimension2
        }
    
    def aggregate_scores_across_files(self, data_type: str, lang: str, attention_type: str = "generation_attention"):
        """Aggregate attention scores across all available files"""
        base_dir = os.path.join(self.output_dir, "semantic_dimension", data_type, lang)
        
        if attention_type == "generation_attention":
            analysis_dir = os.path.join(base_dir, "generation_attention")
        else:
            analysis_dir = os.path.join(base_dir, "self_attention")
        
        if not os.path.exists(analysis_dir):
            print(f"Directory not found: {analysis_dir}")
            return None
        
        # Collect all IPA scores
        all_ipa_dim1_scores = {}
        all_ipa_dim2_scores = {}
        file_count = 0
        
        # Process each file in the directory
        for filename in os.listdir(analysis_dir):
            if filename.endswith('.pkl'):
                try:
                    file_path = os.path.join(analysis_dir, filename)
                    with open(file_path, "rb") as f:
                        data = pkl.load(f)
                    
                    # Check if this is a dictionary (new format) or list (old format)
                    if not isinstance(data, dict):
                        print(f"Skipping old format file: {filename}")
                        continue
                    
                    # Check if this is generation attention analysis file
                    if attention_type == "generation_attention" and "generation_analysis" in data:
                        try:
                            generation_analysis = data["generation_analysis"]
                            dimension1 = data["dimension1"]
                            dimension2 = data["dimension2"]
                            word_tokens = data["word_tokens"]
                            option_tokens = data["option_tokens"]
                            response = data["response"]
                            answer = data["answer"]
                            
                            # Get the aggregated attention scores (already averaged across all heads and layers)
                            word_dim1_raw_all = generation_analysis.get('word_dim1_raw_all', None)
                            word_dim2_raw_all = generation_analysis.get('word_dim2_raw_all', None)
                            
                            if word_dim1_raw_all is not None and word_dim2_raw_all is not None:
                                # These are already averaged across all heads and layers
                                avg_dim1_score = float(word_dim1_raw_all)
                                avg_dim2_score = float(word_dim2_raw_all)
                                
                                # Extract IPA tokens from the word tokens
                                ipa_tokens = self.extract_ipa_tokens_from_word(generation_analysis.get('tokens', []), word_tokens)
                                
                                if ipa_tokens:
                                    # Distribute scores across IPA tokens
                                    for ipa in ipa_tokens:
                                        if ipa not in all_ipa_dim1_scores:
                                            all_ipa_dim1_scores[ipa] = []
                                        all_ipa_dim1_scores[ipa].append(avg_dim1_score)
                                        
                                        if ipa not in all_ipa_dim2_scores:
                                            all_ipa_dim2_scores[ipa] = []
                                        all_ipa_dim2_scores[ipa].append(avg_dim2_score)
                                    
                                    file_count += 1
                                    
                                    # Print detailed statistics for this file
                                    print(f"File: {filename}")
                                    print(f"  Word: {generation_analysis.get('word', 'N/A')}")
                                    print(f"  Input word: {generation_analysis.get('input_word', 'N/A')}")
                                    print(f"  Response: {response}")
                                    print(f"  Answer: {answer}")
                                    print(f"  IPA tokens: {ipa_tokens}")
                                    print(f"  Dimension1 ({dimension1}) score: {avg_dim1_score:.6f}")
                                    print(f"  Dimension2 ({dimension2}) score: {avg_dim2_score:.6f}")
                                    print(f"  Score difference (dim1 - dim2): {avg_dim1_score - avg_dim2_score:.6f}")
                                    print(f"  Score ratio (dim1/dim2): {avg_dim1_score/avg_dim2_score:.3f}" if avg_dim2_score != 0 else "  Score ratio: undefined")
                                    print()
                            else:
                                # Fallback to step analysis if raw_all values are not available
                                step_analyses = generation_analysis.get('step_analyses', [])
                                if step_analyses and len(step_analyses) > 0:
                                    step_analysis = step_analyses[0]
                                    
                                    word_dim1_raw_matrix = step_analysis.get('word_dim1_raw_matrix', None)
                                    word_dim2_raw_matrix = step_analysis.get('word_dim2_raw_matrix', None)
                                    tokens = step_analysis.get('step_tokens', [])
                                    
                                    if word_dim1_raw_matrix is not None and word_dim2_raw_matrix is not None:
                                        # Average across layers and heads to get a single attention matrix
                                        avg_dim1_score = np.mean(word_dim1_raw_matrix)
                                        avg_dim2_score = np.mean(word_dim2_raw_matrix)
                                        
                                        # Extract IPA tokens from the word tokens
                                        ipa_tokens = self.extract_ipa_tokens_from_word(tokens, word_tokens)
                                        
                                        if ipa_tokens:
                                            # Distribute scores across IPA tokens
                                            for ipa in ipa_tokens:
                                                if ipa not in all_ipa_dim1_scores:
                                                    all_ipa_dim1_scores[ipa] = []
                                                all_ipa_dim1_scores[ipa].append(avg_dim1_score)
                                                
                                                if ipa not in all_ipa_dim2_scores:
                                                    all_ipa_dim2_scores[ipa] = []
                                                all_ipa_dim2_scores[ipa].append(avg_dim2_score)
                                            
                                            file_count += 1
                        except Exception as e:
                            print(f"Error processing generation analysis file {filename}: {e}")
                            continue
                    
                    # Check if this is self attention matrix file
                    elif attention_type == "self_attention" and "attention_matrix" in data:
                        try:
                            attention_matrix = data["attention_matrix"]
                            tokens = data.get("tokens", [])
                            dimension1 = data["dimension1"]
                            dimension2 = data["dimension2"]
                            relevant_indices = data.get("relevant_indices", None)
                            
                            # Extract IPA attention scores from the attention matrix
                            ipa_scores = self.extract_ipa_attention_scores(
                                attention_matrix, tokens, relevant_indices, dimension1, dimension2
                            )
                            
                            if ipa_scores:
                                # Aggregate scores
                                for ipa, scores in ipa_scores['dim1_scores'].items():
                                    if ipa not in all_ipa_dim1_scores:
                                        all_ipa_dim1_scores[ipa] = []
                                    all_ipa_dim1_scores[ipa].extend(scores)
                                
                                for ipa, scores in ipa_scores['dim2_scores'].items():
                                    if ipa not in all_ipa_dim2_scores:
                                        all_ipa_dim2_scores[ipa] = []
                                    all_ipa_dim2_scores[ipa].extend(scores)
                                
                                file_count += 1
                        except Exception as e:
                            print(f"Error processing self attention file {filename}: {e}")
                            continue
                    else:
                        print(f"Skipping unrecognized file format: {filename}")
                        continue
                
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
                    continue
        
        print(f"Processed {file_count} files")
        
        # Calculate final averages
        final_ipa_dim1_scores = {}
        final_ipa_dim2_scores = {}
        
        for ipa, scores in all_ipa_dim1_scores.items():
            if scores:  # Only include if there are valid scores
                final_ipa_dim1_scores[ipa] = np.mean(scores)
        
        for ipa, scores in all_ipa_dim2_scores.items():
            if scores:  # Only include if there are valid scores
                final_ipa_dim2_scores[ipa] = np.mean(scores)
        
        return {
            'dim1_scores': final_ipa_dim1_scores,
            'dim2_scores': final_ipa_dim2_scores,
            'file_count': file_count
        }
    
    def create_phoneme_semdim_matrix(self, aggregated_scores):
        """Create a matrix of phoneme-semantic dimension attention scores"""
        if not aggregated_scores:
            return None
        
        # Get all unique IPA symbols and semantic dimensions
        all_ipa_symbols = set()
        all_ipa_symbols.update(aggregated_scores['dim1_scores'].keys())
        all_ipa_symbols.update(aggregated_scores['dim2_scores'].keys())
        
        # Create matrix: rows = IPA symbols, columns = semantic dimensions
        ipa_list = sorted(list(all_ipa_symbols))
        semdim_list = self.semantic_dimension_map
        
        matrix = np.zeros((len(ipa_list), len(semdim_list)))
        
        # Fill the matrix with scores
        for i, ipa in enumerate(ipa_list):
            for j, semdim in enumerate(semdim_list):
                # For now, we'll use the average scores across all dimensions
                # In the future, we can map specific dimensions to their scores
                if ipa in aggregated_scores['dim1_scores']:
                    matrix[i, j] = aggregated_scores['dim1_scores'][ipa]
                elif ipa in aggregated_scores['dim2_scores']:
                    matrix[i, j] = aggregated_scores['dim2_scores'][ipa]
        
        return matrix, ipa_list, semdim_list
    
    def run(self, data_type: str, lang: str, attention_type: str = "generation_attention"):
        """Main execution function"""
        print(f"Processing {data_type} data for language {lang} with {attention_type}")
        
        # Aggregate scores across all files
        aggregated_scores = self.aggregate_scores_across_files(data_type, lang, attention_type)
        
        if not aggregated_scores:
            print("No scores found")
            return None
        
        # Create phoneme-semantic dimension matrix
        matrix, ipa_list, semdim_list = self.create_phoneme_semdim_matrix(aggregated_scores)
        
        # Save results
        output_path = os.path.join(self.output_dir, "semantic_dimension", data_type, lang, attention_type)
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"ipa_semdim_attention_scores_{data_type}_{lang}_{attention_type}.json")
        results = {
            'matrix': matrix.tolist(),
            'ipa_symbols': ipa_list,
            'semantic_dimensions': semdim_list,
            'dim1_scores': aggregated_scores['dim1_scores'],
            'dim2_scores': aggregated_scores['dim2_scores'],
            'file_count': aggregated_scores['file_count'],
            'data_type': data_type,
            'language': lang,
            'attention_type': attention_type
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Number of IPA symbols: {len(ipa_list)}")
        print(f"Number of semantic dimensions: {len(semdim_list)}")
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IPA-Semantic Dimension Attention Scores")
    parser.add_argument('--data-type', type=str, default="ipa", choices=["audio", "original", "romanized", "ipa"],
                       help="Data type to process")
    parser.add_argument('--lang', type=str, default="en", choices=["en", "fr", "ja", "ko"],
                       help="Language to process")
    parser.add_argument('--attention-type', type=str, default="generation_attention", 
                       choices=["generation_attention", "self_attention"],
                       help="Type of attention to analyze")
    parser.add_argument('--model-path', type=str, default="Qwen/Qwen2.5-Omni-7B",
                       help="Model path")
    
    args = parser.parse_args()
    
    asc = AttentionScoreCalculator(
        model_path=args.model_path,
        data_path=data_path,
        data_type=args.data_type,
        lang=args.lang,
        layer_type="generation",
        head="all",
        layer="all",
        compute_type="heatmap"
    )
    
    results = asc.run(args.data_type, args.lang, args.attention_type)
    