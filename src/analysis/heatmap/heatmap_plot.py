# Heatmap Plotting Module for Semantic Dimension Analysis
import json
import re
import os
import argparse
import pickle as pkl
from typing import Union, List, Dict, Any
import warnings
# Suppress matplotlib UserWarnings globally
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from tqdm import tqdm

def set_font_for_language(lang):
    """Set appropriate font based on language with fallback system"""
    # Define font preferences for each language with fallbacks
    font_preferences = {
        "en": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
        "fr": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
        "ko": ["Malgun Gothic", "Nanum Gothic", "DejaVu Sans", "sans-serif"],
        "ja": ["Noto Sans CJK JP", "Source Han Sans JP", "Hiragino Sans", "Yu Gothic", "DejaVu Sans", "sans-serif"]
    }
    
    # Get available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Try to find an available font from preferences
    fonts_to_try = font_preferences.get(lang, ["DejaVu Sans", "sans-serif"])
    
    selected_font = None
    for font in fonts_to_try:
        if font in available_fonts or font == "sans-serif":
            selected_font = font
            break
    
    if selected_font:
        matplotlib.rcParams['font.family'] = selected_font
        if selected_font != "DejaVu Sans":
            print(f"Set font for language '{lang}' to: {selected_font}")
    else:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print(f"Warning: No suitable font found for language '{lang}', using default sans-serif")
    
    # Enable Unicode support and set font properties
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.sans-serif'] = [selected_font] if selected_font else ['sans-serif']

class SemanticDimensionHeatmapPlotter:
    def __init__(
        self,
        output_dir: str = "results/experiments/understanding/attention_heatmap",
        exp_type: str = "semantic_dimension",
        data_type: str = "audio",
        phoneme_mean_map: dict = None
    ):
        self.output_dir = output_dir
        self.exp_type = exp_type
        self.data_type = data_type
        self.phoneme_mean_map = phoneme_mean_map
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def find_token_indices(self, tokens, target_tokens):
        """Find indices of target tokens in the token list"""
        indices = []
        
        def clean_token(token):
            """Clean token by removing 'Ġ' and check if it's a special token"""
            if token.startswith("Ġ"):
                token = token[1:]  # Remove "Ġ"
            return token
        
        def is_special_token(token):
            """Check if token is a special token (contains '<' and '>')"""
            return '<' in token and '>' in token
        
        # Process each target
        for target in target_tokens:
            if isinstance(target, str):
                clean_target = clean_token(target)
                for idx, token in enumerate(tokens):
                    clean_token_str = clean_token(token)
                    if clean_target == clean_token_str:
                        indices.append(idx)
            elif isinstance(target, list):
                clean_targets = [clean_token(t) for t in target]
                for i in range(len(tokens) - len(clean_targets) + 1):
                    match_found = True
                    sequence_indices = []
                    for j, clean_target in enumerate(clean_targets):
                        current_token = tokens[i + j]
                        clean_current = clean_token(current_token)
                        if clean_current != clean_target or (j > 0 and is_special_token(current_token)):
                            match_found = False
                            break
                        sequence_indices.append(i + j)
                    if match_found:
                        indices.extend(sequence_indices)
                        break
        
        return sorted(list(set(indices)))
    
    def filter_relevant_indices(self, attention_matrix, row_tokens, column_tokens, word_tokens, option_tokens, dimension1, dimension2, answer, layer_type="self"):
        """Filter attention matrix to include only relevant token indices"""
        save_row_index = []
        save_column_index = []
        
        # Find word token indices
        word_indices = self.find_token_indices(row_tokens, [word_tokens])
        if self.data_type == "audio":
            word_indices = self.find_token_indices(row_tokens, ["<|AUDIO|>"])
        dim1_indices = self.find_token_indices(row_tokens, [dimension1])
        dim2_indices = self.find_token_indices(row_tokens, [dimension2])
        
        if layer_type == "self":
            for idx, token in enumerate(row_tokens):
                if idx in word_indices or idx in dim1_indices or idx in dim2_indices:
                    save_row_index.append(idx)
            for idx, token in enumerate(column_tokens):
                if idx in word_indices or idx in dim1_indices or idx in dim2_indices:
                    save_column_index.append(idx)
        elif layer_type == "cross":
            for idx, token in enumerate(row_tokens):
                if idx in word_indices or idx in dim1_indices or idx in dim2_indices:
                    save_row_index.append(idx)
            for idx, token in enumerate(column_tokens):
                if idx in option_tokens or idx in dim1_indices or idx in dim2_indices:
                    save_column_index.append(idx)
        elif layer_type == "output":
            answer_indices = self.find_token_indices(row_tokens, [answer])
            for idx, token in enumerate(row_tokens):
                if idx in answer_indices or idx in dim1_indices or idx in dim2_indices:
                    save_row_index.append(idx)
            for idx, token in enumerate(column_tokens):
                if idx in option_tokens or idx in dim1_indices or idx in dim2_indices:
                    save_column_index.append(idx)
        
        # Filter the attention matrix
        if save_row_index and save_column_index:
            if isinstance(attention_matrix, tuple):
                attention_matrix = attention_matrix[0]
            
            if not hasattr(attention_matrix, 'index_select'):
                attention_matrix = torch.tensor(attention_matrix)
            
            tensor_shape = attention_matrix.shape
            
            if len(tensor_shape) == 4:  # [batch_size, layers, seq_len, seq_len]
                max_seq_len = tensor_shape[-1]
                valid_row_indices = [idx for idx in save_row_index if idx < max_seq_len]
                valid_col_indices = [idx for idx in save_column_index if idx < max_seq_len]
                
                if valid_row_indices and valid_col_indices:
                    row_tensor = torch.tensor(valid_row_indices, device=attention_matrix.device)
                    col_tensor = torch.tensor(valid_col_indices, device=attention_matrix.device)
                    filtered_attention_matrix = attention_matrix[:, :, row_tensor][:, :, :, col_tensor]
                else:
                    filtered_attention_matrix = attention_matrix[:, :, :min(3, tensor_shape[-2]), :min(3, tensor_shape[-1])]
                    valid_row_indices = list(range(min(3, tensor_shape[-2])))
                    valid_col_indices = list(range(min(3, tensor_shape[-1])))
            elif len(tensor_shape) == 3:  # [layers, seq_len, seq_len]
                max_seq_len = tensor_shape[-1]
                valid_row_indices = [idx for idx in save_row_index if idx < max_seq_len]
                valid_col_indices = [idx for idx in save_column_index if idx < max_seq_len]
                
                if valid_row_indices and valid_col_indices:
                    row_tensor = torch.tensor(valid_row_indices, device=attention_matrix.device)
                    col_tensor = torch.tensor(valid_col_indices, device=attention_matrix.device)
                    filtered_attention_matrix = attention_matrix[:, row_tensor][:, :, col_tensor]
                else:
                    filtered_attention_matrix = attention_matrix[:, :min(3, tensor_shape[-2]), :min(3, tensor_shape[-1])]
                    valid_row_indices = list(range(min(3, tensor_shape[-2])))
                    valid_col_indices = list(range(min(3, tensor_shape[-1])))
            else:
                filtered_attention_matrix = attention_matrix
                valid_row_indices = save_row_index
                valid_col_indices = save_column_index
        else:
            filtered_attention_matrix = attention_matrix
            valid_row_indices = save_row_index
            valid_col_indices = save_column_index
        
        return filtered_attention_matrix, valid_row_indices, valid_col_indices
    
    def matrix_computation(self, filtered_attention_matrix, purpose, head: Union[int, str], layer: Union[int, str], phoneme_mean_map: dict):
        """Compute attention matrix based on purpose and parameters"""
        if not hasattr(filtered_attention_matrix, 'mean'):
            filtered_attention_matrix = torch.tensor(filtered_attention_matrix)
        
        tensor_shape = filtered_attention_matrix.shape
        
        if purpose == "flow":
            computed_matrix = None
            if phoneme_mean_map:
                # Compute attention flow based on phoneme-meaning mapping
                for phoneme, meaning in phoneme_mean_map.items():
                    for meaning_idx, meaning_value in enumerate(meaning):
                        for head_idx, head_value in enumerate(filtered_attention_matrix[meaning_idx]):
                            for layer_idx, layer_value in enumerate(head_value):
                                pass
            else:
                if len(tensor_shape) == 4:  # [head, layer, seq_len, seq_len]
                    if isinstance(head, str) and isinstance(layer, str):
                        computed_matrix = torch.mean(filtered_attention_matrix, dim=(0, 1))
                    elif isinstance(head, int) and isinstance(layer, int):
                        computed_matrix = filtered_attention_matrix[head, layer]
                    else:
                        computed_matrix = torch.mean(filtered_attention_matrix, dim=0)
                elif len(tensor_shape) == 3:  # [seq_len, seq_len, seq_len] or similar
                    computed_matrix = torch.mean(filtered_attention_matrix, dim=0)
                elif len(tensor_shape) == 2:  # [seq_len, seq_len]
                    computed_matrix = filtered_attention_matrix
                else:
                    computed_matrix = filtered_attention_matrix
        elif purpose == "heatmap":
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
                computed_matrix = torch.mean(filtered_attention_matrix, dim=0)
            elif len(tensor_shape) == 2:  # [seq_len, seq_len]
                computed_matrix = filtered_attention_matrix
            else:
                computed_matrix = filtered_attention_matrix
        
        return computed_matrix
    
    def read_matrix(self, layer_type="self", word_tokens=None, dimension1=None, dimension2=None, lang="en"):
        """Read matrix from pickle file"""
        if not all([word_tokens, dimension1, dimension2]):
            raise ValueError("word_tokens, dimension1, and dimension2 must be provided")
        
        # Create a safe filename
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dimension1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dimension2))
        
        matrix_path = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, f"{safe_word}_{safe_dim1}_{safe_dim2}_{layer_type}.pkl")
        
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
        
        # Load with map_location to handle CUDA tensors
        matrix_data = pkl.load(open(matrix_path, "rb"), map_location='cpu')
        attention_matrix = matrix_data["attention_matrix"]
        dimension1 = matrix_data["dimension1"]
        dimension2 = matrix_data["dimension2"]
        answer = matrix_data["answer"]
        word_tokens = matrix_data["word_tokens"]
        option_tokens = matrix_data["option_tokens"]
        tokens = matrix_data.get("tokens", None)  # Get tokens if available
        
        return attention_matrix, dimension1, dimension2, answer, word_tokens, option_tokens, tokens
    
    def plot_heatmap(self, attention_matrix, tokens, dimension1, dimension2, answer, word_tokens, option_tokens, data_type="audio", layer_type="self", head=0, layer=0, lang="en", save_path=None):
        """Plot attention heatmap for semantic dimension analysis"""
        set_font_for_language(lang)
        
        filtered_matrix, row_indices, col_indices = self.filter_relevant_indices(
            attention_matrix, tokens, tokens, word_tokens, option_tokens, dimension1, dimension2, answer, layer_type
        )
        
        computed_matrix = self.matrix_computation(filtered_matrix, "heatmap", head, layer, self.phoneme_mean_map)
        
        if hasattr(computed_matrix, 'cpu'):
            computed_matrix = computed_matrix.float().cpu().numpy()
        elif hasattr(computed_matrix, 'numpy'):
            computed_matrix = computed_matrix.numpy()
        
        # Create token labels
        if row_indices and col_indices:
            row_labels = [tokens[i].replace("Ġ", "") if tokens[i].startswith("Ġ") else tokens[i] for i in row_indices]
            col_labels = [tokens[i].replace("Ġ", "") if tokens[i].startswith("Ġ") else tokens[i] for i in col_indices]
        else:
            row_labels = [token.replace("Ġ", "") if token.startswith("Ġ") else token for token in tokens]
            col_labels = [token.replace("Ġ", "") if token.startswith("Ġ") else token for token in tokens]
        
        # Ensure matrix and labels have compatible dimensions
        if len(row_labels) != computed_matrix.shape[0] or len(col_labels) != computed_matrix.shape[1]:
            min_rows = min(len(row_labels), computed_matrix.shape[0])
            min_cols = min(len(col_labels), computed_matrix.shape[1])
            computed_matrix = computed_matrix[:min_rows, :min_cols]
            row_labels = row_labels[:min_rows]
            col_labels = col_labels[:min_cols]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(computed_matrix, xticklabels=col_labels, yticklabels=row_labels, 
                   cmap='Blues', annot=True, fmt='.3f')
        
        plt.title(f'Semantic Dimension Attention Heatmap\nWord: {word_tokens}, Dim1: {dimension1}, Dim2: {dimension2}, Answer: {answer}\nLayer: {layer}, Head: {head}, Type: {layer_type}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Save plot
        if save_path is None:
            os.makedirs(os.path.join(self.output_dir, self.exp_type), exist_ok=True)
            save_path = os.path.join(self.output_dir, f"semdim_{data_type}_{self.exp_type}_{layer_type}_layer{layer}_head{head}_{dimension1}_{dimension2}_{word_tokens}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to {save_path}")
    
    def plot_average_heatmap(self, avg_matrix, tokens, dimension1, dimension2, answer, word_tokens, option_tokens, data_type="audio", layer_type="self", lang="en", save_path=None):
        """Plot average attention heatmap across all layers"""
        set_font_for_language(lang)
        
        # Get token indices for labels
        word_indices = self.find_token_indices(tokens, [word_tokens])
        if self.data_type == "audio":
            word_indices = self.find_token_indices(tokens, ["<|AUDIO|>"])
        dim1_indices = self.find_token_indices(tokens, [dimension1])
        dim2_indices = self.find_token_indices(tokens, [dimension2])
        
        # Collect relevant indices
        relevant_indices = []
        for idx, token in enumerate(tokens):
            if idx in word_indices or idx in dim1_indices or idx in dim2_indices:
                relevant_indices.append(idx)
        
        # Create token labels
        if relevant_indices:
            row_labels = [tokens[i].replace("Ġ", "") if tokens[i].startswith("Ġ") else tokens[i] for i in relevant_indices]
            col_labels = [tokens[i].replace("Ġ", "") if tokens[i].startswith("Ġ") else tokens[i] for i in relevant_indices]
        else:
            row_labels = [token.replace("Ġ", "") if token.startswith("Ġ") else token for token in tokens]
            col_labels = [token.replace("Ġ", "") if token.startswith("Ġ") else token for token in tokens]
        
        # Ensure matrix and labels have compatible dimensions
        if len(row_labels) != avg_matrix.shape[0] or len(col_labels) != avg_matrix.shape[1]:
            min_rows = min(len(row_labels), avg_matrix.shape[0])
            min_cols = min(len(col_labels), avg_matrix.shape[1])
            avg_matrix = avg_matrix[:min_rows, :min_cols]
            row_labels = row_labels[:min_rows]
            col_labels = col_labels[:min_cols]
        
        # Create average heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_matrix, xticklabels=col_labels, yticklabels=row_labels, 
                   cmap='Blues', annot=True, fmt='.3f')
        
        plt.title(f'Semantic Dimension Average Attention Heatmap (All Layers)\nWord: {word_tokens}, Dim1: {dimension1}, Dim2: {dimension2}, Answer: {answer}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Save plot
        if save_path is None:
            os.makedirs(os.path.join(self.output_dir, self.data_type, lang), exist_ok=True)
            save_path = os.path.join(self.output_dir, self.data_type, lang, f"semdim_avg_heatmap_{data_type}_{layer_type}_{word_tokens}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved average heatmap to {save_path}")
    
    def plot_flow(self, attention_matrix, tokens, dimension1, dimension2, answer, word_tokens, option_tokens, data_type="audio", layer_type="self", lang="en", save_path=None):
        """Plot attention flow across layers"""
        set_font_for_language(lang)
        
        filtered_matrix, row_indices, col_indices = self.filter_relevant_indices(
            attention_matrix, tokens, tokens, word_tokens, option_tokens, dimension1, dimension2, answer, layer_type
        )
        
        flow_matrix = self.matrix_computation(filtered_matrix, "flow", "all", "all", self.phoneme_mean_map)
        
        if hasattr(flow_matrix, 'cpu'):
            flow_matrix = flow_matrix.float().cpu().numpy()
        elif hasattr(flow_matrix, 'numpy'):
            flow_matrix = flow_matrix.numpy()
        
        # Create flow plot
        plt.figure(figsize=(12, 8))
        
        if len(flow_matrix.shape) == 1:
            plt.plot(range(len(flow_matrix)), flow_matrix, marker='o', linewidth=2, markersize=6, 
                    label=f'Attention Score ({dimension1} vs {dimension2})')
        else:
            for i in range(flow_matrix.shape[0]):
                plt.plot(range(flow_matrix.shape[1]), flow_matrix[i], marker='o', linewidth=2, markersize=6,
                        label=f'Pattern {i+1}')
        
        plt.title(f'Semantic Dimension Attention Flow\nWord: {word_tokens}, Dim1: {dimension1}, Dim2: {dimension2}, Answer: {answer}')
        plt.xlabel('Attention Layer')
        plt.ylabel('Attention Score')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(len(flow_matrix) if len(flow_matrix.shape) == 1 else flow_matrix.shape[1]))
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"semdim_flow_{data_type}_{layer_type}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved flow plot to {save_path}")
    
    def plot_from_pkl(self, word_tokens, dimension1, dimension2, lang="en", layer_type="self", head=0, layer=0, data_type="audio"):
        """Load data from pkl file and create all visualizations"""
        try:
            # Read matrix from pkl file
            attention_matrix, dim1, dim2, answer, word_toks, option_toks, tokens = self.read_matrix(
                layer_type=layer_type, 
                word_tokens=word_tokens, 
                dimension1=dimension1, 
                dimension2=dimension2, 
                lang=lang
            )
            
            # Use tokens from pkl file if available, otherwise create dummy tokens
            if tokens is None:
                tokens = [word_tokens, dimension1, dimension2, answer]
            
            # Plot individual heatmap
            self.plot_heatmap(
                attention_matrix, tokens, dim1, dim2, answer, word_toks, option_toks,
                data_type=data_type, layer_type=layer_type, head=head, layer=layer, lang=lang
            )
            
            # Plot flow
            self.plot_flow(
                attention_matrix, tokens, dim1, dim2, answer, word_toks, option_toks,
                data_type=data_type, layer_type=layer_type, lang=lang
            )
            
            print(f"Successfully created visualizations for {word_tokens} - {dimension1}-{dimension2}")
            
        except Exception as e:
            print(f"Error creating visualizations for {word_tokens} - {dimension1}-{dimension2}: {e}")
    
    def list_available_pkl_files(self, lang="en"):
        """List all available pkl files for a given language"""
        pkl_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang)
        if not os.path.exists(pkl_dir):
            print(f"No pkl directory found: {pkl_dir}")
            return []
        
        pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
        print(f"Found {len(pkl_files)} pkl files in {pkl_dir}")
        return pkl_files
    
    def batch_plot_from_pkl(self, lang="en", layer_type="self", head=0, layer=0, data_type="audio"):
        """Batch plot from all available pkl files for a language"""
        pkl_files = self.list_available_pkl_files(lang)
        
        for pkl_file in tqdm(pkl_files, desc=f"Processing {lang} files"):
            try:
                # Parse filename to extract word_tokens, dimension1, dimension2
                parts = pkl_file.replace('.pkl', '').split('_')
                if len(parts) >= 4:
                    word_tokens = parts[0]
                    dimension1 = parts[1]
                    dimension2 = parts[2]
                    
                    self.plot_from_pkl(
                        word_tokens, dimension1, dimension2, lang, layer_type, head, layer, data_type
                    )
                else:
                    print(f"Could not parse filename: {pkl_file}")
                    
            except Exception as e:
                print(f"Error processing {pkl_file}: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Dimension Heatmap Plotter")
    parser.add_argument('--output-dir', type=str, 
                       default="results/experiments/understanding/attention_heatmap",
                       help="Output directory for heatmaps and matrices")
    parser.add_argument('--data-type', type=str, default="audio", 
                       choices=["audio", "original", "romanized", "ipa"],
                       help="Data type to process")
    parser.add_argument('--languages', nargs='+', default=["en", "fr", "ko", "ja"], 
                       help="Languages to process")
    parser.add_argument('--layer-type', type=str, default="self", 
                       choices=["self", "cross", "output"],
                       help="Layer type for attention analysis")
    parser.add_argument('--head', type=int, default=0, 
                       help="Attention head index")
    parser.add_argument('--layer', type=int, default=0, 
                       help="Layer index")
    parser.add_argument('--batch-mode', action='store_true', 
                       help="Process all available pkl files in batch mode")
    parser.add_argument('--word-tokens', type=str, default=None, 
                       help="Specific word tokens to plot")
    parser.add_argument('--dimension1', type=str, default=None, 
                       help="First dimension")
    parser.add_argument('--dimension2', type=str, default=None, 
                       help="Second dimension")
    parser.add_argument('--lang', type=str, default="en", 
                       help="Language for specific plotting")
    
    args = parser.parse_args()
    
    print(f"Initializing SemanticDimensionHeatmapPlotter...")
    print(f"Output directory: {args.output_dir}")
    print(f"Data type: {args.data_type}")
    print(f"Languages: {args.languages}")
    
    # Initialize plotter
    plotter = SemanticDimensionHeatmapPlotter(
        output_dir=args.output_dir,
        exp_type="semantic_dimension",
        data_type=args.data_type
    )
    
    if args.batch_mode:
        # Batch process all available pkl files
        for lang in args.languages:
            print(f"\nProcessing language: {lang}")
            plotter.batch_plot_from_pkl(
                lang=lang, 
                layer_type=args.layer_type, 
                head=args.head, 
                layer=args.layer, 
                data_type=args.data_type
            )
    else:
        # Process specific word and dimensions
        if args.word_tokens and args.dimension1 and args.dimension2:
            plotter.plot_from_pkl(
                word_tokens=args.word_tokens,
                dimension1=args.dimension1,
                dimension2=args.dimension2,
                lang=args.lang,
                layer_type=args.layer_type,
                head=args.head,
                layer=args.layer,
                data_type=args.data_type
            )
        else:
            print("Please provide --word-tokens, --dimension1, and --dimension2 for specific plotting")
            print("Or use --batch-mode to process all available files")
    
    print(f"\nPlotting completed!")
    print(f"Results saved to: {args.output_dir}") 