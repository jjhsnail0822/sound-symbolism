# Model : Qwen2.5-Omni-7B
import json
import re
import os
import argparse
import pickle as pkl
from typing import Union
import warnings
# Suppress matplotlib UserWarnings globally
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import seaborn as sns
import numpy as np
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from tqdm import tqdm
from qwen_omni_utils import process_mm_info
def set_font_for_language(lang):
    """Set appropriate font based on language with fallback system"""
    import os
    
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
        # Only print if it's not the default fallback
        if selected_font != "DejaVu Sans":
            print(f"Set font for language '{lang}' to: {selected_font}")
    else:
        # Ultimate fallback
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print(f"Warning: No suitable font found for language '{lang}', using default sans-serif")
    
    # Enable Unicode support and set font properties
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.sans-serif'] = [selected_font] if selected_font else ['sans-serif']
language = ["en", "fr", "ko", "ja"]
data_types = ["original", "romanized", "ipa", "audio"]
data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"
prompt_path = "data/prompts/prompts.json"
problem_per_language = 10
phoneme_mean_map_path = None # TODO
phoneme_mean_map = None # TODO
data_type = "audio" # Change into argparse later, with "audio", "original", "romanized", "ipa"
with open(prompt_path, "r") as f:
    prompts = json.load(f)
class QwenOmniSemanticDimensionVisualizer:
    def __init__(
            self,
            model_path:str,
            data_path:str="data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json",
            output_dir:str="results/experiments/understanding/attention_heatmap",
            exp_type:str="semantic_dimension",
            data_type:str="audio",
            max_tokens:int=32,
            temperature:float=0.0,
        ):
        # print(f"Starting __init__ function")
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.data_type = data_type
        self.exp_type = exp_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
        )
        self.model.disable_talker()
        
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        self.data = self.load_data()
        self.load_base_prompt()
        # print(f"Ending __init__ function")
    
    def load_data(self):
        # print(f"Starting load_data function")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        # print(f"Ending load_data function")
        return self.data
    
    def load_base_prompt(self):
        # print(f"Starting load_base_prompt function")
        self.prompts = prompts[self.exp_type][f"semantic_dimension_binary_{self.data_type}"]["user_prompt"]
        # print(f"Ending load_base_prompt function")
        return self.prompts
    
    def prmpt_dims_answrs(self, prompt, data, dimension_name=None):
        # print(f"Starting prmpt_dims_answrs function")
        if self.data_type == "audio":
            audio_path = f"data/processed/nat/tts/{data['language']}/{data['word']}.wav"
            # Check if audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            word = audio_path
        else:
            data_type_key = "word" if self.data_type == "original" \
                else "romanization" if self.data_type == "romanized" \
                else "ipa"
            if data_type_key not in data:
                raise KeyError(f"Data type '{data_type_key}' not found in data: {data.keys()}")
            word = data[data_type_key]
            
        # Extract dimension information
        dimension_info = data["dimensions"][dimension_name]
        dimension1 = dimension_name.split("-")[0]
        dimension2 = dimension_name.split("-")[1]
        answer = dimension_info["answer"]
        
        # Use the correct word value for prompt formatting
        constructed_prompt = prompt.format(
            word=word,
            dimension1=dimension1,
            dimension2=dimension2,
        )
        # print(f"Ending prmpt_dims_answrs function")
        return constructed_prompt, dimension1, dimension2, answer, word, dimension_name
    
    def get_attention_matrix(self, prompt, data):
        # print(f"Starting get_attention_matrix function")
        """Extract attention matrix from the model"""
        # Create conversation format
        if self.data_type == "audio":
            audio_path = f'data/processed/nat/tts/{data["language"]}/{data["word"]}.wav'
            # Check if audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Split prompt to insert audio in the middle (similar to MCQ format)
            if "<AUDIO>" in prompt:
                # If prompt already has AUDIO placeholder
                question_parts = prompt.split("<AUDIO>")
                if len(question_parts) == 2:
                    question_first_part = question_parts[0]
                    question_second_part = question_parts[1]
                else:
                    # If no AUDIO placeholder, insert audio after the word
                    word_placeholder = "{word}"
                    if word_placeholder in prompt:
                        parts = prompt.split(word_placeholder)
                        question_first_part = parts[0] + data["word"]
                        question_second_part = parts[1] if len(parts) > 1 else ""
                    else:
                        question_first_part = prompt
                        question_second_part = ""
            else:
                # Insert audio after the word
                word_placeholder = "{word}"
                if word_placeholder in prompt:
                    parts = prompt.split(word_placeholder)
                    question_first_part = parts[0] + data["word"]
                    question_second_part = parts[1] if len(parts) > 1 else ""
                else:
                    question_first_part = prompt
                    question_second_part = ""
            
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_first_part},
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": question_second_part},
                    ],
                },
            ]
        else:
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        
        USE_AUDIO_IN_VIDEO = True
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        # Get attention matrices
        with torch.no_grad():
            thinker_model = self.model.thinker.model
            outputs = thinker_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_attentions=True,
                return_dict=True
            )
        
        attentions = outputs.attentions
        tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # print(f"Ending get_attention_matrix function")
        return attentions, tokens, inputs
    
    def save_matrix(self, attention_matrix, dimension1, dimension2, answer, word_tokens, option_tokens, layer_type="self", lang="en"):
        # print(f"Starting save_matrix function")
        """Save matrix as pickle file
        # Composition
        - Matrix array
        - dimension1, dimension2, answer
        - corresponding word tokens with index
        - corresponding option tokens with index
        """
        matrix_data = {
            "attention_matrix": attention_matrix,
            "dimension1": dimension1,
            "dimension2": dimension2,
            "answer": answer,
            "word_tokens": word_tokens,
            "option_tokens": option_tokens
        }
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a safe filename
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dimension1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dimension2))
        
        save_path = os.path.join(output_dir, f"{safe_word}_{safe_dim1}_{safe_dim2}_{layer_type}.pkl")
        with open(save_path, "wb") as f:
            pkl.dump(matrix_data, f)
        # print(f"Saved attention matrix to {save_path}")
        # print(f"Ending save_matrix function")
    
    def read_matrix(self, layer_type="self", word_tokens=None, dimension1=None, dimension2=None, lang="en"):
        # print(f"Starting read_matrix function")
        """Read matrix from pickle file
        # Composition
        - Matrix array
        - dimension1, dimension2, answer
        - corresponding word tokens with index
        - corresponding option tokens with index
        """
        if not all([word_tokens, dimension1, dimension2]):
            raise ValueError("word_tokens, dimension1, and dimension2 must be provided")
        
        # Create a safe filename
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dimension1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dimension2))
        
        matrix_path = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, f"{safe_word}_{safe_dim1}_{safe_dim2}_{layer_type}.pkl")
        
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
        
        attention_matrix = pkl.load(open(matrix_path, "rb"))
        dimension1 = attention_matrix["dimension1"]
        dimension2 = attention_matrix["dimension2"]
        answer = attention_matrix["answer"]
        word_tokens = attention_matrix["word_tokens"]
        option_tokens = attention_matrix["option_tokens"]
        # print(f"Ending read_matrix function")
        return attention_matrix, dimension1, dimension2, answer, word_tokens, option_tokens
    
    def find_token_indices(self, tokens, target_tokens):
        # print(f"Starting find_token_indices function")
        """Find indices of target tokens in the token list
        
        Args:
            tokens: List of tokens from the model
            target_tokens: List of target tokens to find (can be multi-token sequences)
        
        Returns:
            List of indices where target tokens are found
        """
        indices = []
        
        # Helper function to clean token (remove "Ġ" and check if it's a special token)
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
            # If target is a single token
            if isinstance(target, str):
                # Clean the target token
                clean_target = clean_token(target)
                
                # Find all occurrences of this target
                for idx, token in enumerate(tokens):
                    clean_token_str = clean_token(token)
                    if clean_target == clean_token_str:
                        indices.append(idx)
            
            # If target is a list of tokens (multi-token sequence)
            elif isinstance(target, list):
                # Clean all target tokens
                clean_targets = [clean_token(t) for t in target]
                
                # Find continuous sequence of target tokens
                for i in range(len(tokens) - len(clean_targets) + 1):
                    match_found = True
                    sequence_indices = []
                    
                    # Check if the sequence starting at position i matches our target
                    for j, clean_target in enumerate(clean_targets):
                        current_token = tokens[i + j]
                        clean_current = clean_token(current_token)
                        
                        # If this token doesn't match, or if it's a special token in the middle, break
                        if clean_current != clean_target or (j > 0 and is_special_token(current_token)):
                            match_found = False
                            break
                        sequence_indices.append(i + j)
                    
                    # If we found a complete match, add all indices
                    if match_found:
                        indices.extend(sequence_indices)
                        break  # Only take the first occurrence
        
        # Remove duplicates and sort
        indices = sorted(list(set(indices)))
        
        # print(f"Ending find_token_indices function")
        return indices
    
    def filter_relevant_indices(
        self,
        attention_matrix,
        row_tokens,
        column_tokens,
        word_tokens,
        option_tokens,
        dimension1,
        dimension2,
        answer,
        layer_type="self"
    ):
        # print(f"Starting filter_relevant_indices function")
        
        # Must include the file index of word token, dim1, dim2
        save_row_index = []
        save_column_index = []
        
        # Find word token indices
        word_indices = self.find_token_indices(row_tokens, [word_tokens])
        if self.data_type == "audio":
            word_indices = self.find_token_indices(row_tokens, ["<|AUDIO|>"])
        dim1_indices = self.find_token_indices(row_tokens, [dimension1])
        dim2_indices = self.find_token_indices(row_tokens, [dimension2])
        
        if layer_type == "self":
            # Find the index of word token, dim1, dim2
            for idx, token in enumerate(row_tokens):
                if idx in word_indices or idx in dim1_indices or idx in dim2_indices:
                    save_row_index.append(idx)
                    
            for idx, token in enumerate(column_tokens):
                if idx in word_indices or idx in dim1_indices or idx in dim2_indices:
                    save_column_index.append(idx)
                    
        elif layer_type == "cross":
            # Find the index of word token, dim1, dim2
            for idx, token in enumerate(row_tokens):
                if idx in word_indices or idx in dim1_indices or idx in dim2_indices:
                    save_row_index.append(idx)
                    
            for idx, token in enumerate(column_tokens):
                if idx in option_tokens or idx in dim1_indices or idx in dim2_indices:
                    save_column_index.append(idx)
        
        elif layer_type == "output":
            # Find the index of word token, dim1, dim2
            answer_indices = self.find_token_indices(row_tokens, [answer])
            for idx, token in enumerate(row_tokens):
                if idx in answer_indices or idx in dim1_indices or idx in dim2_indices:
                    save_row_index.append(idx)
                    
            for idx, token in enumerate(column_tokens):
                if idx in option_tokens or idx in dim1_indices or idx in dim2_indices:
                    save_column_index.append(idx)
        # Inside the attention matrix, remove the rows and columns that are not in the index
        if save_row_index and save_column_index:
            # Handle the case where attention_matrix is a tuple of attention tensors
            if isinstance(attention_matrix, tuple):
                # print(f"Debug - attention_matrix is a tuple with {len(attention_matrix)} elements")
                # For tuple of attention tensors, we need to process each layer
                # For now, let's take the first layer as an example
                attention_matrix = attention_matrix[0]  # Take first layer
            
            # Convert to tensor if it's not already
            if not hasattr(attention_matrix, 'index_select'):
                attention_matrix = torch.tensor(attention_matrix)
            
            # Handle different tensor dimensions
            tensor_shape = attention_matrix.shape
            
            # Check if indices are within bounds
            if len(tensor_shape) == 4:  # [batch_size, layers, seq_len, seq_len]
                max_seq_len = tensor_shape[-1]  # Last dimension is sequence length
                
                # Filter indices to be within bounds
                valid_row_indices = [idx for idx in save_row_index if idx < max_seq_len]
                valid_col_indices = [idx for idx in save_column_index if idx < max_seq_len]
                
                if valid_row_indices and valid_col_indices:
                    # For 4D tensor [batch_size, layers, seq_len, seq_len], filter the last two dimensions
                    # Convert indices to tensors on the same device as attention_matrix
                    row_tensor = torch.tensor(valid_row_indices, device=attention_matrix.device)
                    col_tensor = torch.tensor(valid_col_indices, device=attention_matrix.device)
                    
                    # Use advanced indexing to properly filter the attention matrix
                    # [batch_size, layers, seq_len, seq_len] -> [batch_size, layers, filtered_seq_len, filtered_seq_len]
                    filtered_attention_matrix = attention_matrix[:, :, row_tensor][:, :, :, col_tensor]
                else:
                    # If no valid indices, return a small subset of the original matrix
                    print(f"Warning: No valid indices found, using first few tokens")
                    filtered_attention_matrix = attention_matrix[:, :, :min(3, tensor_shape[-2]), :min(3, tensor_shape[-1])]
                    valid_row_indices = list(range(min(3, tensor_shape[-2])))
                    valid_col_indices = list(range(min(3, tensor_shape[-1])))
                    
            elif len(tensor_shape) == 3:  # [layers, seq_len, seq_len]
                max_seq_len = tensor_shape[-1]  # Last dimension is sequence length
                # Filter indices to be within bounds
                valid_row_indices = [idx for idx in save_row_index if idx < max_seq_len]
                valid_col_indices = [idx for idx in save_column_index if idx < max_seq_len]
                
                if valid_row_indices and valid_col_indices:
                    # For 3D tensor [layers, seq_len, seq_len], filter the last two dimensions
                    # Convert indices to tensors on the same device as attention_matrix
                    row_tensor = torch.tensor(valid_row_indices, device=attention_matrix.device)
                    col_tensor = torch.tensor(valid_col_indices, device=attention_matrix.device)
                    
                    # Use advanced indexing to properly filter the attention matrix
                    # [layers, seq_len, seq_len] -> [layers, filtered_seq_len, filtered_seq_len]
                    filtered_attention_matrix = attention_matrix[:, row_tensor][:, :, col_tensor]
                else:
                    # If no valid indices, return a small subset of the original matrix
                    # print(f"Warning: No valid indices found, using first few tokens")
                    filtered_attention_matrix = attention_matrix[:, :min(3, tensor_shape[-2]), :min(3, tensor_shape[-1])]
                    valid_row_indices = list(range(min(3, tensor_shape[-2])))
                    valid_col_indices = list(range(min(3, tensor_shape[-1])))
            else:
                # For other dimensions, just return the original
                filtered_attention_matrix = attention_matrix
                valid_row_indices = save_row_index
                valid_col_indices = save_column_index
        else:
            filtered_attention_matrix = attention_matrix
            valid_row_indices = save_row_index
            valid_col_indices = save_column_index
        # print(f"Ending filter_relevant_indices function")
        return filtered_attention_matrix, valid_row_indices, valid_col_indices
           
    def matrix_computation(
        self,
        filtered_attention_matrix,
        purpose,
        head:Union[int, str],
        layer:Union[int, str],
        phoneme_mean_map:dict
    ):
        # Convert to tensor if it's not already
        if not hasattr(filtered_attention_matrix, 'mean'):
            filtered_attention_matrix = torch.tensor(filtered_attention_matrix)
        
        # Check tensor dimensions and handle accordingly
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
                # For 3D tensor, just take the mean across the first dimension
                computed_matrix = torch.mean(filtered_attention_matrix, dim=0)
            elif len(tensor_shape) == 2:  # [seq_len, seq_len]
                computed_matrix = filtered_attention_matrix
            else:
                # For 1D or other dimensions, just use as is
                computed_matrix = filtered_attention_matrix
        return computed_matrix
    
    def plot_heatmap(self, attention_matrix, tokens, dimension1, dimension2, answer, word_tokens, option_tokens, data_type="audio", layer_type="self", head=0, layer=0, lang="en"):
        """Plot attention heatmap for semantic dimension analysis"""
        
        # Set font based on language
        set_font_for_language(lang)
        
        # Filter relevant indices
        filtered_matrix, row_indices, col_indices = self.filter_relevant_indices(
            attention_matrix, tokens, tokens, word_tokens, option_tokens, dimension1, dimension2, answer, layer_type
        )
        # Compute matrix based on purpose
        computed_matrix = self.matrix_computation(filtered_matrix, "heatmap", head, layer, phoneme_mean_map)
        # Convert to CPU numpy array if it's a tensor
        if hasattr(computed_matrix, 'cpu'):
            # Convert to float32 first to handle BFloat16
            computed_matrix = computed_matrix.float().cpu().numpy()
        elif hasattr(computed_matrix, 'numpy'):
            computed_matrix = computed_matrix.numpy()
        
        # Create token labels
        if row_indices and col_indices:
            # Remove "Ġ" from token labels
            row_labels = [tokens[i].replace("Ġ", "") if tokens[i].startswith("Ġ") else tokens[i] for i in row_indices]
            col_labels = [tokens[i].replace("Ġ", "") if tokens[i].startswith("Ġ") else tokens[i] for i in col_indices]
        else:
            # Remove "Ġ" from all token labels
            row_labels = [token.replace("Ġ", "") if token.startswith("Ġ") else token for token in tokens]
            col_labels = [token.replace("Ġ", "") if token.startswith("Ġ") else token for token in tokens]
        # Ensure matrix and labels have compatible dimensions
        if len(row_labels) != computed_matrix.shape[0] or len(col_labels) != computed_matrix.shape[1]:
            # Adjust matrix size to match labels
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
        # Remove X-axis rotation
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Save plot
        os.makedirs(os.path.join(self.output_dir, self.exp_type), exist_ok=True)
        save_path = os.path.join(self.output_dir, f"semdim_{data_type}_{self.exp_type}_{layer_type}_layer{layer}_head{head}_{dimension1}_{dimension2}_{word_tokens}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to {save_path}")
        # print(f"Ending plot_heatmap function")
    
    def plot_average_heatmap(self, avg_matrix, tokens, dimension1, dimension2, answer, word_tokens, option_tokens, data_type="audio", layer_type="self", lang="en"):
        """Plot average attention heatmap across all layers"""
        # Set font based on language
        set_font_for_language(lang)
        
        # Get token indices for labels (use the same filtering logic as plot_heatmap)
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
        
        # Create token labels (remove "Ġ" if present)
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
        # Remove X-axis rotation
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Save plot
        os.makedirs(os.path.join(self.output_dir, self.data_type, lang), exist_ok=True)
        save_path = os.path.join(self.output_dir, self.data_type, lang, f"semdim_avg_heatmap_{data_type}_{layer_type}_{word_tokens}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved average heatmap to {save_path}")
    
    def plot_flow(self, attention_matrix, tokens, dimension1, dimension2, answer, word_tokens, option_tokens, data_type="audio", layer_type="self", lang="en"):
        # print(f"Starting plot_flow function")
        """Plot attention flow across layers"""
        
        # Set font based on language
        set_font_for_language(lang)
        
        # Filter relevant indices
        filtered_matrix, row_indices, col_indices = self.filter_relevant_indices(
            attention_matrix, tokens, tokens, word_tokens, option_tokens, dimension1, dimension2, answer, layer_type
        )
        
        # Compute flow matrix
        flow_matrix = self.matrix_computation(filtered_matrix, "flow", "all", "all", phoneme_mean_map)
        
        # Convert to CPU numpy array if it's a tensor
        if hasattr(flow_matrix, 'cpu'):
            # print(f"Debug - Converting flow tensor to CPU numpy array")
            # Convert to float32 first to handle BFloat16
            flow_matrix = flow_matrix.float().cpu().numpy()
        elif hasattr(flow_matrix, 'numpy'):
            # print(f"Debug - Converting flow to numpy array")
            flow_matrix = flow_matrix.numpy()
        
        # Create flow plot
        plt.figure(figsize=(12, 8))
        
        # Plot attention flow across all layers
        if len(flow_matrix.shape) == 1:
            # Single line plot
            plt.plot(range(len(flow_matrix)), flow_matrix, marker='o', linewidth=2, markersize=6, 
                    label=f'Attention Score ({dimension1} vs {dimension2})')
        else:
            # Multiple lines plot (if we have multiple attention patterns)
            for i in range(flow_matrix.shape[0]):
                plt.plot(range(flow_matrix.shape[1]), flow_matrix[i], marker='o', linewidth=2, markersize=6,
                        label=f'Pattern {i+1}')
        
        plt.title(f'Semantic Dimension Attention Flow\nWord: {word_tokens}, Dim1: {dimension1}, Dim2: {dimension2}, Answer: {answer}')
        plt.xlabel('Attention Layer')
        plt.ylabel('Attention Score')
        plt.ylim(0, 1.0)  # Set Y-axis maximum to 1.0
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set X-axis to show integer values for all layers
        plt.xticks(range(len(flow_matrix) if len(flow_matrix.shape) == 1 else flow_matrix.shape[1]))
        
        # Save plot
        save_path = os.path.join(self.output_dir, f"semdim_flow_{data_type}_{layer_type}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved flow plot to {save_path}")
        # print(f"Ending plot_flow function")
    
    def inference_with_hooks(self, word, lang, constructed_prompt, dim1, dim2, answer, data, dimension_name):
        # print(f"Starting inference_with_hooks function")
        """Main inference function with attention extraction and visualization"""
        # print(f"Processing word: {data['word']}, language: {data['language']}")
        # print(f"Constructed prompt: {constructed_prompt}")
        # print(f"Dimension1: {dim1}, Dimension2: {dim2}, Answer: {answer}")
        # print(f"Dimension Name: {dimension_name}")
        # Get attention matrix
        attentions, tokens, inputs = self.get_attention_matrix(constructed_prompt, data)
        # Extract word and option tokens
        word_tokens = data['word']
        option_tokens = [dim1, dim2]
        # Save attention matrix
        self.save_matrix(attentions, dim1, dim2, answer, word_tokens, option_tokens, "self", lang)
        # Generate heatmaps for all layers and calculate average
        num_layers = len(attentions)
        all_layer_matrices = []
        # Process all layers for heatmaps
        for layer in range(num_layers):
            # Get attention matrix for this layer
            # attentions[layer] shape: [batch, seq_len, seq_len] or [seq_len, seq_len]
            layer_attention = attentions[layer][0] if attentions[layer].ndim == 3 else attentions[layer]
            # Filter and compute matrix for this layer
            filtered_matrix, row_indices, col_indices = self.filter_relevant_indices(
                layer_attention, tokens, tokens, word_tokens, option_tokens, dim1, dim2, answer, "self"
            )
            computed_matrix = self.matrix_computation(filtered_matrix, "heatmap", 0, layer, phoneme_mean_map)
            # Convert to numpy for averaging
            if hasattr(computed_matrix, 'cpu'):
                computed_matrix_np = computed_matrix.float().cpu().numpy()
            elif hasattr(computed_matrix, 'numpy'):
                computed_matrix_np = computed_matrix.numpy()
            else:
                computed_matrix_np = np.array(computed_matrix)
            all_layer_matrices.append(computed_matrix_np)
            # Plot heatmap for this layer
            self.plot_heatmap(
                layer_attention, tokens, dim1, dim2, answer, word_tokens, option_tokens,
                data_type=self.data_type, layer_type="self", head=0, layer=layer, lang=lang
            )
        # Calculate and plot average heatmap
        if all_layer_matrices:
            avg_matrix = np.mean(np.stack(all_layer_matrices), axis=0)
            self.plot_average_heatmap(
                avg_matrix, tokens, dim1, dim2, answer, word_tokens, option_tokens,
                data_type=self.data_type, layer_type="self", lang=lang
            )
        # Plot attention flow across layers
        self.plot_flow(
            attentions, tokens, dim1, dim2, answer, word_tokens, option_tokens,
            data_type=self.data_type, layer_type="self", lang=lang
        )
        # print(f"Ending inference_with_hooks function")
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Semantic Dimension Attention Heatmap Visualization")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Omni-7B", 
                       help="Model path (default: Qwen/Qwen2.5-Omni-7B)")
    parser.add_argument('--data-path', type=str, 
                       default="data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json",
                       help="Path to semantic dimension data JSON file")
    parser.add_argument('--output-dir', type=str, 
                       default="results/experiments/understanding/attention_heatmap",
                       help="Output directory for heatmaps and matrices")
    parser.add_argument('--data-type', type=str, default="audio", 
                       choices=["audio", "original", "romanized", "ipa"],
                       help="Data type to process")
    parser.add_argument('--max-tokens', type=int, default=32, 
                       help="Maximum tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.0, 
                       help="Sampling temperature")
    parser.add_argument('--max-samples', type=int, default=None, 
                       help="Maximum number of samples to process (default: all)")
    parser.add_argument('--languages', nargs='+', default=["en", "fr", "ko", "ja"], 
                       help="Languages to process")
    
    args = parser.parse_args()
    
    print(f"Initializing QwenOmniSemanticDimensionVisualizer...")
    print(f"Model: {args.model}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data type: {args.data_type}")
    print(f"Languages: {args.languages}")
    
    # Initialize visualizer
    visualizer = QwenOmniSemanticDimensionVisualizer(
        model_path=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        exp_type="semantic_dimension",
        data_type=args.data_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from data file")
    
    # Get all unique dimension keys
    all_dimension_keys = set()
    for sample in data:
        if "dimensions" in sample:
            all_dimension_keys.update(sample["dimensions"].keys())
    
    print(f"Found {len(all_dimension_keys)} unique dimension keys: {sorted(all_dimension_keys)}")
    
    # Process samples
    processed_count = 0
    for lang in args.languages:
        print(f"\nProcessing language: {lang}")
        
        # Filter samples for this language
        lang_samples = [sample for sample in data if sample.get("language") == lang]
        print(f"Found {len(lang_samples)} samples for language {lang}")
        
        if args.max_samples:
            lang_samples = lang_samples[:args.max_samples]
            print(f"Limiting to {len(lang_samples)} samples")
        
        for sample_idx, sample in enumerate(tqdm(lang_samples, desc=f"Processing {lang}")):
            try:
                # Process each dimension for this sample
                for dimension_name in sample.get("dimensions", {}):
                    print(f"\nProcessing sample {sample_idx + 1}/{len(lang_samples)} - {sample['word']} - {dimension_name}")
                    
                    # Construct prompt and get dimension info
                    constructed_prompt, dim1, dim2, answer, word, dim_name = visualizer.prmpt_dims_answrs(
                        visualizer.prompts, sample, dimension_name
                    )
                    
                    # Run inference with hooks
                    visualizer.inference_with_hooks(
                        word, lang, constructed_prompt, dim1, dim2, answer, sample, dimension_name
                    )
                    
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing sample {sample_idx} ({sample.get('word', 'unknown')}): {e}")
                continue
    
    print(f"\nProcessing completed!")
    print(f"Total samples processed: {processed_count}")
    print(f"Results saved to: {args.output_dir}")
    
    # Clean up
    del visualizer.model
    del visualizer.processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
