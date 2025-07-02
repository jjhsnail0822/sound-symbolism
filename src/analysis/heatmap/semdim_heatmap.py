# Model : Qwen2.5-Omni-7B
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 5000 --languages en
import json
import re
import os
import argparse
import pickle as pkl
from typing import Union
import warnings
import numpy as np
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch
from tqdm import tqdm
from qwen_omni_utils import process_mm_info

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
    
    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data
    
    def load_base_prompt(self):
        self.prompts = prompts[self.exp_type][f"semantic_dimension_binary_{self.data_type}"]["user_prompt"]
        return self.prompts
    
    def prmpt_dims_answrs(self, prompt:str, data, dimension_name:str=None):
        if self.data_type == "audio":
            audio_path = f"data/processed/nat/tts/{data['language']}/{data['word']}.wav"
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
        dimension_info = data["dimensions"][dimension_name]
        dimension1 = dimension_name.split("-")[0]
        dimension2 = dimension_name.split("-")[1]
        answer = dimension_info["answer"]
        constructed_prompt = prompt.format(
            word=word,
            dimension1=dimension1,
            dimension2=dimension2,
        )
        # print(f"[Prompt] {constructed_prompt}")
        return constructed_prompt, dimension1, dimension2, answer, word, dimension_name
    
    def get_attention_matrix(self, prompt:str, data:dict):
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
        
        return attentions, tokens, inputs
    
    def _clean_token(self, token):
        # Remove leading/trailing special characters and punctuation (Ġ, Ċ, [, ], ,, ., :, ;, !, ?, \n, \r, \t)
        return re.sub(r'^[ĠĊ\[\],.:;!?\n\r\t]+|[ĠĊ\[\],.:;!?\n\r\t]+$', '', token)

    def find_subtoken_sequence_indices(self, tokens, target_subtokens):
        cleaned_tokens = [self._clean_token(t) for t in tokens]
        cleaned_target = [self._clean_token(t) for t in target_subtokens]
        matches = []
        for i in range(len(cleaned_tokens) - len(cleaned_target) + 1):
            if cleaned_tokens[i:i+len(cleaned_target)] == cleaned_target:
                matches.append(list(range(i, i+len(cleaned_target))))
        return matches

    def find_tag_spans(self, tokens, tag_string, max_window=5):
        """
        tokens: list of tokens
        tag_string: e.g. 'WORD', 'SEMANTICDIMENSION', 'OPTIONS'
        Returns: list of [start, end] (inclusive) index spans where the tag appears
        """
        cleaned_tokens = [self._clean_token(t) for t in tokens]
        tag_string = tag_string.replace(" ", "").upper()
        matches = []
        for window in range(1, max_window+1):
            for i in range(len(cleaned_tokens) - window + 1):
                window_str = ''.join(cleaned_tokens[i:i+window]).replace(" ", "").upper()
                if window_str == tag_string:
                    matches.append(list(range(i, i+window)))
        return matches

    def extract_relevant_token_indices(self, tokens, dimension1, dimension2, word=None):
        # [WORD] tag
        word_tag_matches = self.find_tag_spans(tokens, 'WORD')
        if len(word_tag_matches) < 2:
            raise ValueError(f"[WORD] tag appears less than twice in tokens: {tokens}")
        second_word_tag_span = word_tag_matches[1]
        # [SEMANTIC DIMENSION] tag
        semdim_tag_matches = self.find_tag_spans(tokens, 'SEMANTICDIMENSION')
        if not semdim_tag_matches:
            raise ValueError(f"[SEMANTIC DIMENSION] tag not found in tokens: {tokens}")
        semdim_span = semdim_tag_matches[0]
        # [OPTIONS] tag
        options_tag_matches = self.find_tag_spans(tokens, 'OPTIONS')
        options_span = options_tag_matches[0] if options_tag_matches else None
        # 실제 word (예: 'a', 'banana' 등) subtoken 시퀀스 인덱스
        word_indices = []
        if word is not None:
            word_subtokens = self.processor.tokenizer.tokenize(word)
            word_matches = self.find_subtoken_sequence_indices(tokens, word_subtokens)
            for match in word_matches:
                word_indices.extend(match)
        # word span: 두 번째 [WORD] 다음부터 [SEMANTIC DIMENSION] 전까지
        word_span_start = second_word_tag_span[-1] + 1
        word_span_end = semdim_span[0]
        word_span = list(range(word_span_start, word_span_end))
        # [SEMANTIC DIMENSION] 이후 {dimension1}, {dimension2} 인덱스 (subword 포함)
        search_start = semdim_span[-1] + 1
        search_end = options_span[0] if options_span else len(tokens)
        dim1_indices = []
        dim2_indices = []
        dim1_subtokens = self.processor.tokenizer.tokenize(dimension1)
        for i in range(search_start, search_end - len(dim1_subtokens) + 1):
            if [self._clean_token(t) for t in tokens[i:i+len(dim1_subtokens)]] == [self._clean_token(t) for t in dim1_subtokens]:
                dim1_indices.extend(list(range(i, i+len(dim1_subtokens))))
        dim2_subtokens = self.processor.tokenizer.tokenize(dimension2)
        for i in range(search_start, search_end - len(dim2_subtokens) + 1):
            if [self._clean_token(t) for t in tokens[i:i+len(dim2_subtokens)]] == [self._clean_token(t) for t in dim2_subtokens]:
                dim2_indices.extend(list(range(i, i+len(dim2_subtokens))))
        dim1_indices = sorted(set(dim1_indices))
        dim2_indices = sorted(set(dim2_indices))
        relevant_indices = sorted(set(word_span + dim1_indices + dim2_indices + word_indices))
        return relevant_indices, word_span, dim1_indices, dim2_indices, word_indices
    
    def save_matrix(self, attention_matrix, dimension1, dimension2, answer, word_tokens, option_tokens, layer_type="self", lang="en", tokens=None, relevant_indices=None):
        """Save matrix as pickle file
        # Composition
        - Matrix array
        - dimension1, dimension2, answer
        - corresponding word tokens with index
        - corresponding option tokens with index
        - tokens (for visualization)
        - relevant_indices (추가)
        """
        matrix_data = {
            "attention_matrix": attention_matrix,
            "dimension1": dimension1,
            "dimension2": dimension2,
            "answer": answer,
            "word_tokens": word_tokens,
            "option_tokens": option_tokens,
            "tokens": tokens,
            "relevant_indices": relevant_indices
        }
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang)
        os.makedirs(output_dir, exist_ok=True)
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dimension1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dimension2))
        save_path = os.path.join(output_dir, f"{safe_word}_{safe_dim1}_{safe_dim2}_{layer_type}.pkl")
        with open(save_path, "wb") as f:
            pkl.dump(matrix_data, f)
    
    def read_matrix(self, layer_type="self", word_tokens=None, dimension1=None, dimension2=None, lang="en"):
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
        return attention_matrix, dimension1, dimension2, answer, word_tokens, option_tokens
    
    def find_token_indices(self, tokens, target_tokens):
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
    
    def inference_with_hooks(self, word, lang, constructed_prompt, dim1, dim2, answer, data, dimension_name):
        attentions, tokens, inputs = self.get_attention_matrix(constructed_prompt, data)
        relevant_indices, word_span, dim1_indices, dim2_indices, word_indices = self.extract_relevant_token_indices(tokens, dim1, dim2, word=data['word'])
        if isinstance(attentions, tuple):
            attention_matrix = attentions[0]
        else:
            attention_matrix = attentions
        attn_filtered = attention_matrix[:, :, relevant_indices][:, :, :, relevant_indices]
        self.save_matrix(attn_filtered, dim1, dim2, answer, data['word'], [dim1, dim2], "self", lang, tokens, relevant_indices)
        print(f"Saved filtered attention matrix for {data['word']} - {dim1}-{dim2} to pickle file")

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
    languages = ["en", "fr", "ko", "ja"]
    total_num_of_dimensions = 0
    total_num_of_words = 0
    total_num_of_words_per_language = {lang: 0 for lang in languages}
    # languages = args.languages[0].split(",")
    
    for lang in languages:
        print(f"\nProcessing language: {lang}")
        # Filter samples for this language
        lang_data = data[lang]
        print(f"Found {len(lang_data)} samples for language {lang}")
        
        if args.max_samples:
            lang_data = lang_data[:args.max_samples]
            print(f"Limiting to {len(lang_data)} samples")
        
        for sample_idx, sample in enumerate(tqdm(lang_data, desc=f"Processing {lang}")):
            
            # Process each dimension for this sample
            for dimension_name in sample.get("dimensions", {}):
                # print(f"\nProcessing sample {sample_idx + 1}/{len(lang_data)} - {sample['word']} - {dimension_name}")
                
                # Construct prompt and get dimension info
                constructed_prompt, dim1, dim2, answer, word, dim_name = visualizer.prmpt_dims_answrs(
                    visualizer.prompts, sample, dimension_name
                )
                
                # Run inference with hooks
                visualizer.inference_with_hooks(
                    word, lang, constructed_prompt, dim1, dim2, answer, sample, dimension_name
                )
                
                total_num_of_dimensions += 1
            total_num_of_words += 1
            total_num_of_words_per_language[lang] += 1

    
    print(f"\nProcessing completed!")
    print(f"Total samples processed: {total_num_of_dimensions}")
    print(f"Total number of words: {total_num_of_words}")
    print(f"Total number of words per language: {total_num_of_words_per_language}")
    print(f"Results saved to: {args.output_dir}")
    
    # Clean up
    del visualizer.model
    del visualizer.processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
