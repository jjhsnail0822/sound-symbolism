# Model : Qwen2.5-Omni-7B
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 2 --data-type original
import json
import re
import os
import argparse
import pickle as pkl
from typing import Union
import warnings
import numpy as np
import gc
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
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
SYSTEM_TEMPLATE = {
    "role": "system",
    "content": [
        {"type": "text", "text": SYSTEM_PROMPT}
    ],
}

with open(prompt_path, "r") as f:
    prompts = json.load(f)

class QwenOmniSemanticDimensionVisualizer:
    def __init__(
            self,
            model_path:str,
            data_path:str=data_path,
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
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="eager",
        )
        self.model.disable_talker()
        self.load_base_prompt()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        self.data = self.load_data()
    
    def load_base_prompt(self):
        self.prompts = prompts[self.exp_type][f"semantic_dimension_binary_{self.data_type}"]["user_prompt"]
        return self.prompts
    
    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data
    
    def prmpt_dims_answrs(self, prompt:str, data, dimension_name:str=None):
        if self.data_type != "audio":
            data_type_key = "word" if self.data_type == "original" \
                else "romanization" if self.data_type == "romanized" \
                else "ipa"
            if data_type_key not in data:
                raise KeyError(f"Data type '{data_type_key}' not found in data: {data.keys()}")
        dimension_info = data["dimensions"][dimension_name]
        dimension1 = dimension_name.split("-")[0]
        dimension2 = dimension_name.split("-")[1]
        answer = dimension_info["answer"]
        
        if self.data_type == "audio":
            word = f"data/processed/nat/tts/{data['language']}/{data['word']}.wav"
            # audio 타입일 때는 conversation 형태로 반환
            if "{audio}" in prompt:
                constructed_prompt = [
                    {"type": "text", "text": prompt.split("{audio}")[0]},
                    {"type": "audio", "audio": word},
                    {"type": "text", "text": prompt.split("{audio}")[1].format(
                        dimension1=dimension1,
                        dimension2=dimension2,
                    )},
                ]
            else:
                # fallback: 일반 문자열로 처리
                constructed_prompt = prompt.format(
                    word=data["word"],
                    dimension1=dimension1,
                    dimension2=dimension2,
                )
        else:
            word = data[data_type_key]
            constructed_prompt = prompt.format(
                word=word,
                dimension1=dimension1,
                dimension2=dimension2,
            )
        return constructed_prompt, dimension1, dimension2, answer, word, dimension_name
    
    def create_conversation(self, prompt, data):
        """통합된 conversation 생성 함수"""
        if self.data_type == "audio":
            audio_path = f'data/processed/nat/tts/{data["language"]}/{data["word"]}.wav'
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # prompt가 리스트 형태인 경우 (prmpt_dims_answrs에서 생성된 경우)
            if isinstance(prompt, list):
                conversation = [
                    SYSTEM_TEMPLATE,
                    {
                        "role": "user",
                        "content": prompt
                    },
                ]
            else:
                # 기존 로직: 문자열 prompt를 파싱
                if "<AUDIO>" in prompt:
                    question_parts = prompt.split("<AUDIO>")
                    if len(question_parts) == 2:
                        question_first_part = question_parts[0]
                        question_second_part = question_parts[1]
                    else:
                        word_placeholder = "{word}"
                        if word_placeholder in prompt:
                            parts = prompt.split(word_placeholder)
                            question_first_part = parts[0] + data["word"]
                            question_second_part = parts[1] if len(parts) > 1 else ""
                        else:
                            question_first_part = prompt
                            question_second_part = ""
                else:
                    word_placeholder = "{word}"
                    if word_placeholder in prompt:
                        parts = prompt.split(word_placeholder)
                        question_first_part = parts[0] + data["word"]
                        question_second_part = parts[1] if len(parts) > 1 else ""
                    else:
                        question_first_part = prompt
                        question_second_part = ""
                
                conversation = [
                    SYSTEM_TEMPLATE,
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
            # non-audio 타입: 단순 텍스트
            conversation = [
                SYSTEM_TEMPLATE,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        
        return conversation
    
    def get_attention_matrix(self, prompt, data:dict):
        # 통합된 conversation 생성 함수 사용
        conversation = self.create_conversation(prompt, data)
        
        USE_AUDIO_IN_VIDEO = True
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        # Audio가 제대로 로드되었는지 확인
        if self.data_type == "audio" and (audios is None or len(audios) == 0):
            print(f"Warning: No audio loaded for {data['word']}")
            # Fallback: audio 없이 텍스트만으로 처리
            conversation_text_only = [
                SYSTEM_TEMPLATE,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Given a spoken word '{data['word']}', which semantic feature best describes the word based on auditory impression?"}
                    ],
                },
            ]
            text = self.processor.apply_chat_template(conversation_text_only, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation_text_only, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
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
        joined_target = ''.join(cleaned_target)
        target_len = len(cleaned_target)
        for i in range(len(cleaned_tokens) - target_len + 1):
            window = cleaned_tokens[i:i+target_len]
            joined_window = ''.join(window)
            if joined_window == joined_target:
                matches.append(list(range(i, i+target_len)))
        if not matches and target_len > 1:
            for i in range(len(cleaned_tokens)):
                for j in range(i+1, len(cleaned_tokens)+1):
                    window = cleaned_tokens[i:j]
                    joined_window = ''.join(window)
                    if joined_window == joined_target:
                        matches.append(list(range(i, j)))
                        break
        return matches

    def find_tag_spans(self, tokens, tag_string, max_window=5):
        cleaned_tokens = [self._clean_token(t) for t in tokens]
        tag_string = tag_string.replace(" ", "")
        matches = []
        for window in range(1, max_window+1):
            for i in range(len(cleaned_tokens) - window + 1):
                window_str = ''.join(cleaned_tokens[i:i+window]).replace(" ", "")
                if window_str == tag_string:
                    matches.append(list(range(i, i+window)))
        return matches

    def extract_relevant_token_indices(self, tokens, dimension1, dimension2, word=None):
        word_tag_matches = self.find_tag_spans(tokens, 'WORD')
        if len(word_tag_matches) < 2:
            raise ValueError(f"[WORD] tag appears less than twice in tokens: {tokens}")
        second_word_tag_span = word_tag_matches[1]
        semdim_tag_matches = self.find_tag_spans(tokens, 'SEMANTICDIMENSION')
        if not semdim_tag_matches:
            raise ValueError(f"[SEMANTIC DIMENSION] tag not found in tokens: {tokens}")
        semdim_span = semdim_tag_matches[0]
        options_tag_matches = self.find_tag_spans(tokens, 'OPTIONS')
        options_span = options_tag_matches[0] if options_tag_matches else None
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
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, "self_attention")
        os.makedirs(output_dir, exist_ok=True)
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dimension1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dimension2))
        save_path = os.path.join(output_dir, f"{safe_word}_{safe_dim1}_{safe_dim2}_{layer_type}.pkl")
        with open(save_path, "wb") as f:
            pkl.dump(matrix_data, f)
    
    def read_matrix(self, layer_type="self", attention_type="self_attention", word_tokens=None, dimension1=None, dimension2=None, lang="en"):
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dimension1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dimension2))
        
        matrix_path = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, attention_type, f"{safe_word}_{safe_dim1}_{safe_dim2}_{layer_type}.pkl")
        
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
        indices = []
        
        def clean_token(token):
            """Clean token by removing 'Ġ' and check if it's a special token"""
            if token.startswith("Ġ"):
                token = token[1:]
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
                        break # Only take the first occurrence
        
        indices = sorted(list(set(indices)))
        
        return indices
    
    def filter_relevant_indices(self, attention_matrix, row_tokens, column_tokens, word_tokens, option_tokens, dimension1, dimension2, answer, layer_type="self"):
        save_row_index = []
        save_column_index = []
        
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
                    
        elif layer_type == "output":
            answer_indices = self.find_token_indices(row_tokens, [answer])
            for idx, token in enumerate(row_tokens):
                if idx in answer_indices or idx in dim1_indices or idx in dim2_indices:
                    save_row_index.append(idx)
                    
            for idx, token in enumerate(column_tokens):
                if idx in option_tokens or idx in dim1_indices or idx in dim2_indices:
                    save_column_index.append(idx)
        if save_row_index and save_column_index:
            if isinstance(attention_matrix, tuple):
                attention_matrix = attention_matrix[0]
            if not hasattr(attention_matrix, 'index_select'):
                attention_matrix = torch.tensor(attention_matrix)
            
            tensor_shape = attention_matrix.shape
            
            if len(tensor_shape) == 4:  # [batch_size, layers, seq_len, seq_len]
                max_seq_len = tensor_shape[-1]  # Last dimension is sequence length
                valid_row_indices = [idx for idx in save_row_index if idx < max_seq_len]
                valid_col_indices = [idx for idx in save_column_index if idx < max_seq_len]
                
                if valid_row_indices and valid_col_indices:
                    # For 4D tensor [batch_size, layers, seq_len, seq_len], filter the last two dimensions
                    row_tensor = torch.tensor(valid_row_indices, device=attention_matrix.device)
                    col_tensor = torch.tensor(valid_col_indices, device=attention_matrix.device)
                    
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
                # For other dimensions, just return the original
                filtered_attention_matrix = attention_matrix
                valid_row_indices = save_row_index
                valid_col_indices = save_column_index
        else:
            filtered_attention_matrix = attention_matrix
            valid_row_indices = save_row_index
            valid_col_indices = save_column_index
        return filtered_attention_matrix, valid_row_indices, valid_col_indices
    
    def inference_with_hooks(self, word, lang, constructed_prompt, dim1, dim2, answer, data, dimension_name):
        # Get forward pass attention (existing functionality)
        attentions, tokens, inputs = self.get_attention_matrix(constructed_prompt, data)
        relevant_indices, word_span, dim1_indices, dim2_indices, word_indices = self.extract_relevant_token_indices(tokens, dim1, dim2, word=data['word'])
        if isinstance(attentions, tuple):
            attention_matrix = attentions[0]
        else:
            attention_matrix = attentions
        attn_filtered = attention_matrix[:, :, relevant_indices][:, :, :, relevant_indices]
        self.save_matrix(attn_filtered, dim1, dim2, answer, data['word'], [dim1, dim2], "self", lang, tokens, relevant_indices)
        print(f"Saved filtered attention matrix for {data['word']} - {dim1}-{dim2} to pickle file")
        
        print(f"Extracting generation attention for {data['word']} - {dim1}-{dim2}...")
        generation_attentions, generation_tokens, generation_inputs, final_input_ids, generated_text = self.get_generation_attention_matrix(
            constructed_prompt, data, max_new_tokens=self.max_tokens
        )
        
        # Save generation attention matrix (filtered)
        self.save_generation_attention_matrix(
            generation_attentions, dim1, dim2, answer, data['word'], [dim1, dim2], lang, generation_tokens, final_input_ids, generation_tokens
        )
        
        data_type_key = {"audio": "audio", "original": "word", "romanized": "romanization", "ipa": "ipa"}
        
        if self.data_type != "audio":
            input_word = data[data_type_key[self.data_type]]
        else:
            input_word = data["word"]  # audio 타입일 때는 실제 단어 이름 사용

        # Analyze generation attention patterns
        generation_analysis = self.extract_generation_attention_analysis(
            generation_attentions, generation_tokens, answer, lang, dim1, dim2, data['word'], input_word
        )
        
        # Save generation attention analysis
        self.save_generation_attention_analysis(
            generation_analysis, dim1, dim2, answer, data['word'], [dim1, dim2], lang, generation_tokens, generated_text
        )
        print(f"Saved generation attention analysis for {data['word']} - {dim1}-{dim2}")

    def get_generation_attention_matrix(self, prompt, data: dict, max_new_tokens: int = 32):
        """Extract attention matrix during text generation (autoregressive decoding)"""
        # 통합된 conversation 생성 함수 사용
        conversation = self.create_conversation(prompt, data)
        
        USE_AUDIO_IN_VIDEO = True
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        # Audio가 제대로 로드되었는지 확인
        if self.data_type == "audio" and (audios is None or len(audios) == 0):
            print(f"Warning: No audio loaded for {data['word']}")
            conversation_text_only = [
                SYSTEM_TEMPLATE,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Given a spoken word '{data['word']}', which semantic feature best describes the word based on auditory impression?"}
                    ],
                },
            ]
            text = self.processor.apply_chat_template(conversation_text_only, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation_text_only, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
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
        
        # Store attention matrices for each generation step
        all_attention_matrices = []
        all_tokens = []
        current_input_ids = inputs['input_ids'].clone()
        current_attention_mask = inputs['attention_mask'].clone()
        
        # Get initial tokens
        initial_tokens = self.processor.tokenizer.convert_ids_to_tokens(current_input_ids[0])
        all_tokens.append(initial_tokens.copy())
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.model.thinker(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    output_attentions=True,
                    return_dict=True,
                    use_cache=True
                )
                attentions = outputs.attentions
                all_attention_matrices.append(attentions)
                logits = outputs.logits[:, -1, :]  # Take last token's logits
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                next_token_id = next_token_id.to(current_input_ids.device)
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token_id, device=current_input_ids.device)], dim=-1)
                new_token = self.processor.tokenizer.convert_ids_to_tokens(next_token_id[0])
                all_tokens.append(new_token)
                if next_token_id.item() == self.processor.tokenizer.eos_token_id:
                    break
        
        # Convert all_tokens to a single list
        final_tokens = []
        for token_list in all_tokens:
            final_tokens.extend(token_list)
        
        # Decode generated text
        input_length = len(all_tokens[0])  # input 토큰 개수
        generated_ids = current_input_ids[0][input_length:]
        generated_text = self.processor.tokenizer.decode(generated_ids)
        
        return all_attention_matrices, final_tokens, inputs, current_input_ids, generated_text

    def _find_token_indices_by_string(self, tokens, target_string):
        cleaned_tokens = [self._clean_token(t) for t in tokens]
        cleaned_target = self._clean_token(target_string)
        matches = []
        for i in range(len(cleaned_tokens)):
            for j in range(i+1, len(cleaned_tokens)+1):
                window = cleaned_tokens[i:j]
                if ''.join(window) == cleaned_target:
                    matches.append(list(range(i, j)))
                    break
        flat = []
        for m in matches:
            flat.extend(m)
        return sorted(set(flat))

    def robust_subtoken_match(self, tokens, target_string):
        """Clean and tokenize both tokens and target_string, then robustly find subtoken sequence matches."""
        cleaned_tokens = [self._clean_token(t) for t in tokens]
        tokenized_target = self.processor.tokenizer.tokenize(self._clean_token(target_string))
        cleaned_target = [self._clean_token(t) for t in tokenized_target]
        joined_target = ''.join(cleaned_target)
        matches = []
        for i in range(len(cleaned_tokens)):
            for j in range(i+1, len(cleaned_tokens)+1):
                window = cleaned_tokens[i:j]
                if ''.join(window) == joined_target:
                    matches.append(list(range(i, j)))
                    break  # Only first match per start index
        return matches

    def extract_generation_attention_analysis(self, all_attention_matrices:list[tuple[torch.Tensor, ...]], tokens:list[str], answer:str, lang:str, dimension1:str, dimension2:str, word:str, input_word:str=None):
        """Analyze attention patterns during generation, focusing on output tokens"""
        # Find the indices of the answer tokens in the final token sequence
        answer_subtokens = self.processor.tokenizer.tokenize(answer)
        input_word = input_word.replace(" ", "")
        answer_indices = self.find_subtoken_sequence_indices(tokens, answer_subtokens)
        # Flatten answer_indices
        temp_answer_indices = []
        for index in answer_indices:
            for i in index:
                temp_answer_indices.append(i)
        answer_indices = temp_answer_indices

        # Use direct token comparison for better matching
        dim1_indices = self._find_token_indices_by_string(tokens, dimension1)
        dim2_indices = self._find_token_indices_by_string(tokens, dimension2)

        # --- word_indices 처리 ---
        word_indices = []
        if self.data_type == "audio":
            # Audio token 범위 찾기: <|audio_bos|>부터 <|audio_eos|>까지
            audio_bos_token = "<|audio_bos|>"
            audio_eos_token = "<|audio_eos|>"
            audio_token = "<|AUDIO|>"
            
            # <|audio_bos|>와 <|audio_eos|> 인덱스 찾기
            audio_bos_indices = [i for i, t in enumerate(tokens) if self._clean_token(t) == self._clean_token(audio_bos_token)]
            audio_eos_indices = [i for i, t in enumerate(tokens) if self._clean_token(t) == self._clean_token(audio_eos_token)]
            
            if audio_bos_indices and audio_eos_indices:
                # 첫 번째 audio_bos와 첫 번째 audio_eos 사이의 <|AUDIO|> 토큰들 찾기
                start_idx = audio_bos_indices[0] + 1  # audio_bos 다음부터
                end_idx = audio_eos_indices[0]  # audio_eos 전까지
                
                audio_indices = []
                for i in range(start_idx, end_idx):
                    if self._clean_token(tokens[i]) == self._clean_token(audio_token):
                        audio_indices.append(i)
                
                # 각 <|AUDIO|> 토큰을 f"{word}_{num}" 형태로 대체하고 인덱스 저장
                for idx, audio_idx in enumerate(audio_indices):
                    word_indices.append(audio_idx)
                    # 토큰 리스트에서 해당 위치의 토큰을 대체
                    if audio_idx < len(tokens):
                        tokens[audio_idx] = f"{word}_{idx}"
            else:
                print(f"Warning: Could not find audio_bos or audio_eos tokens in {word}")
                # Fallback: 기존 방식으로 <|AUDIO|> 토큰 찾기
                audio_indices = [i for i, t in enumerate(tokens) if self._clean_token(t) == self._clean_token(audio_token)]
                for idx, audio_idx in enumerate(audio_indices):
                    word_indices.append(audio_idx)
                    if audio_idx < len(tokens):
                        tokens[audio_idx] = f"{word}_{idx}"
        elif self.data_type == "original" and lang == "en":
            # input_word로 인덱스 찾기
            indices = self._find_token_indices_by_string(tokens, input_word)
            word_indices = indices
        else:
            # 1. input_word를 tokenize
            tokenized_input_word_list = self.processor.tokenizer.tokenize(input_word)
            tokenized_input_word_list = [self._clean_token(t) for t in tokenized_input_word_list]
            # 2. tokenized_input_word_list를 하나의 string으로 결합
            tokenized_word = "".join(tokenized_input_word_list)
            # 3. tokens를 _clean_token으로 전처리
            cleaned_tokens = [self._clean_token(t) for t in tokens]
            word_indices = []
            
            # 4. [WORD] 태그가 두 번째로 나타나는 인덱스 찾기
            word_tag_matches = self.find_tag_spans(cleaned_tokens, 'WORD')
            if len(word_tag_matches) < 2:
                print(f"Warning: [WORD] tag appears less than twice in tokens")
                word_indices = []
            else:
                second_word_tag_span = word_tag_matches[1]  # 두 번째 [WORD] 태그
                second_word_end_idx = second_word_tag_span[-1]  # 두 번째 [WORD] 태그의 끝 인덱스
                
                # 5. [SEMANTICDIMENSION] 태그가 나타나는 인덱스 찾기
                semdim_tag_matches = self.find_tag_spans(cleaned_tokens, 'SEMANTICDIMENSION')
                if not semdim_tag_matches:
                    print(f"Warning: [SEMANTIC DIMENSION] tag not found in tokens")
                    word_indices = []
                else:
                    semdim_span = semdim_tag_matches[0]
                    semdim_start_idx = semdim_span[0]
                    search_start = second_word_end_idx + 1
                    search_end = semdim_start_idx
                    search_tokens = cleaned_tokens[search_start:search_end]
                    combined_string = "".join(search_tokens)
                    if combined_string == tokenized_word:
                        for i in range(search_start, search_end):
                            if cleaned_tokens[i] != '':
                                word_indices.append(i)
                    else:
                        search_range_length = search_end - search_start
                        for remove_count in range(1, search_range_length):
                            left_removed_tokens = search_tokens[remove_count:]
                            left_combined = "".join(left_removed_tokens)
                            if left_combined == tokenized_word:
                                for i in range(search_start + remove_count, search_end):
                                    if cleaned_tokens[i] != '':
                                        word_indices.append(i)
                                break
                        if not word_indices:
                            for remove_count in range(1, search_range_length):
                                right_removed_tokens = search_tokens[:-remove_count]
                                right_combined = "".join(right_removed_tokens)
                                if right_combined == tokenized_word:
                                    for i in range(search_start, search_end - remove_count):
                                        if cleaned_tokens[i] != '':
                                            word_indices.append(i)
                                    break
                        if not word_indices:
                            for left_remove in range(1, search_range_length):
                                for right_remove in range(1, search_range_length - left_remove):
                                    middle_tokens = search_tokens[left_remove:-right_remove]
                                    middle_combined = "".join(middle_tokens)
                                    if middle_combined == tokenized_word:
                                        for i in range(search_start + left_remove, search_end - right_remove):
                                            if cleaned_tokens[i] != '':
                                                word_indices.append(i)
                                        break
                                if word_indices:
                                    break
                    if not word_indices:
                        print(f"Warning: Could not find word '{input_word}' (tokenized as {tokenized_input_word_list}) in tokens")
                        print(f"Search range: {search_start} to {search_end}")
                        print(f"Tokens in search range: {cleaned_tokens[search_start:search_end]}")
                        print(f"Combined string: {combined_string}")
                        print(f"Tokenized word: {tokenized_word}")
                        print(f"Tokenized word list: {tokenized_input_word_list}")
        # Analyze attention for each generation step
        generation_attention_analysis = []
        for step, step_attentions in enumerate(all_attention_matrices):
            step_analysis = {
                'step': step,
                'generated_token': tokens[-(len(all_attention_matrices) - step)] if step < len(all_attention_matrices) else None,
                'attention_to_answer': [],
                'attention_to_dimensions': [],
                'attention_to_word': [],
                'layer_attention_patterns': []
            }
            for layer_idx, layer_attention in enumerate(step_attentions):
                layer_attention = layer_attention[0]
                layer_analysis = {
                    'layer': layer_idx,
                    'head_attention_patterns': []
                }
                for head_idx in range(layer_attention.shape[0]):
                    head_attention = layer_attention[head_idx]
                    if step > 0:
                        last_token_attention = head_attention[-1, :]
                        attention_to_answer = sum(last_token_attention[idx] for idx in answer_indices if idx < len(last_token_attention))
                        attention_to_dim1 = sum(last_token_attention[idx] for idx in dim1_indices if idx < len(last_token_attention))
                        attention_to_dim2 = sum(last_token_attention[idx] for idx in dim2_indices if idx < len(last_token_attention))
                        attention_to_word = sum(last_token_attention[idx] for idx in word_indices if idx < len(last_token_attention))
                        if isinstance(attention_to_word, int):
                            breakpoint()
                            
                        head_analysis = {
                            'head': head_idx,
                            'word': word,
                            'input_word': input_word,
                            'attention_to_answer': attention_to_answer.item(),
                            'attention_to_dim1': attention_to_dim1.item(),
                            'attention_to_dim2': attention_to_dim2.item(),
                            'attention_to_word': attention_to_word.item(),
                            'full_attention_vector': last_token_attention.cpu().float().numpy()
                        }
                        layer_analysis['head_attention_patterns'].append(head_analysis)
                step_analysis['layer_attention_patterns'].append(layer_analysis)
            generation_attention_analysis.append(step_analysis)
        return generation_attention_analysis

    def save_generation_attention_analysis(self, generation_analysis, dimension1, dimension2, answer, word_tokens, option_tokens, lang="en", tokens=None, generated_text=None):
        """Save generation attention analysis as pickle file"""
        analysis_data = {
            "generation_analysis": generation_analysis,
            "dimension1": dimension1,
            "dimension2": dimension2,
            "answer": answer,
            "word_tokens": word_tokens,
            "option_tokens": option_tokens,
            "tokens": tokens,
            "generated_text": generated_text,
            "analysis_type": "generation_attention"
        }
        
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, "generation_attention")
        os.makedirs(output_dir, exist_ok=True)
        
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dimension1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dimension2))
        
        save_path = os.path.join(output_dir, f"{safe_word}_{safe_dim1}_{safe_dim2}_generation_analysis.pkl")
        
        with open(save_path, "wb") as f:
            pkl.dump(analysis_data, f)
        
        print(f"Generation attention analysis saved to: {save_path}")

    def save_generation_attention_matrix(self, all_attention_matrices, dimension1, dimension2, answer, word_tokens, option_tokens, lang="en", tokens=None, current_input_ids=None, all_tokens=None):
        if current_input_ids is not None and all_tokens is not None:
            input_length = len(all_tokens[0])
            input_ids = current_input_ids[0][:input_length]
            generated_ids = current_input_ids[0][input_length:]
            
            input_text = self.processor.tokenizer.decode(input_ids)
            generated_text = self.processor.tokenizer.decode(generated_ids)
            full_text = self.processor.tokenizer.decode(current_input_ids[0])
        else:
            input_text = "unknown"
            generated_text = "unknown"
            full_text = "unknown"
        
        filtered_attention_matrices = []
        relevant_indices_list = []
        
        for step, step_attentions in enumerate(all_attention_matrices):
            layer_attention = step_attentions[0]
            current_seq_len = layer_attention.shape[-1]
            relevant_indices, word_span, dim1_indices, dim2_indices, word_indices = self.extract_relevant_token_indices(
                tokens[:current_seq_len], dimension1, dimension2, word=word_tokens
            )
            
            filtered_step_attentions = []
            for layer_idx, layer_attn in enumerate(step_attentions):
                if len(relevant_indices) > 0:
                    filtered_attn = layer_attn[:, :, relevant_indices][:, :, :, relevant_indices]
                else:
                    filtered_attn = layer_attn
                
                filtered_step_attentions.append(filtered_attn)
            
            filtered_attention_matrices.append(tuple(filtered_step_attentions))
            relevant_indices_list.append(relevant_indices)
        
        matrix_data = {
            "attention_matrices": filtered_attention_matrices,
            "relevant_indices": relevant_indices_list,
            "dimension1": dimension1,
            "dimension2": dimension2,
            "answer": answer,
            "word_tokens": word_tokens,
            "option_tokens": option_tokens,
            "tokens": tokens,
            "input_text": input_text,
            "generated_text": generated_text,
            "full_text": full_text,
            "analysis_type": "generation_attention_matrix"
        }
        
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang)
        os.makedirs(output_dir, exist_ok=True)
        
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dimension1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dimension2))
        
        save_path = os.path.join(output_dir, f"gen_{safe_word}_{safe_dim1}_{safe_dim2}.pkl")
        
        with open(save_path, "wb") as f:
            pkl.dump(matrix_data, f)
        
        print(f"Generation attention matrix saved to: {save_path}")
        return save_path

    def analyze_generation_attention_summary(self, generation_analysis):
        if not generation_analysis:
            return None
        
        summary = {
            'total_steps': len(generation_analysis),
            'generated_tokens': [],
            'average_attention_to_answer': 0.0,
            'average_attention_to_dimensions': 0.0,
            'average_attention_to_word': 0.0,
            'layer_attention_summary': {},
            'head_attention_summary': {}
        }
        
        total_attention_to_answer = 0.0
        total_attention_to_dimensions = 0.0
        total_attention_to_word = 0.0
        step_count = 0
        
        for step_data in generation_analysis:
            if step_data['generated_token']:
                summary['generated_tokens'].append(step_data['generated_token'])
        
        for step_data in generation_analysis:
            if step_data['generated_token']:
                step_count += 1
                
                for layer_data in step_data['layer_attention_patterns']:
                    layer_idx = layer_data['layer']
                    
                    if layer_idx not in summary['layer_attention_summary']:
                        summary['layer_attention_summary'][layer_idx] = {
                            'total_attention_to_answer': 0.0,
                            'total_attention_to_dimensions': 0.0,
                            'total_attention_to_word': 0.0,
                            'step_count': 0
                        }
                    
                    for head_data in layer_data['head_attention_patterns']:
                        head_idx = head_data['head']
                        head_key = f"layer_{layer_idx}_head_{head_idx}"
                        
                        if head_key not in summary['head_attention_summary']:
                            summary['head_attention_summary'][head_key] = {
                                'total_attention_to_answer': 0.0,
                                'total_attention_to_dimensions': 0.0,
                                'total_attention_to_word': 0.0,
                                'step_count': 0
                            }
                        
                        summary['layer_attention_summary'][layer_idx]['total_attention_to_answer'] += head_data['attention_to_answer']
                        summary['layer_attention_summary'][layer_idx]['total_attention_to_dimensions'] += head_data['attention_to_dim1'] + head_data['attention_to_dim2']
                        summary['layer_attention_summary'][layer_idx]['total_attention_to_word'] += head_data['attention_to_word']
                        summary['layer_attention_summary'][layer_idx]['step_count'] += 1
                        
                        summary['head_attention_summary'][head_key]['total_attention_to_answer'] += head_data['attention_to_answer']
                        summary['head_attention_summary'][head_key]['total_attention_to_dimensions'] += head_data['attention_to_dim1'] + head_data['attention_to_dim2']
                        summary['head_attention_summary'][head_key]['total_attention_to_word'] += head_data['attention_to_word']
                        summary['head_attention_summary'][head_key]['step_count'] += 1
                        
                        total_attention_to_answer += head_data['attention_to_answer']
                        total_attention_to_dimensions += head_data['attention_to_dim1'] + head_data['attention_to_dim2']
                        total_attention_to_word += head_data['attention_to_word']
        
        # Calculate averages
        if step_count > 0:
            summary['average_attention_to_answer'] = total_attention_to_answer / step_count
            summary['average_attention_to_dimensions'] = total_attention_to_dimensions / step_count
            summary['average_attention_to_word'] = total_attention_to_word / step_count
        
        # Calculate layer averages
        for layer_idx in summary['layer_attention_summary']:
            layer_data = summary['layer_attention_summary'][layer_idx]
            if layer_data['step_count'] > 0:
                layer_data['average_attention_to_answer'] = layer_data['total_attention_to_answer'] / layer_data['step_count']
                layer_data['average_attention_to_dimensions'] = layer_data['total_attention_to_dimensions'] / layer_data['step_count']
                layer_data['average_attention_to_word'] = layer_data['total_attention_to_word'] / layer_data['step_count']
        
        # Calculate head averages
        for head_key in summary['head_attention_summary']:
            head_data = summary['head_attention_summary'][head_key]
            if head_data['step_count'] > 0:
                head_data['average_attention_to_answer'] = head_data['total_attention_to_answer'] / head_data['step_count']
                head_data['average_attention_to_dimensions'] = head_data['total_attention_to_dimensions'] / head_data['step_count']
                head_data['average_attention_to_word'] = head_data['total_attention_to_word'] / head_data['step_count']
        
        return summary

    def print_generation_attention_summary(self, summary):
        if not summary:
            print("No generation attention data available")
            return
        
        print(f"\n=== Generation Attention Summary ===")
        print(f"Total generation steps: {summary['total_steps']}")
        print(f"Generated tokens: {summary['generated_tokens']}")
        print(f"\nAverage attention scores:")
        print(f"  - To answer: {summary['average_attention_to_answer']:.4f}")
        print(f"  - To dimensions: {summary['average_attention_to_dimensions']:.4f}")
        print(f"  - To word: {summary['average_attention_to_word']:.4f}")
        
        print(f"\n=== Layer-wise Attention Summary ===")
        for layer_idx in sorted(summary['layer_attention_summary'].keys()):
            layer_data = summary['layer_attention_summary'][layer_idx]
            print(f"Layer {layer_idx}:")
            print(f"  - Avg attention to answer: {layer_data.get('average_attention_to_answer', 0):.4f}")
            print(f"  - Avg attention to dimensions: {layer_data.get('average_attention_to_dimensions', 0):.4f}")
            print(f"  - Avg attention to word: {layer_data.get('average_attention_to_word', 0):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Semantic Dimension Attention Heatmap Visualization")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Omni-7B", help="Model path (default: Qwen/Qwen2.5-Omni-7B)")
    parser.add_argument('--data-path', type=str, default="data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json",
                       help="Path to semantic dimension data JSON file")
    parser.add_argument('--output-dir', type=str, default="results/experiments/understanding/attention_heatmap",
                       help="Output directory for heatmaps and matrices")
    parser.add_argument('--data-type', type=str, default="audio", choices=["audio", "original", "romanized", "ipa"],
                       help="Data type to process")
    parser.add_argument('--max-tokens', type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.0, help="Sampling temperature")
    parser.add_argument('--max-samples', type=int, default=None, help="Maximum number of samples to process (default: all)")
    parser.add_argument('--languages', nargs='+', default=["en", "fr", "ko", "ja"], help="Languages to process")
    
    args = parser.parse_args()
    max_samples:int = args.max_samples
    print(f"Data type: {args.data_type}")

    visualizer = QwenOmniSemanticDimensionVisualizer(
        model_path=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        exp_type="semantic_dimension",
        data_type=args.data_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    languages = ["en", "fr", "ja", "ko"]
    # languages = ["fr", "ja", "ko"]
    total_num_of_dimensions = 0
    total_num_of_words = 0
    total_num_of_words_per_language = {lang: 0 for lang in languages}
    
    for lang in languages:
        print(f"\nProcessing language: {lang}")
        lang_data = visualizer.data[lang]
        print(f"Found {len(lang_data)} samples for language {lang}")
        
        if max_samples:
            lang_data = lang_data[:max_samples]
            print(f"Limiting to {len(lang_data)} samples")
        
        for sample_idx, sample in enumerate(tqdm(lang_data, desc=f"Processing {lang}")):
            
            # Process each dimension for this sample
            for dimension_name in sample.get("dimensions", {}):
                constructed_prompt, dim1, dim2, answer, word, dim_name = visualizer.prmpt_dims_answrs(
                    visualizer.prompts, sample, dimension_name
                )
                
                # Run inference with hooks
                visualizer.inference_with_hooks(
                    word, lang, constructed_prompt, dim1, dim2, answer, sample, dimension_name
                )
                
                # Print generation attention summary for the first few samples
                if total_num_of_dimensions < 3:  # Only for first 3 samples to avoid spam
                    # Load the saved generation analysis
                    output_dir = os.path.join(visualizer.output_dir, visualizer.exp_type, visualizer.data_type, lang, "generation_attention")
                    safe_word = re.sub(r'[^\w\-_.]', '_', str(sample['word']))
                    safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dim1))
                    safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dim2))
                    analysis_path = os.path.join(output_dir, f"{safe_word}_{safe_dim1}_{safe_dim2}_generation_analysis.pkl")
                    
                    if os.path.exists(analysis_path):
                        with open(analysis_path, "rb") as f:
                            analysis_data = pkl.load(f)
                        
                        generation_analysis = analysis_data["generation_analysis"]
                        summary = visualizer.analyze_generation_attention_summary(generation_analysis)
                        print(f"\n=== Generation Attention Summary for {sample['word']} - {dim1}-{dim2} ===")
                        visualizer.print_generation_attention_summary(summary)
                
                total_num_of_dimensions += 1
            total_num_of_words += 1
            total_num_of_words_per_language[lang] += 1
            gc.collect()
            torch.cuda.empty_cache()
    
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
