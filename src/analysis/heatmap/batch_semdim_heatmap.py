# Model : Qwen2.5-Omni-7B
# python src/analysis/heatmap/batch_semdim_heatmap.py --max-samples 30 --data-type ipa
# python src/analysis/heatmap/batch_semdim_heatmap.py --max-samples 3000 --data-type audio --constructed --flip
import json
import re
import os
import argparse
import pickle as pkl
from typing import Union
import numpy as np
import gc
import gc
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch
from tqdm import tqdm
from qwen_omni_utils import process_mm_info

data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"
prompt_path = "data/prompts/prompts.json"
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
            output_dir:str="results/experiments/understanding/attention_heatmap/nat",
            exp_type:str="semantic_dimension",
            data_type:str="audio",
            max_tokens:int=32,
            temperature:float=0.0,
            flip:bool=False,
            constructed:bool=False,
        ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.data_type = data_type
        self.exp_type = exp_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.flip = flip
        self.constructed = constructed
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Try to use flash_attention_2, but fallback to eager if not supported
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
        self.model_name = "qwen-7B" if "7B" in self.model_path else "qwen-4B"
    
    def load_base_prompt(self):
        self.prompts = prompts[self.exp_type][f"semantic_dimension_binary_{self.data_type}"]["user_prompt"]
        return self.prompts
    
    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data
    
    def prmpt_dims_answrs(self, prompt:str, data:dict, dimension_name:str=None):
        if self.data_type != "audio":
            data_type_key = "word" if self.data_type == "original" else "ipa"
        dimension_info = data["dimensions"][dimension_name]
        self.dim1, self.dim2 = dimension_name.split("-")[0], dimension_name.split("-")[1]
        answer = dimension_info["answer"]
        # Flip if needed
        if self.flip:
            self.dim1, self.dim2 = self.dim2, self.dim1
        if self.data_type == "audio":
            if self.constructed:
                word = f"data/processed/art/tts/{data['word']}.wav"
            else:
                word = f"data/processed/nat/tts/{data['language']}/{data['word']}.wav"
            # For audio type, we need to format the prompt with dimension info first
            if "{audio}" in prompt:
                constructed_prompt = [
                    {"type": "text", "text": prompt.split("{audio}")[0]},
                    {"type": "audio", "audio": word},
                    {"type": "text", "text": prompt.split("{audio}")[1].format(dimension1=dim1, dimension2=dim2)},
                ]
            else:
                raise ValueError("Audio prompt could not find audio token")
        else:
            word = data[data_type_key]
            constructed_prompt = prompt.format(
                word=word,
                dimension1=dim1,
                dimension2=dim2,
            )
        return constructed_prompt, dim1, dim2, answer, word, dimension_name
    
    def create_conversation(self, prompt:str, data:dict):
        if self.data_type == "audio":
            audio_path = f'data/processed/nat/tts/{data["language"]}/{data["word"]}.wav'
            if isinstance(prompt, list):
                conversation = [
                    SYSTEM_TEMPLATE,
                    {
                        "role": "user",
                        "content": prompt
                    },
                ]
            else:
                word_placeholder = "{word}"
                if "<AUDIO>" in prompt:
                    question_parts = prompt.split("<AUDIO>")
                    if len(question_parts) == 2:
                        question_first_part = question_parts[0]
                        question_second_part = question_parts[1]
                    else:
                        if word_placeholder in prompt:
                            parts = prompt.split(word_placeholder)
                            question_first_part = parts[0] + data["word"]
                            question_second_part = parts[1] if len(parts) > 1 else ""
                        else:
                            question_first_part = prompt
                            question_second_part = ""
                else:
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
            conversation = [
                SYSTEM_TEMPLATE,
                {"role": "user","content": [{"type": "text", "text": prompt}]},
            ]
        return conversation
    
    def trim_silence_from_audio(self, audio_data:np.ndarray, threshold:float=0.01):
        if audio_data is None or len(audio_data) == 0:
            return audio_data
        non_zero_indices = np.nonzero(np.abs(audio_data) > threshold)[0]
        if len(non_zero_indices) == 0:
            print(f"[WARNING] Audio is completely silent!")
            return audio_data
        start_idx = non_zero_indices[0]
        trimmed_audio = audio_data[start_idx:]
        return trimmed_audio

    def get_attention_matrix(self, prompt:str, data:dict):
        conversation = self.create_conversation(prompt, data)
        
        USE_AUDIO_IN_VIDEO = True
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        if self.data_type == "audio":
            if self.constructed:
                audio_path = f"data/processed/art/tts/{data['word']}.wav"
            else:
                audio_path = f'data/processed/nat/tts/{data["language"]}/{data["word"]}.wav'
            
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        if self.data_type == "audio" and audios is not None:
            for i, audio in enumerate(audios):
                trimmed_audio = self.trim_silence_from_audio(audio)
                audios[i] = trimmed_audio
        
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with torch.no_grad():
            thinker_model = self.model.thinker.model
            outputs = thinker_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_attentions=True, return_dict=True)
        
        attentions = outputs.attentions
        tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return attentions, tokens, inputs
    
    def _clean_token(self, token:str):
        return re.sub(r'^[ĠĊ\[\],.:;!?\n\r\t]+|[ĠĊ\[\],.:;!?\n\r\t]+$', '', token)

    def find_subtoken_sequence_indices(self, tokens:list[str], target_subtokens:list[str]):
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

    def find_tag_spans(self, tokens:list[str], tag_string:str, max_window:int=5):
        cleaned_tokens = [self._clean_token(t) for t in tokens]
        tag_string = tag_string.replace(" ", "")
        matches = []
        for window in range(1, max_window+1):
            for i in range(len(cleaned_tokens) - window + 1):
                window_str = ''.join(cleaned_tokens[i:i+window]).replace(" ", "")
                if window_str == tag_string:
                    matches.append(list(range(i, i+window)))
        return matches

    def extract_relevant_token_indices(self, tokens:list[str], dimension1:str, dimension2:str, input_word:str, word:str=None):
        word_tag_matches = self.find_tag_spans(tokens, 'WORD')
        semdim_tag_matches = self.find_tag_spans(tokens, 'SEMANTICDIMENSION')
        options_tag_matches = self.find_tag_spans(tokens, 'OPTIONS')
        answer_tag_matches = self.find_tag_spans(tokens, "Answerwith")

        second_word_tag_span = word_tag_matches[1]
        semdim_span = semdim_tag_matches[0]
        options_span = options_tag_matches[0] if options_tag_matches else None
        answer_span = answer_tag_matches[0]
        
        word_indices = []
        if word is not None:
            word_subtokens = self.processor.tokenizer.tokenize(input_word)
            word_matches = self.find_subtoken_sequence_indices(tokens, word_subtokens)
            for match in word_matches:
                word_indices.extend(match)
        word_span_start = second_word_tag_span[-1] + 1
        word_span_end = semdim_span[0]
        word_span = list(range(word_span_start+1, word_span_end-2))
        
        search_start = options_span[-1] + 1
        search_end = answer_span[0] if answer_span else len(tokens)
        
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
        dim1_indices, dim2_indices = sorted(set(dim1_indices)), sorted(set(dim2_indices))
        relevant_indices = sorted(set(word_span + dim1_indices + dim2_indices + word_indices))
        breakpoint()
        return relevant_indices
    
    def save_matrix(self, attention_matrix, dimension1:str, dimension2:str, answer:str, word_tokens:list[str], option_tokens:list[str], layer_type:str="self", lang:str="en", tokens:list[str]=None, relevant_indices:list[int]=None):
        dim1, dim2 = dimension1, dimension2
        matrix_data = {"attention_matrix": attention_matrix, "dimension1": dim1, "dimension2": dim2, "answer": answer, "word_tokens": word_tokens, "option_tokens": option_tokens, "tokens": tokens, "relevant_indices": relevant_indices}
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, "self_attention", f"{dim1}_{dim2}")
        os.makedirs(output_dir, exist_ok=True)
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        save_path = os.path.join(output_dir, f"{safe_word}_{dim1}_{dim2}_{layer_type}.pkl")

        with open(save_path, "wb") as f:
            pkl.dump(matrix_data, f)

    def find_token_indices(self, tokens:list[str], target_tokens:list[str]):
        indices = []
        
        def is_special_token(token):
            return '<' in token and '>' in token
        
        for target in target_tokens:
            if isinstance(target, str):
                clean_target = self._clean_token(target)
                
                for idx, token in enumerate(tokens):
                    clean_token_str = self._clean_token(token)
                    if clean_target == clean_token_str:
                        indices.append(idx)
            
            elif isinstance(target, list):
                clean_targets = [self._clean_token(t) for t in target]
                
                for i in range(len(tokens) - len(clean_targets) + 1):
                    match_found = True
                    sequence_indices = []
                    
                    for j, clean_target in enumerate(clean_targets):
                        current_token = tokens[i + j]
                        clean_current = self._clean_token(current_token)
                        
                        if clean_current != clean_target or (j > 0 and is_special_token(current_token)):
                            match_found = False
                            break
                        sequence_indices.append(i + j)
                    
                    if match_found:
                        indices.extend(sequence_indices)
                        break
        
        indices = sorted(list(set(indices)))
        return indices

    def inference_with_hooks_batch(self, words:list[str], langs:list[str], constructed_prompts:list[str], dim1s:list[str], dim2s:list[str], answers:list[str], datas:list[dict], dimension_names:list[str], batch_size:int=64):
        num_samples = len(words)
        results = []
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_prompts = constructed_prompts[batch_start:batch_end]
            batch_datas = datas[batch_start:batch_end]
            batch_words = words[batch_start:batch_end]
            batch_langs = langs[batch_start:batch_end]
            batch_dim1s = dim1s[batch_start:batch_end]
            batch_dim2s = dim2s[batch_start:batch_end]
            batch_answers = answers[batch_start:batch_end]
            batch_dimension_names = dimension_names[batch_start:batch_end]

            # Prepare batch inputs
            conversations = [self.create_conversation(prompt, data) for prompt, data in zip(batch_prompts, batch_datas)]
            USE_AUDIO_IN_VIDEO = True
            def ensure_str(x):
                if isinstance(x, list):
                    return ''.join(map(str, x))
                return str(x)
            texts = [
                ensure_str(self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False))
                for conv in conversations
            ]
            audios = []
            images = []
            videos = []
            for conv, data in zip(conversations, batch_datas):
                a, i, v = process_mm_info(conv, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                audios.append(a[0] if a and len(a) > 0 else None)
                images.append(i[0] if i and len(i) > 0 else None)
                videos.append(v[0] if v and len(v) > 0 else None)
            if self.data_type == "audio":
                for idx, audio in enumerate(audios):
                    if audio is not None:
                        audios[idx] = self.trim_silence_from_audio(audio)

            inputs = self.processor(
                text=texts,
                audio=audios if self.data_type == "audio" else None,
                images=images if any(images) else None,
                videos=videos if any(videos) else None,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            with torch.no_grad():
                thinker_model = self.model.thinker.model
                outputs = thinker_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_attentions=True, return_dict=True)
            attentions = outputs.attentions
            # tokens: list of list
            tokens_batch = [self.processor.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs['input_ids']]
            # For each sample in batch, do analysis and save
            for i in range(batch_end - batch_start):
                tokens = tokens_batch[i]
                data = batch_datas[i]
                word = batch_words[i]
                lang = batch_langs[i]
                dim1 = batch_dim1s[i]
                dim2 = batch_dim2s[i]
                answer = batch_answers[i]
                dimension_name = batch_dimension_names[i]
                if self.data_type == "audio":
                    input_word = "<|AUDIO|>"
                elif self.data_type == "original":
                    input_word = data['word']
                elif self.data_type == "ipa":
                    input_word = data["ipa"]
                elif self.data_type == "romanized":
                    input_word = data["romanization"]
                relevant_indices = self.extract_relevant_token_indices(tokens, dim1, dim2, input_word, word=data['word'])
                # Extract per-sample attention: [layer][i] shape [num_heads, seq, seq]
                sample_attentions = [layer_attn[i] for layer_attn in attentions]  # list of [num_heads, seq, seq]
                attn_filtered = torch.stack([
                    attn[:, relevant_indices][:, :, relevant_indices] for attn in sample_attentions
                ])  # [layer, num_heads, rel, rel]
                self.save_matrix(attn_filtered, dim1, dim2, answer, data['word'], [dim1, dim2], "self", lang, tokens, relevant_indices)
                generation_attentions, generation_tokens, _, final_input_ids, response, input_length = self.get_generation_attention_matrix(
                    batch_prompts[i], data, max_new_tokens=self.max_tokens
                )
                data_type_key = {"audio": "audio", "original": "word", "romanized": "romanization", "ipa": "ipa"}
                if self.data_type != "audio":
                    input_word = data[data_type_key[self.data_type]]
                else:
                    input_word = data["word"]
                generation_analysis = self.extract_generation_attention_analysis(
                    generation_attentions, generation_tokens, response, answer, lang, dim1, dim2, data['word'], input_word
                )
                self.save_generation_attention_matrix(
                    generation_attentions, dim1, dim2, answer, data['word'], input_word, [dim1, dim2], lang, generation_tokens, final_input_ids, generation_tokens
                )
                self.save_generation_attention_analysis(
                    generation_analysis, dim1, dim2, answer, data['word'], [dim1, dim2], lang, generation_tokens, response
                )
                results.append((data['word'], dim1, dim2))
        return results

    def get_generation_attention_matrix(self, prompt:str, data:dict, max_new_tokens:int=32):
        conversation = self.create_conversation(prompt, data)
        
        USE_AUDIO_IN_VIDEO = True
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        if self.data_type == "audio" and (audios is None or len(audios) == 0):
            print(f"[WARNING] No audio loaded for {data['word']}")
            raise ValueError(f"Audio file not found or could not be loaded for {data['word']}. Expected path: data/processed/nat/tts/{data['language']}/{data['word']}.wav")
        
        if self.data_type == "audio" and audios is not None:
            for i, audio in enumerate(audios):
                trimmed_audio = self.trim_silence_from_audio(audio)
                audios[i] = trimmed_audio

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
        
        all_attention_matrices = []
        all_tokens = []
        current_input_ids = inputs['input_ids'].clone()
        current_attention_mask = inputs['attention_mask'].clone()
        initial_tokens = self.processor.tokenizer.convert_ids_to_tokens(current_input_ids[0])
        one_token_id = self.processor.tokenizer.convert_tokens_to_ids("1")
        two_token_id = self.processor.tokenizer.convert_tokens_to_ids("2")
        all_tokens.append(initial_tokens.copy())
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                if self.data_type == "audio" and 'input_features' in inputs:
                    outputs = self.model.thinker(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        input_features=inputs['input_features'],
                        feature_attention_mask=inputs['feature_attention_mask'],
                        output_attentions=True,
                        return_dict=True,
                        use_cache=True
                    )
                else:
                    outputs = self.model.thinker(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        output_attentions=True,
                        return_dict=True,
                        use_cache=True
                    )
                attentions = outputs.attentions
                all_attention_matrices.append(attentions)
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                next_token_id = next_token_id.to(current_input_ids.device)
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token_id, device=current_input_ids.device)], dim=-1)
                new_token = self.processor.tokenizer.convert_ids_to_tokens(next_token_id[0])
                all_tokens.append(new_token)
                
                if next_token_id.item() in [one_token_id, two_token_id]:
                    break
        
        # Convert all_tokens to a single list
        final_tokens = []
        for token_list in all_tokens:
            final_tokens.extend(token_list)
        
        # Decode generated text
        input_length = len(all_tokens[0])
        response_ids = current_input_ids[0][input_length:]
        response = self.processor.tokenizer.decode(response_ids)
        
        return all_attention_matrices, final_tokens, inputs, current_input_ids, response, input_length

    def _analyze_generation_step_completeness(self, step_tokens:list[str], step_idx:int, dim1:str, dim2:str):
        if not step_tokens:
            return False, "Empty tokens"
        
        last_tokens = step_tokens[-1:] if len(step_tokens) >= 1 else step_tokens
        eos_found = any(token in ['1', '2'] for token in last_tokens)
        
        if eos_found:
            return True, "Complete with EOS"
        
        last_token = step_tokens[-1] if step_tokens else ""
        if last_token in [dim1, dim2] or last_token.lower() in [dim1.lower(), dim2.lower()]:
            return True, "Complete answer"
        
        return True, "Appears complete"

    def _find_target_tokens_in_step(self, step_tokens:list[str], word:str, dim1:str, dim2:str):
        target_indices = {"word": [], "dim1": [], "dim2": []}
        
        def remove_indices(indices: list[list[int]], threshold: int):
            filtered_indices = []
            for sublist in indices:
                if all(i > threshold for i in sublist):
                    filtered_indices.append(sublist)
            return filtered_indices
        
        if self.data_type == "audio":
            audio_token = "<|AUDIO|>"
            for i, token in enumerate(step_tokens):
                if self._clean_token(token) == self._clean_token(audio_token):
                    target_indices['word'].append(i)
        else:
            word_subtokens = self.processor.tokenizer.tokenize(word)
            word_matches = self.find_subtoken_sequence_indices(step_tokens, word_subtokens)
            word_matches = remove_indices(word_matches, 60)
            for match in word_matches:
                target_indices['word'].extend(match)
                
        dim1_subtokens = self.processor.tokenizer.tokenize(dim1)
        dim1_matches = self.find_subtoken_sequence_indices(step_tokens, dim1_subtokens)
        dim1_matches = remove_indices(dim1_matches, 77)
        for match in dim1_matches:
            target_indices['dim1'].extend(match)
        
        dim2_subtokens = self.processor.tokenizer.tokenize(dim2)
        dim2_matches = self.find_subtoken_sequence_indices(step_tokens, dim2_subtokens)
        dim1_max_index = max(target_indices['dim1'])
        dim2_matches = remove_indices(dim2_matches, dim1_max_index)
        for match in dim2_matches:
            target_indices['dim2'].extend(match)
        
        for key in target_indices:
            target_indices[key] = sorted(set(target_indices[key]))
        
        return target_indices

    def _calculate_step_attention_scores(self, attention_matrix:torch.Tensor, tokens:list[str], query_indices:list[int], key_indices:list[int], step_idx:int, head_idx:int):
        total_score = 0.0
        valid_pairs = 0
        seq_len = attention_matrix.shape[0]  # Should be equal to shape[1] for self-attention

        for q_idx in query_indices:
            if q_idx < seq_len:
                for k_idx in key_indices:
                    if k_idx < seq_len:
                        if k_idx <= q_idx:
                            score = attention_matrix[q_idx, k_idx].item()
                            total_score += score
                            valid_pairs += 1

        return total_score, valid_pairs

    def _calculate_step_normalized_attention(self, attention_matrix, query_indices, key_indices):
        total_query_attention = 0.0
        target_attention = 0.0
        seq_len = attention_matrix.shape[0]  # Should be equal to shape[1] for self-attention
        
        for q_idx in query_indices:
            if q_idx < seq_len:
                row_sum = attention_matrix[q_idx, :q_idx+1].sum().item()
                total_query_attention += row_sum
                for k_idx in key_indices:
                    if k_idx < seq_len and k_idx <= q_idx:
                        target_score = attention_matrix[q_idx, k_idx].item()
                        target_attention += target_score
        
        if total_query_attention > 0:
            return target_attention / total_query_attention
        return 0.0

    def _aggregate_step_scores(self, scores_matrix, aggregation_type):
        if aggregation_type == "all":
            return scores_matrix.mean().item()
        elif aggregation_type == "layers":
            return scores_matrix.mean(dim=0)
        elif aggregation_type == "heads":
            return scores_matrix.mean(dim=1)
        elif aggregation_type == "individual":
            return scores_matrix
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")

    def extract_generation_attention_analysis(self, all_attention_matrices:list[tuple[torch.Tensor, ...]], tokens:list[str], response:str, answer:str, lang:str, dim1:str, dim2:str, word:str, input_word:str=None):
        response = "1." if answer == dim1 else "2."
        # Filter out incomplete generation steps
        valid_steps = []
        step_tokens_list = []
        
        for step_idx, step_attentions in enumerate(all_attention_matrices):
            if hasattr(step_attentions, '__len__') and len(step_attentions) > 0:
                seq_len = step_attentions[0].shape[-1]
            else:
                continue
            
            step_tokens = tokens[:seq_len]
            is_complete, _ = self._analyze_generation_step_completeness(step_tokens, step_idx, dim1, dim2)
            
            if is_complete:
                valid_steps.append(step_attentions)
                step_tokens_list.append(step_tokens)
        
        if not valid_steps:
            return None
        step_analyses = []
        
        for step_idx, (step_attentions, step_tokens) in enumerate(zip(valid_steps, step_tokens_list)):
            target_indices = self._find_target_tokens_in_step(step_tokens, input_word, dim1, dim2)
            step_analysis = self._analyze_single_step_attention(step_attentions, step_tokens, target_indices, step_idx)
            step_analyses.append(step_analysis)
        
        final_analysis = self._aggregate_step_analyses(step_analyses, word, input_word, dim1, dim2, answer, response, tokens)
        return final_analysis

    def _analyze_single_step_attention(self, step_attentions:list[tuple[torch.Tensor, ...]], step_tokens:list[str], target_indices:dict[str, list[int]], step_idx:int):
        num_layers = len(step_attentions)  # 28 layers
        num_heads = step_attentions[0].shape[1] if step_attentions else 0  # 28 heads
        seq_len = step_attentions[0].shape[-1] if step_attentions else 0  # sequence length for this step
        
        # Calculate sliding window offset
        sliding_window_offset = 0
        if len(step_tokens) > seq_len:
            sliding_window_offset = len(step_tokens) - seq_len
            step_tokens = step_tokens[sliding_window_offset:]
        elif len(step_tokens) < seq_len:
            step_tokens.extend([''] * (seq_len - len(step_tokens)))
        
        # Initialize matrices
        word_dim1_raw_matrix = torch.zeros(num_layers, num_heads)
        word_dim2_raw_matrix = torch.zeros(num_layers, num_heads)
        word_dim1_norm_matrix = torch.zeros(num_layers, num_heads)
        word_dim2_norm_matrix = torch.zeros(num_layers, num_heads)
        
        # Self-attention analysis
        for layer_idx in range(num_layers): # layer_attention shape: [batch, heads, seq, seq]
            layer_attention = step_attentions[layer_idx]
            layer_attention = layer_attention[0]
            
            for head_idx in range(num_heads): # head_attention shape: [seq, seq]
                head_attention = layer_attention[head_idx]
                word_dim1_raw, _ = self._calculate_step_attention_scores(
                    head_attention, step_tokens, target_indices['dim1'], target_indices['word'], step_idx, head_idx
                )
                word_dim2_raw, _ = self._calculate_step_attention_scores(
                    head_attention, step_tokens, target_indices['dim2'], target_indices['word'], step_idx, head_idx
                )
                word_dim1_norm = self._calculate_step_normalized_attention(
                    head_attention, target_indices['dim1'], target_indices['word']
                )
                word_dim2_norm = self._calculate_step_normalized_attention(
                    head_attention, target_indices['dim2'], target_indices['word']
                )
                
                # Store in matrices
                word_dim1_raw_matrix[layer_idx, head_idx] = word_dim1_raw
                word_dim2_raw_matrix[layer_idx, head_idx] = word_dim2_raw
                word_dim1_norm_matrix[layer_idx, head_idx] = word_dim1_norm
                word_dim2_norm_matrix[layer_idx, head_idx] = word_dim2_norm
        
        return {
            'step_idx': step_idx,
            'step_tokens': step_tokens,
            'target_indices': target_indices,
            'word_dim1_raw_matrix': word_dim1_raw_matrix,
            'word_dim2_raw_matrix': word_dim2_raw_matrix,
            'word_dim1_norm_matrix': word_dim1_norm_matrix,
            'word_dim2_norm_matrix': word_dim2_norm_matrix,
        }

    def _aggregate_step_analyses(self, step_analyses:list[dict], word:str, input_word:str, dim1:str, dim2:str, answer:str, response:str, tokens:list[str]):
        if not step_analyses:
            return None
        
        # Aggregate matrices across steps
        num_steps = len(step_analyses)
        num_layers = step_analyses[0]['word_dim1_raw_matrix'].shape[0]
        num_heads = step_analyses[0]['word_dim1_raw_matrix'].shape[1]

        # Initialize aggregated matrices
        agg_word_dim1_raw = torch.zeros(num_layers, num_heads)
        agg_word_dim2_raw = torch.zeros(num_layers, num_heads)
        agg_word_dim1_norm = torch.zeros(num_layers, num_heads)
        agg_word_dim2_norm = torch.zeros(num_layers, num_heads)
        
        # Sum across steps
        for step_analysis in step_analyses:
            agg_word_dim1_raw += step_analysis['word_dim1_raw_matrix']
            agg_word_dim2_raw += step_analysis['word_dim2_raw_matrix']
            agg_word_dim1_norm += step_analysis['word_dim1_norm_matrix']
            agg_word_dim2_norm += step_analysis['word_dim2_norm_matrix']
        
        # Average across steps
        agg_word_dim1_raw /= num_steps
        agg_word_dim2_raw /= num_steps
        agg_word_dim1_norm /= num_steps
        agg_word_dim2_norm /= num_steps
        
        # Aggregate scores for different levels
        aggregation_levels = ["all", "layers", "heads", "individual"]
        
        word_dim1_raw_agg = {level: self._aggregate_step_scores(agg_word_dim1_raw, level) for level in aggregation_levels}
        word_dim2_raw_agg = {level: self._aggregate_step_scores(agg_word_dim2_raw, level) for level in aggregation_levels}
        word_dim1_norm_agg = {level: self._aggregate_step_scores(agg_word_dim1_norm, level) for level in aggregation_levels}
        word_dim2_norm_agg = {level: self._aggregate_step_scores(agg_word_dim2_norm, level) for level in aggregation_levels}
        
        # Create final analysis dictionary
        final_analysis = {
            'word': word,
            'input_word': input_word,
            'dimension1': dim1,
            'dimension2': dim2,
            'answer': answer,
            'response': response,
            'tokens': tokens,
            'num_steps_analyzed': num_steps,
            
            # Self-attention raw scores
            'word_dim1_raw_all': word_dim1_raw_agg['all'],
            'word_dim1_raw_layers': word_dim1_raw_agg['layers'],
            'word_dim1_raw_heads': word_dim1_raw_agg['heads'],
            'word_dim1_raw_individual': word_dim1_raw_agg['individual'],
            
            'word_dim2_raw_all': word_dim2_raw_agg['all'],
            'word_dim2_raw_layers': word_dim2_raw_agg['layers'],
            'word_dim2_raw_heads': word_dim2_raw_agg['heads'],
            'word_dim2_raw_individual': word_dim2_raw_agg['individual'],
            
            # Self-attention normalized scores
            'word_dim1_norm_all': word_dim1_norm_agg['all'],
            'word_dim1_norm_layers': word_dim1_norm_agg['layers'],
            'word_dim1_norm_heads': word_dim1_norm_agg['heads'],
            'word_dim1_norm_individual': word_dim1_norm_agg['individual'],
            
            'word_dim2_norm_all': word_dim2_norm_agg['all'],
            'word_dim2_norm_layers': word_dim2_norm_agg['layers'],
            'word_dim2_norm_heads': word_dim2_norm_agg['heads'],
            'word_dim2_norm_individual': word_dim2_norm_agg['individual'],
            
            # Raw matrices for detailed analysis
            'word_dim1_raw_matrix': agg_word_dim1_raw.cpu().numpy(),
            'word_dim2_raw_matrix': agg_word_dim2_raw.cpu().numpy(),
            'word_dim1_norm_matrix': agg_word_dim1_norm.cpu().numpy(),
            'word_dim2_norm_matrix': agg_word_dim2_norm.cpu().numpy(),
            
            # Step-by-step analysis for debugging
            'step_analyses': step_analyses,
        }
        
        return final_analysis

    def save_generation_attention_analysis(self, generation_analysis, dim1:str, dim2:str, answer:str, word_tokens:str, option_tokens:str, lang="en", tokens=None, response=None):
        analysis_data = {
            "generation_analysis": generation_analysis,
            "dimension1": dim1,
            "dimension2": dim2,
            "answer": answer,
            "word_tokens": word_tokens,
            "option_tokens": option_tokens,
            "tokens": tokens,
            "response": response,
            "input_word": generation_analysis.get('input_word', ''),
            "analysis_type": "generation_attention"
        }
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, "generation_attention")
        os.makedirs(output_dir, exist_ok=True)
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dim1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dim2))
        save_path = os.path.join(output_dir, f"{safe_word}_{safe_dim1}_{safe_dim2}_generation_analysis.pkl")
        with open(save_path, "wb") as f:
            pkl.dump(analysis_data, f)
        print(f"Generation attention analysis saved to: {save_path}")

    def save_generation_attention_matrix(self, all_attention_matrices, dim1:str, dim2:str, answer:str, word_tokens:str, input_word:str, option_tokens:str, lang="en", tokens=None, current_input_ids=None, all_tokens=None):
        if current_input_ids is not None and all_tokens is not None:
            input_length = len(all_tokens)
            input_ids = current_input_ids[0][:input_length]
            response_ids = current_input_ids[0][input_length:]
            input_text = self.processor.tokenizer.decode(input_ids)
            response = self.processor.tokenizer.decode(response_ids)
            full_text = self.processor.tokenizer.decode(current_input_ids[0])
        else:
            input_text = "unknown"
            response = "unknown"
            full_text = "unknown"
        filtered_attention_matrices = []
        relevant_indices_list = []
        for step, step_attentions in enumerate(all_attention_matrices):
            layer_attention = step_attentions[0]
            current_seq_len = layer_attention.shape[-1]
            tmp_tokens = tokens[:current_seq_len]
            relevant_indices = self.extract_relevant_token_indices(tmp_tokens, dim1, dim2, input_word, word=word_tokens)
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
            "dimension1": dim1,
            "dimension2": dim2,
            "answer": answer,
            "word_tokens": word_tokens,
            "option_tokens": option_tokens,
            "tokens": tokens,
            "input_text": input_text,
            "response": response,
            "full_text": full_text,
            "analysis_type": "generation_attention_matrix"
        }
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, "generation_attention", f"{self.dim1}_{self.dim2}")
        os.makedirs(output_dir, exist_ok=True)
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        safe_dim1 = re.sub(r'[^\w\-_.]', '_', str(dim1))
        safe_dim2 = re.sub(r'[^\w\-_.]', '_', str(dim2))
        save_path = os.path.join(output_dir, f"{safe_word}_{safe_dim1}_{safe_dim2}.pkl")
        with open(save_path, "wb") as f:
            pkl.dump(matrix_data, f)
        print(f"Generation attention matrix saved to: {save_path}")
        return save_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2.5-Omni Semantic Dimension Attention Heatmap Visualization")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-Omni-7B", help="Model path (default: Qwen/Qwen2.5-Omni-7B)")
    parser.add_argument('--data-path', type=str, default="data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json",
                       help="Path to semantic dimension data JSON file")
    parser.add_argument('--output-dir', type=str, default="results/experiments/understanding/attention_heatmap/nat",
                       help="Output directory for heatmaps and matrices")
    parser.add_argument('--data-type', type=str, default="audio", choices=["audio", "original", "romanized", "ipa"],
                       help="Data type to process")
    parser.add_argument('--max-tokens', type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.0, help="Sampling temperature")
    parser.add_argument('--max-samples', type=int, default=None, help="Maximum number of samples to process (default: all)")
    parser.add_argument('--languages', nargs='+', default=["en", "fr", "ko", "ja"], help="Languages to process")
    parser.add_argument('--flip', action='store_true', help="Flip dim1 and dim2 in prompts and outputs")
    parser.add_argument('--constructed', '-c', action='store_true', help="Use constructed words as dataset")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for inference")
    args = parser.parse_args()
    max_samples:int = args.max_samples
    print(f"Data type: {args.data_type}")

    if args.constructed:
        data_path = "data/processed/art/semantic_dimension/semantic_dimension_binary_gt.json"
        output_dir = "results/experiments/understanding/attention_heatmap/con"
    else:
        data_path = args.data_path
        output_dir = args.output_dir
    
    visualizer = QwenOmniSemanticDimensionVisualizer(
        model_path=args.model,
        data_path=data_path,
        output_dir=output_dir,
        exp_type="semantic_dimension",
        data_type=args.data_type,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        flip=args.flip,
        constructed=args.constructed
    )
    if args.constructed:
        languages = ["art"]
    else:
        languages = ["en", "fr", "ja", "ko"]
        # languages = ["ko"]
    total_num_of_dimensions = 0
    total_num_of_words = 0
    total_num_of_words_per_language = {lang: 0 for lang in languages}
    start_index = 0

    batch_size = args.batch_size
    for lang in languages:
        print(f"\nProcessing language: {lang}")
        lang_data = visualizer.data[lang]
        print(f"Found {len(lang_data)} samples for language {lang}")
        
        if max_samples:
            lang_data = lang_data[start_index:max_samples]
            print(f"Limiting to {len(lang_data)} samples")

        # Prepare batch lists
        words, langs, constructed_prompts, dim1s, dim2s, answers, datas, dimension_names = [], [], [], [], [], [], [], []
        for sample_idx, sample in enumerate(lang_data):
            for dimension_name in sample.get("dimensions", {}):
                constructed_prompt, dim1, dim2, answer, word, dim_name = visualizer.prmpt_dims_answrs(
                    visualizer.prompts, sample, dimension_name
                )
                words.append(word)
                langs.append(lang)
                constructed_prompts.append(constructed_prompt)
                dim1s.append(dim1)
                dim2s.append(dim2)
                answers.append(answer)
                datas.append(sample)
                dimension_names.append(dimension_name)
        # Batch inference
        num_samples = len(words)
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            visualizer.inference_with_hooks_batch(
                words[batch_start:batch_end],
                langs[batch_start:batch_end],
                constructed_prompts[batch_start:batch_end],
                dim1s[batch_start:batch_end],
                dim2s[batch_start:batch_end],
                answers[batch_start:batch_end],
                datas[batch_start:batch_end],
                dimension_names[batch_start:batch_end],
                batch_size=batch_size
            )
            total_num_of_dimensions += (batch_end - batch_start)
        total_num_of_words += len(lang_data)
        total_num_of_words_per_language[lang] += len(lang_data)
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
