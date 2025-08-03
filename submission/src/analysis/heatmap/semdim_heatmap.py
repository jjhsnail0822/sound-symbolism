# Model : Qwen2.5-Omni-7B
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 30 --data-type ipa
# python src/analysis/heatmap/semdim_heatmap.py --max-samples 2 --data-type ipa --constructed
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


language = ["en", "fr", "ko", "ja"]
data_types = ["original", "romanized", "ipa", "audio"]
data_path = "data/processed/nat/semantic_dimension/semantic_dimension_binary_gt.json"
prompt_path = "data/prompts/prompts.json"
problem_per_language = 10
SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
SYSTEM_TEMPLATE = {
    "role": "system",
    "content": [
        {"type": "text", "text": SYSTEM_PROMPT}
    ],
}

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
            # attn_implementation="flash_attention_2",
            attn_implementation="eager",
        )
        self.model.disable_talker()
        self.load_base_prompt()
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
    
    def prmpt_dims_answrs(self, prompt:str, data, dimension_name:str=None):
        if self.data_type != "audio":
            data_type_key = "word" if self.data_type == "original" \
                else "romanization" if self.data_type == "romanized" \
                else "ipa"
            if data_type_key not in data:
                raise KeyError(f"Data type '{data_type_key}' not found in data: {data.keys()}")
        dimension_info = data["dimensions"][dimension_name]
        dim1 = dimension_name.split("-")[0]
        dim2 = dimension_name.split("-")[1]
        answer = dimension_info["answer"]
        # Flip if needed
        if self.flip:
            dim1, dim2 = dim2, dim1
        if self.data_type == "audio":
            if self.constructed:
                word = f"data/processed/art/tts/{data['word']}.wav"
            else:
                word = f"data/processed/nat/tts/{data['language']}/{data['word']}.wav"
            # For audio type, we need to format the prompt with dimension info first
            if "{audio}" in prompt:
                prompt_parts = prompt.split("{audio}")
                text_before = prompt_parts[0].format(
                    word=data['word'],
                    dimension1=dim1,
                    dimension2=dim2
                )
                text_after = prompt_parts[1].format(
                    word=data['word'],
                    dimension1=dim1,
                    dimension2=dim2
                )
                constructed_prompt = [
                    # {"type": "text", "text": prompt.split("{audio}")[0]},
                    {"type": "text", "text": text_before},
                    {"type": "audio", "audio": word},
                    # {"type": "text", "text": prompt.split("{audio}")[1].format(dimension1=dimension1, dimension2=dimension2)},
                    {"type": "text", "text": text_after},
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
    
    def create_conversation(self, prompt, data):
        if self.data_type == "audio":
            audio_path = f'data/processed/nat/tts/{data["language"]}/{data["word"]}.wav'
            # print(f"[DEBUG] create_conversation audio_path: {audio_path}, Exists: {os.path.exists(audio_path)}")
            # print(f"[DEBUG] conversation: {prompt}")
            if isinstance(prompt, list):
                conversation = [
                    SYSTEM_TEMPLATE,
                    {
                        "role": "user",
                        "content": prompt
                    },
                ]
            else:
                # breakpoint() # EDIT LATER
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
        # breakpoint()
        return conversation
    
    def trim_silence_from_audio(self, audio_data, threshold=0.01):
        """Remove silence from the beginning of audio"""
        if audio_data is None or len(audio_data) == 0:
            return audio_data
            
        # Find the first non-silent sample
        non_zero_indices = np.nonzero(np.abs(audio_data) > threshold)[0]
        
        if len(non_zero_indices) == 0:
            print(f"[WARNING] Audio is completely silent!")
            return audio_data
            
        start_idx = non_zero_indices[0]
        trimmed_audio = audio_data[start_idx:]
        
        # print(f"[DEBUG] Audio trimmed: {len(audio_data)} -> {len(trimmed_audio)} samples")
        # print(f"[DEBUG] Trimmed audio first 10 values: {trimmed_audio[:10]}")
        
        return trimmed_audio

    def get_attention_matrix(self, prompt, data:dict):
        conversation = self.create_conversation(prompt, data)
        
        USE_AUDIO_IN_VIDEO = True
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # Debug: Check audio file directly before process_mm_info
        if self.data_type == "audio":
            if self.constructed:
                audio_path = f"data/processed/art/tts/{data['word']}.wav"
            else:
                audio_path = f'data/processed/nat/tts/{data["language"]}/{data["word"]}.wav'
            # print(f"[DEBUG] Checking audio file directly: {audio_path}")
            
            # Check file size
            import os
            file_size = os.path.getsize(audio_path)
            # print(f"[DEBUG] Audio file size: {file_size} bytes")
            
            # Try to load with librosa directly
            try:
                import librosa
                audio_data, sr = librosa.load(audio_path, sr=16000)
                # print(f"[DEBUG] Direct librosa load - shape: {audio_data.shape}, sample_rate: {sr}")
                # print(f"[DEBUG] Direct librosa load - first 10 values: {audio_data[:10]}")
                # print(f"[DEBUG] Direct librosa load - min: {audio_data.min()}, max: {audio_data.max()}, mean: {audio_data.mean()}")
                
                # Check for non-zero audio content
                non_zero_indices = np.nonzero(np.abs(audio_data) > 0.01)[0]  # Threshold for silence
                if len(non_zero_indices) > 0:
                    start_idx = non_zero_indices[0]
                    end_idx = non_zero_indices[-1] + 1
                    # print(f"[DEBUG] Non-zero audio content from index {start_idx} to {end_idx}")
                    # print(f"[DEBUG] Non-zero audio length: {end_idx - start_idx} samples ({((end_idx - start_idx) / sr):.3f}s)")
                    # print(f"[DEBUG] Non-zero audio values: {audio_data[start_idx:start_idx+10]}")
                else:
                    # print(f"[DEBUG] WARNING: No significant audio content found!")
                    pass
                    
            except Exception as e:
                print(f"[DEBUG] Direct librosa load failed: {e}")
            
            # Debug: Check conversation structure that will be passed to process_audio_info
            # print(f"[DEBUG] Conversation structure for process_audio_info:")
            # for i, msg in enumerate(conversation):
            #     print(f"[DEBUG] Message {i}: {msg}")
            #     if isinstance(msg.get('content'), list):
            #         for j, ele in enumerate(msg['content']):
            #             print(f"[DEBUG]   Element {j}: {ele}")
            #             if ele.get('type') == 'audio':
            #                 print(f"[DEBUG]     Audio path: {ele.get('audio')}")
            #                 print(f"[DEBUG]     Audio start: {ele.get('audio_start', 0.0)}")
            #                 print(f"[DEBUG]     Audio end: {ele.get('audio_end', None)}")
        
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        if self.data_type == "audio" and (audios is None or len(audios) == 0):
            # print(f"[DEBUG] process_mm_info audios: {audios}, images: {images}, videos: {videos}")
            if self.data_type == "audio":
                if not audios or len(audios) == 0:
                    print(f"[WARNING] No audio loaded for {data['word']}")
                    # Instead of falling back to text-only, raise an error to ensure audio is properly loaded
                    raise ValueError(f"Audio file not found or could not be loaded for {data['word']}. Expected path: data/processed/nat/tts/{data['language']}/{data['word']}.wav")
            # Remove the fallback to text-only conversation
            # conversation_text_only = [
            #     SYSTEM_TEMPLATE,
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "text", "text": f"Given a spoken word '{data['word']}', which semantic feature best describes the word based on auditory impression?"}
            #         ],
            #     },
            # ]
            # text = self.processor.apply_chat_template(conversation_text_only, add_generation_prompt=True, tokenize=False)
            # audios, images, videos = process_mm_info(conversation_text_only, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        # Debug: Check what process_mm_info returned
        if self.data_type == "audio" and audios is not None:
            # print(f"[DEBUG] process_mm_info returned audios: {len(audios)} items")
            for i, audio in enumerate(audios):
                # print(f"[DEBUG] Audio {i} - shape: {audio.shape}, dtype: {audio.dtype}")
                # print(f"[DEBUG] Audio {i} - first 10 values: {audio[:10]}")
                # print(f"[DEBUG] Audio {i} - min: {audio.min()}, max: {audio.max()}, mean: {audio.mean()}")
                
                # Apply silence trimming
                trimmed_audio = self.trim_silence_from_audio(audio)
                audios[i] = trimmed_audio
        
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        # print(f"[DEBUG] processor inputs: {inputs.keys()}")
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with torch.no_grad():
            # print(f"[DEBUG] Model input_ids: {inputs['input_ids']}")
            # print(f"[DEBUG] Model audio: {audios}")
            thinker_model = self.model.thinker.model
            outputs = thinker_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_attentions=True, return_dict=True)
        
        attentions = outputs.attentions
        tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return attentions, tokens, inputs
    
    def _clean_token(self, token):
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

    def extract_relevant_token_indices(self, tokens, dimension1, dimension2, input_word, word=None):
        # tokens[:current_seq_len], dimension1, dimension2, word
        word_tag_matches = self.find_tag_spans(tokens, 'WORD')
        second_word_tag_span = word_tag_matches[1]
        semdim_tag_matches = self.find_tag_spans(tokens, 'SEMANTICDIMENSION')
        semdim_span = semdim_tag_matches[0]
        options_tag_matches = self.find_tag_spans(tokens, 'OPTIONS')
        options_span = options_tag_matches[0] if options_tag_matches else None
        word_indices = []
        if word is not None:
            word_subtokens = self.processor.tokenizer.tokenize(input_word)
            word_matches = self.find_subtoken_sequence_indices(tokens, word_subtokens)
            for match in word_matches:
                word_indices.extend(match)
        word_span_start = second_word_tag_span[-1] + 1
        word_span_end = semdim_span[0]
        word_span = list(range(word_span_start, word_span_end))
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
        return relevant_indices
    
    def save_matrix(self, attention_matrix, dimension1, dimension2, answer, word_tokens, option_tokens, layer_type="self", lang="en", tokens=None, relevant_indices=None):
        # Flip if needed
        dim1, dim2 = dimension1, dimension2
        matrix_data = {"attention_matrix": attention_matrix, "dimension1": dim1, "dimension2": dim2, "answer": answer, "word_tokens": word_tokens, "option_tokens": option_tokens, "tokens": tokens, "relevant_indices": relevant_indices}
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, "self_attention", f"{dim1}_{dim2}")
        os.makedirs(output_dir, exist_ok=True)
        safe_word = re.sub(r'[^\w\-_.]', '_', str(word_tokens))
        save_path = os.path.join(output_dir, f"{safe_word}_{dim1}_{dim2}_{layer_type}.pkl")
        with open(save_path, "wb") as f:
            pkl.dump(matrix_data, f)

    def find_token_indices(self, tokens, target_tokens):
        indices = []
        
        def is_special_token(token):
            return '<' in token and '>' in token
        
        # Process each target
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
    
    def inference_with_hooks(self, word, lang, constructed_prompt, dim1, dim2, answer, data, dimension_name):
        attentions, tokens, _ = self.get_attention_matrix(constructed_prompt, data)
        if self.data_type == "audio":
            input_word = "<|AUDIO|>"
        elif self.data_type == "original":
            input_word = data['word']
        elif self.data_type == "ipa":
            input_word = data["ipa"]
        elif self.data_type == "romanized":
            input_word = data["romanization"]

        relevant_indices = self.extract_relevant_token_indices(tokens, dim1, dim2, input_word, word=data['word'])
        if isinstance(attentions, tuple):
            attention_matrix = attentions[0]
        else:
            attention_matrix = attentions
        attn_filtered = attention_matrix[:, :, relevant_indices][:, :, :, relevant_indices]
        self.save_matrix(attn_filtered, dim1, dim2, answer, data['word'], [dim1, dim2], "self", lang, tokens, relevant_indices)
        # print(f"Saved filtered attention matrix for {data['word']} - {dim1}-{dim2} to pickle file")
        
        # print(f"Extracting generation attention for {data['word']} - {dim1}-{dim2}...")
        generation_attentions, generation_tokens, _, final_input_ids, response, input_length = self.get_generation_attention_matrix(
            constructed_prompt, data, max_new_tokens=self.max_tokens
        )
        # print(f"Generation attentions length: {len(generation_attentions)}")
        data_type_key = {"audio": "audio", "original": "word", "romanized": "romanization", "ipa": "ipa"}
        
        if self.data_type != "audio":
            input_word = data[data_type_key[self.data_type]]
        else:
            input_word = data["word"]
            
        generation_analysis = self.extract_generation_attention_analysis(
            generation_attentions, generation_tokens, response, answer, lang, dim1, dim2, data['word'], input_word, input_length
        )
        
        self.save_generation_attention_matrix(
            generation_attentions, dim1, dim2, answer, data['word'], input_word, [dim1, dim2], lang, generation_tokens, final_input_ids, generation_tokens
        )

        self.save_generation_attention_analysis(
            generation_analysis, dim1, dim2, answer, data['word'], [dim1, dim2], lang, generation_tokens, response
        )
        print(f"Saved generation attention analysis for {data['word']} - {dim1}-{dim2}")
        # breakpoint()

    def get_generation_attention_matrix(self, prompt, data: dict, max_new_tokens: int = 32):
        """Extract attention matrix during text generation (autoregressive decoding)"""

        conversation = self.create_conversation(prompt, data)
        
        USE_AUDIO_IN_VIDEO = True
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        # breakpoint()
        if self.data_type == "audio" and (audios is None or len(audios) == 0):
            print(f"[WARNING] No audio loaded for {data['word']}")
            # Instead of falling back to text-only, raise an error to ensure audio is properly loaded
            raise ValueError(f"Audio file not found or could not be loaded for {data['word']}. Expected path: data/processed/nat/tts/{data['language']}/{data['word']}.wav")
            # Remove the fallback to text-only conversation
            # conversation_text_only = [
            #     SYSTEM_TEMPLATE,
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "text", "text": f"Given a spoken word '{data['word']}', which semantic feature best describes the word based on auditory impression?"}
            #         ],
            #     },
            # ]
            # text = self.processor.apply_chat_template(conversation_text_only, add_generation_prompt=True, tokenize=False)
            # audios, images, videos = process_mm_info(conversation_text_only, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        
        # Apply silence trimming for generation as well
        if self.data_type == "audio" and audios is not None:
            for i, audio in enumerate(audios):
                trimmed_audio = self.trim_silence_from_audio(audio)
                audios[i] = trimmed_audio
        # breakpoint()
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
        # print("\n[DEBUG] === Model Input Debugging ===")
        # print("[DEBUG] Prompt text (first 500 chars):", text[:500])
        # if self.data_type == "audio":
        #     print("[DEBUG] Audio: type:", type(audios), "len:", len(audios) if audios is not None else None)
        #     if audios and len(audios) > 0:
        #         print("[DEBUG] Audio[0] shape:", audios[0].shape, "dtype:", audios[0].dtype)
        # print("[DEBUG] inputs keys:", list(inputs.keys()))
        # print("[DEBUG] input_ids shape:", inputs['input_ids'].shape)
        # print("[DEBUG] attention_mask shape:", inputs['attention_mask'].shape)
        # if 'input_features' in inputs:
        #     print("[DEBUG] input_features shape:", inputs['input_features'].shape)
        # if 'feature_attention_mask' in inputs:
        #     print("[DEBUG] feature_attention_mask shape:", inputs['feature_attention_mask'].shape)
        # print("[DEBUG] input_ids (first 20):", inputs['input_ids'][0][:20])
        # print("[DEBUG] === End Model Input Debugging ===\n")
        
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
                # print(f"\n=== Generation Step {step} ===")
                # print(f"Step {step}: Input sequence length: {len(current_input_ids[0])}")
                # print(f"Step {step}: Input tokens: {self.processor.tokenizer.convert_ids_to_tokens(current_input_ids[0])[-10:]}")
                
                # For generation, we need to pass the input_features (processed audio) to the model
                if self.data_type == "audio" and 'input_features' in inputs:
                    # print("[DEBUG] Passing to model: input_ids shape:", current_input_ids.shape)
                    # print("[DEBUG] Passing to model: attention_mask shape:", current_attention_mask.shape)
                    # print("[DEBUG] Passing to model: input_features shape:", inputs['input_features'].shape)
                    # print("[DEBUG] Passing to model: feature_attention_mask shape:", inputs['feature_attention_mask'].shape)
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

                # Store attention matrices for this step
                all_attention_matrices.append(attentions)
                
                # Get next token
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                next_token_id = next_token_id.to(current_input_ids.device)
                
                # Add new token to sequence
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token_id, device=current_input_ids.device)], dim=-1)
                
                # Store new token
                new_token = self.processor.tokenizer.convert_ids_to_tokens(next_token_id[0])
                all_tokens.append(new_token)
                
                # Check if generation should stop
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
        # print(f"[Debug] Response: {response}")
        # breakpoint()
        
        # Check the last token
        last_token_id = current_input_ids[0][-1].item()
        last_token = self.processor.tokenizer.convert_ids_to_tokens(last_token_id)
        return all_attention_matrices, final_tokens, inputs, current_input_ids, response, input_length

    def _analyze_generation_step_completeness(self, step_tokens, step_idx, dimension1, dimension2):
        """Analyze if a generation step is complete (has proper ending)"""
        if not step_tokens:
            return False, "Empty tokens"
        
        # Check for EOS token or proper sentence ending
        last_tokens = step_tokens[-1:] if len(step_tokens) >= 1 else step_tokens
        
        # Check for EOS token
        eos_found = any(token in ['1', '2'] for token in last_tokens)
        
        if eos_found:
            return True, "Complete with EOS"
        
        # Check if it looks like a complete answer
        last_token = step_tokens[-1] if step_tokens else ""
        if last_token in [dimension1, dimension2] or last_token.lower() in [dimension1.lower(), dimension2.lower()]:
            return True, "Complete answer"
        
        return True, "Appears complete"

    def _find_target_tokens_in_step(self, step_tokens, word, dimension1, dimension2):
        """Find target token indices for a specific generation step"""
        target_indices = {"word": [], "dim1": [], "dim2": []}
        
        def remove_indices(indices: list[list[int]], threshold: int):
            """Remove sublists that contain any value below or equal to threshold"""
            # Create a new list with only valid sublists
            filtered_indices = []
            for sublist in indices:
                # Check if all values in the sublist are above threshold
                if all(i > threshold for i in sublist):
                    filtered_indices.append(sublist)
            return filtered_indices
        
        # print(f"Step tokens length: {len(step_tokens)}")
        # print(f"Step tokens: {step_tokens[-10:]}")  # Show last 10 tokens
        
        if self.data_type == "audio":
            audio_token = "<|AUDIO|>"
            for i, token in enumerate(step_tokens):
                if self._clean_token(token) == self._clean_token(audio_token):
                    target_indices['word'].append(i)
        else:
            # For non-audio, find the actual word tokens
            word_subtokens = self.processor.tokenizer.tokenize(word)
            word_matches = self.find_subtoken_sequence_indices(step_tokens, word_subtokens)
            word_matches = remove_indices(word_matches, 60)
            for match in word_matches:
                target_indices['word'].extend(match)
                
        # Find dimension indices
        dim1_subtokens = self.processor.tokenizer.tokenize(dimension1)
        dim1_matches = self.find_subtoken_sequence_indices(step_tokens, dim1_subtokens)
        dim1_matches = remove_indices(dim1_matches, 77)
        for match in dim1_matches:
            target_indices['dim1'].extend(match)
        
        dim2_subtokens = self.processor.tokenizer.tokenize(dimension2)
        dim2_matches = self.find_subtoken_sequence_indices(step_tokens, dim2_subtokens)
        dim1_max_index = max(target_indices['dim1'])
        dim2_matches = remove_indices(dim2_matches, dim1_max_index)
        for match in dim2_matches:
            target_indices['dim2'].extend(match)
        
        # Remove duplicates and sort
        for key in target_indices:
            target_indices[key] = sorted(set(target_indices[key]))
            
        # Debug: Print found indices
        # print(f"Found indices - word: {target_indices['word']}, dim1: {target_indices['dim1']}, dim2: {target_indices['dim2']}")
        
        return target_indices

    def _calculate_step_attention_scores(self, attention_matrix, tokens, query_indices, key_indices, step_idx, head_idx, sliding_window_offset=0):
        """Calculate attention scores for a specific step and head"""
        total_score = 0.0
        valid_pairs = 0
        
        # Get the sequence length for this step
        seq_len = attention_matrix.shape[0]  # Should be equal to shape[1] for self-attention
        
        # Debug: Print token information
        # print(f"Step {step_idx}, Head {head_idx}: Query indices: {query_indices}")
        # print(f"Step {step_idx}, Head {head_idx}: Key indices: {key_indices}")
        # print(f"Step {step_idx}, Head {head_idx}: Query tokens: {[tokens[i] for i in query_indices if i < len(tokens)]}")
        # print(f"Step {step_idx}, Head {head_idx}: Key tokens: {[tokens[i] for i in key_indices if i < len(tokens)]}")
        
        # Calculate attention scores: query_indices -> key_indices
        # attention_matrix[q_idx, k_idx] = how much token q_idx attends to token k_idx
        for q_idx in query_indices:
            if q_idx < seq_len:
                for k_idx in key_indices:
                    if k_idx < seq_len:
                        # Check if this is a valid attention (causal attention: q_idx can only attend to k_idx <= q_idx)
                        if k_idx <= q_idx:  # Causal attention constraint
                            score = attention_matrix[q_idx, k_idx].item()
                            total_score += score
                            valid_pairs += 1
                            # print(f"Step {step_idx}, Head {head_idx}: Score {score:.4f} at ({q_idx}, {k_idx}) - {tokens[q_idx]} -> {tokens[k_idx]}")
                        else:
                            pass
                            # print(f"Step {step_idx}, Head {head_idx}: Skipping invalid attention ({q_idx}, {k_idx}) - causal constraint")
                    else:
                        pass
                        # print(f"Warning: Key index {k_idx} out of bounds for step {step_idx} (seq_len: {seq_len})")
            else:
                pass
                # print(f"Warning: Query index {q_idx} out of bounds for step {step_idx} (seq_len: {seq_len})")
        
        # print(f"Step {step_idx}, Head {head_idx}: Total score: {total_score:.4f}, Valid pairs: {valid_pairs}")
        # breakpoint()
        return total_score, valid_pairs

    def _calculate_step_normalized_attention(self, attention_matrix, query_indices, key_indices, sliding_window_offset=0):
        """Calculate normalized attention distribution for a specific step"""
        total_query_attention = 0.0
        target_attention = 0.0
        
        # Get the sequence length for this step
        seq_len = attention_matrix.shape[0]  # Should be equal to shape[1] for self-attention
        
        for q_idx in query_indices:
            if q_idx < seq_len:
                # Sum of all attention from this query token (only valid causal attention)
                row_sum = attention_matrix[q_idx, :q_idx+1].sum().item()  # Only attend to tokens <= q_idx
                total_query_attention += row_sum
                # Sum of attention to target key tokens
                for k_idx in key_indices:
                    if k_idx < seq_len and k_idx <= q_idx:  # Causal attention constraint
                        target_score = attention_matrix[q_idx, k_idx].item()
                        target_attention += target_score
        
        if total_query_attention > 0:
            return target_attention / total_query_attention
        return 0.0

    def _aggregate_step_scores(self, scores_matrix, aggregation_type):
        """Aggregate scores across layers and heads for a specific step"""
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

    def extract_generation_attention_analysis(self, all_attention_matrices:list[tuple[torch.Tensor, ...]], tokens:list[str], response:str, answer:str, lang:str, dimension1:str, dimension2:str, word:str, input_word:str=None, input_length:int=None):
        """Analyze attention patterns during generation, focusing on output tokens"""
        # print(f"\n=== Generation Attention Analysis for {word} - {dimension1}-{dimension2} ===")
        # print(f"Generated text: {response}")
        # print(f"Expected answer: {answer}")
        # print(f"Input word: {input_word}")
        response = "1." if answer == dimension1 else "2."
        # Filter out incomplete generation steps
        valid_steps = []
        step_tokens_list = []
        
        for step_idx, step_attentions in enumerate(all_attention_matrices):
            # Get the sequence length for this step from attention matrix
            if hasattr(step_attentions, '__len__') and len(step_attentions) > 0:
                seq_len = step_attentions[0].shape[-1]  # Should be equal to shape[-2] for self-attention
                # print(f"Step {step_idx}: attention matrix shape indicates {seq_len} tokens")
            else:
                # print(f"Step {step_idx}: no attention matrices found")
                continue
            
            # Get tokens for this step based on the actual sequence length
            # Each step includes all tokens up to that point (input + generated)
            step_tokens = tokens[:seq_len]
            
            # print(f"Step {step_idx}: Using tokens 0 to {seq_len-1} (total: {len(step_tokens)})")
            # print(f"Step {step_idx}: Last few tokens: {step_tokens[-5:] if len(step_tokens) >= 5 else step_tokens}")
            
            # Analyze completeness
            # breakpoint()
            is_complete, reason = self._analyze_generation_step_completeness(step_tokens, step_idx, dimension1, dimension2)
            
            # print(f"Step {step_idx}: {'✓' if is_complete else '✗'} - {reason}")
            # print(f"  Tokens length: {len(step_tokens)}, Expected: {seq_len}")
            # print(f"  Tokens: {step_tokens[-5:] if len(step_tokens) > 5 else step_tokens}")
            
            if is_complete:
                valid_steps.append(step_attentions)
                step_tokens_list.append(step_tokens)
            # else:
            #     print(f"  Skipping incomplete step {step_idx}")
        
        if not valid_steps:
            # print("Warning: No valid generation steps found!")
            return None
        
        # print(f"\nValid steps: {len(valid_steps)} out of {len(all_attention_matrices)}")
        
        # Analyze each valid step
        step_analyses = []
        
        for step_idx, (step_attentions, step_tokens) in enumerate(zip(valid_steps, step_tokens_list)):
            # print(f"\n--- Analyzing Step {step_idx} ---")
            # print(f"Step tokens: {step_tokens}")
            
            # Find target tokens for this step
            target_indices = self._find_target_tokens_in_step(
                step_tokens, input_word, dimension1, dimension2
            )

            # print(f"Target indices found:")
            # print(f"  Word: {target_indices['word']} -> {[step_tokens[i] for i in target_indices['word'] if i < len(step_tokens)]}")
            # print(f"  Dim1: {target_indices['dim1']} -> {[step_tokens[i] for i in target_indices['dim1'] if i < len(step_tokens)]}")
            # print(f"  Dim2: {target_indices['dim2']} -> {[step_tokens[i] for i in target_indices['dim2'] if i < len(step_tokens)]}")
            
            # Analyze attention patterns for this step
            step_analysis = self._analyze_single_step_attention(
                step_attentions, step_tokens, target_indices, step_idx
            )
            
            step_analyses.append(step_analysis)
        
        # Aggregate results across all steps
        final_analysis = self._aggregate_step_analyses(step_analyses, word, input_word, dimension1, dimension2, answer, response, tokens)
        
        # print(f"\n=== Analysis Summary ===")
        # print(f"Self-attention scores - dim1: {final_analysis['word_dim1_raw_all']:.4f}, dim2: {final_analysis['word_dim2_raw_all']:.4f}")
        
        return final_analysis

    def _analyze_single_step_attention(self, step_attentions, step_tokens, target_indices, step_idx):
        # Get attention matrices
        # step_attentions is a tuple of tensors, each tensor is [batch, heads, seq, seq]
        # len(step_attentions) = number of layers (28)
        # step_attentions[0].shape = [batch, heads, seq, seq] = [1, 28, seq_len, seq_len]
        
        num_layers = len(step_attentions)  # 28 layers
        num_heads = step_attentions[0].shape[1] if step_attentions else 0  # 28 heads
        seq_len = step_attentions[0].shape[-1] if step_attentions else 0  # sequence length for this step
        
        # print(f"Debug - num_layers: {num_layers}, num_heads: {num_heads}, seq_len: {seq_len}")
        # print(f"Debug - step_attentions[0].shape: {step_attentions[0].shape if step_attentions else 'None'}")
        # print(f"Debug - step_tokens length: {len(step_tokens)}")
        
        # Calculate sliding window offset
        sliding_window_offset = 0
        if len(step_tokens) > seq_len:
            sliding_window_offset = len(step_tokens) - seq_len
            # print(f"Step {step_idx}: Sliding window detected! Offset: {sliding_window_offset}")
            # print(f"Step {step_idx}: Using tokens from index {sliding_window_offset} to {len(step_tokens)-1}")
            # Adjust step_tokens to match attention matrix size
            step_tokens = step_tokens[sliding_window_offset:]
        elif len(step_tokens) < seq_len:
            # print(f"Warning: Token length mismatch! step_tokens: {len(step_tokens)}, attention seq_len: {seq_len}")
            # Pad with empty tokens if needed (shouldn't happen in normal cases)
            step_tokens.extend([''] * (seq_len - len(step_tokens)))
        
        # Initialize matrices
        word_dim1_raw_matrix = torch.zeros(num_layers, num_heads)
        word_dim2_raw_matrix = torch.zeros(num_layers, num_heads)
        word_dim1_norm_matrix = torch.zeros(num_layers, num_heads)
        word_dim2_norm_matrix = torch.zeros(num_layers, num_heads)
        
        # Self-attention analysis
        for layer_idx in range(num_layers):
            # layer_attention shape: [batch, heads, seq, seq]
            layer_attention = step_attentions[layer_idx]
            # Remove batch dimension: [heads, seq, seq]
            layer_attention = layer_attention[0]
            
            for head_idx in range(num_heads):
                # head_attention shape: [seq, seq]
                head_attention = layer_attention[head_idx]
                
                # Calculate attention scores
                # word_dim1_raw, _ = self._calculate_step_attention_scores(
                #     head_attention, step_tokens, target_indices['word'], target_indices['dim1'], step_idx, head_idx, sliding_window_offset
                # )
                # word_dim2_raw, _ = self._calculate_step_attention_scores(
                #     head_attention, step_tokens, target_indices['word'], target_indices['dim2'], step_idx, head_idx, sliding_window_offset
                # )
                word_dim1_raw, _ = self._calculate_step_attention_scores(
                    head_attention, step_tokens, target_indices['dim1'], target_indices['word'], step_idx, head_idx, sliding_window_offset
                )
                word_dim2_raw, _ = self._calculate_step_attention_scores(
                    head_attention, step_tokens, target_indices['dim2'], target_indices['word'], step_idx, head_idx, sliding_window_offset
                )

                # word_dim1_norm = self._calculate_step_normalized_attention(
                #     head_attention, target_indices['word'], target_indices['dim1'], sliding_window_offset
                # )
                # word_dim2_norm = self._calculate_step_normalized_attention(
                #     head_attention, target_indices['word'], target_indices['dim2'], sliding_window_offset
                # )
                                                
                word_dim1_norm = self._calculate_step_normalized_attention(
                    head_attention, target_indices['dim1'], target_indices['word'], sliding_window_offset
                )
                word_dim2_norm = self._calculate_step_normalized_attention(
                    head_attention, target_indices['dim2'], target_indices['word'], sliding_window_offset
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

    def _aggregate_step_analyses(self, step_analyses, word, input_word, dimension1, dimension2, answer, response, tokens):
        """Aggregate analyses from all steps into final results"""
        
        if not step_analyses:
            return None
        
        # Aggregate matrices across steps
        num_steps = len(step_analyses)
        num_layers = step_analyses[0]['word_dim1_raw_matrix'].shape[0]
        num_heads = step_analyses[0]['word_dim1_raw_matrix'].shape[1]
        # breakpoint()
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
            'dimension1': dimension1,
            'dimension2': dimension2,
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

    def save_generation_attention_analysis(self, generation_analysis, dimension1, dimension2, answer, word_tokens, option_tokens, lang="en", tokens=None, response=None):
        # dim1, dim2 = (dimension2, dimension1) if self.flip else (dimension1, dimension2)
        dim1, dim2 = dimension1, dimension2
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

    def save_generation_attention_matrix(self, all_attention_matrices, dimension1, dimension2, answer, word_tokens, input_word, option_tokens, lang="en", tokens=None, current_input_ids=None, all_tokens=None):
        dim1, dim2 = (dimension2, dimension1) if self.flip else (dimension1, dimension2)
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
        output_dir = os.path.join(self.output_dir, self.exp_type, self.data_type, lang, "generation_attention", f"{dim1}_{dim2}")
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
        # languages = ["en", "fr", "ja", "ko"]
        languages = ["ko"]
    total_num_of_dimensions = 0
    total_num_of_words = 0
    total_num_of_words_per_language = {lang: 0 for lang in languages}
    start_index = 0

    for lang in languages:
        print(f"\nProcessing language: {lang}")
        lang_data = visualizer.data[lang]
        print(f"Found {len(lang_data)} samples for language {lang}")
        
        if max_samples:
            lang_data = lang_data[start_index:max_samples]
            print(f"Limiting to {len(lang_data)} samples")

        
        for sample_idx, sample in enumerate(tqdm(lang_data, desc=f"Processing {lang}")):
            for dimension_name in sample.get("dimensions", {}):
                constructed_prompt, dim1, dim2, answer, word, dim_name = visualizer.prmpt_dims_answrs(
                    visualizer.prompts, sample, dimension_name
                )
                
                visualizer.inference_with_hooks(
                    word, lang, constructed_prompt, dim1, dim2, answer, sample, dimension_name
                )

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
