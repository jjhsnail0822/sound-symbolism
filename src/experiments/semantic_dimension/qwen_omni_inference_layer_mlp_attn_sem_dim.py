import argparse
import json
import os
import random
import re

import numpy as np
import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

import numpy as np
import pickle


# interpretability
qwen_token_1 = 16
qwen_token_2 = 17

local_hidden_states = {}
local_mlp_states = {}
local_attn_states = {}

global_hidden_states = {}
global_mlp_states = {}
global_attn_states = {}

hooks = []


def save_hidden_for_each_layer(layer_id, dict_store):
    def inner(module, input, output):
        # only save the first token (in the ideal case, the token is the very first one!)
        if layer_id not in dict_store:
            if isinstance(output, tuple): # block, attention
                hidden = output[0][0, -1].detach().float().cpu().numpy()
            else:
                hidden = output[0, -1].detach().float().cpu().numpy()
            dict_store[layer_id] = hidden

    return inner


class QwenOmniMCQExperiment:
    def __init__(
            self,
            model_path: str,
            data_path: str,
            output_dir: str,
            max_tokens: int,
            input_type: str = "original",
            word_group: str = "common",
            sem_dim: str = None,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.input_type = input_type
        self.word_group = word_group
        self.sem_dim = sem_dim

        # Load Qwen Omni model
        print(f"Loading Qwen Omni model from {self.model_path}")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            # attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()
        self.thinker_lm_head = self.model.thinker.lm_head
        self.thinker_norm = self.model.thinker.model.norm
        for layer_id, module in enumerate(self.model.thinker.model.layers):
            hidden_hook = save_hidden_for_each_layer(layer_id, local_hidden_states)
            mlp_hook = save_hidden_for_each_layer(layer_id, local_mlp_states)
            attn_hook = save_hidden_for_each_layer(layer_id, local_attn_states)
            
            module.register_forward_hook(hidden_hook)
            module.mlp.register_forward_hook(mlp_hook)
            module.self_attn.register_forward_hook(attn_hook)

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

    @torch.no_grad()
    def run_mcq_experiment(self):
        # Load MCQ data
        print(f"Loading MCQ data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            mcq_data = json.load(f)
        print(f"Loaded {len(mcq_data)} questions.")
        
        # Filter by semantic dimension if specified
        if self.sem_dim:
            original_count = len(mcq_data)
            mcq_data = [q for q in mcq_data if q['meta_data']['dimension'].split("-")[0] == self.sem_dim]
            print(f"Filtered to {len(mcq_data)} questions with semantic dimension '{self.sem_dim}' (from {original_count}).")

        mcq_data = mcq_data[:100]
        print(f"Using first 100 questions as no specific semantic dimension was provided.")

        # mlp attn hidden states path
        mlp_attn_hidden_states_dir = "./results/mlp_attn_hidden_states_sem_dim"
        os.makedirs(mlp_attn_hidden_states_dir, exist_ok=True)
        
        # Add sem_dim to the filename if it's specified
        if self.sem_dim:
            pickle_path = os.path.join(mlp_attn_hidden_states_dir, f"{self.input_type}_{self.word_group}_{self.sem_dim}.pkl")
        else:
            pickle_path = os.path.join(mlp_attn_hidden_states_dir, f"{self.input_type}_{self.word_group}.pkl")
            
        print(f"Saving to {pickle_path}")

        # Run experiment
        print(f"Running MCQ experiment on {len(mcq_data)} questions...")
        for query_idx, query in enumerate(tqdm(mcq_data)):
            # validate
            local_hidden_states.clear()
            local_mlp_states.clear()
            local_attn_states.clear()
            
            # input
            inputs = self.get_input_tensors(query)

            # inference (generate)
            text_ids = self.model.generate(
                **inputs,
                use_audio_in_video=True,
                max_new_tokens=self.max_tokens,
            )

            # check output
            full_text = self.processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            if "assistant\n" in full_text:
                model_answer = full_text.split("assistant\n")[-1].strip()
            else:
                raise ValueError(f"Unexpected format in model output: {full_text}")
            answer_match = re.search(r'\d+', model_answer)
            if answer_match:
                extracted_answer = answer_match.group(0)
            else:
                extracted_answer = None
            if extracted_answer is None:
                print(f"Warning: Model output is empty for query: {query['question'][:50]}...")
                extracted_answer = "0"

            # get correctness
            try:
                is_correct = int(extracted_answer) == query['meta_data']['answer']
            except ValueError:
                print(f"Warning: Model output '{extracted_answer}' is not a valid integer. Marking as incorrect.")
                is_correct = False

            # interpretability
            input_length = inputs["input_ids"].shape[-1]
            pure_out = text_ids[0, input_length:]

            # save only for the ideal output
            if pure_out.shape[-1] == 3:
                example_key = self._get_example_key(query)
                global_hidden_states[example_key] = local_hidden_states.copy()
                global_mlp_states[example_key] = local_mlp_states.copy()
                global_attn_states[example_key] = local_attn_states.copy()

        # interpretability
        for hook in hooks:
            hook.remove()

        # save
        results = { 
            "hidden_states": global_hidden_states,
            "mlp_states": global_mlp_states,
            "attn_states": global_attn_states,
        }
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
            
        # Clean up
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_input_tensors(self, query):
        conversation = self._get_conversation(query)

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        return inputs

    def _get_conversation(self, query):
        word = query['meta_data']['word']
        language = query['meta_data']['language']

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text",
                     "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
        ]

        if 'audio' in self.input_type and '<AUDIO>' in query['question']:
            question_first_part = query['question'].split("<AUDIO>")[0]
            question_second_part = query['question'].split("<AUDIO>")[1]
            if language == 'art':
                audio_path = f'data/processed/art/tts/{word}.wav'
            else:
                audio_path = f'data/processed/nat/tts/{language}/{word}.wav'
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question_first_part},
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": question_second_part},
                ],
            })
        else:  # text experiment
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": query['question']},
                ],
            })

        return conversation

    def _get_example_key(self, query):
        word = query['meta_data']['word']
        language = query['meta_data']['language']
        dimension = query["meta_data"]["dimension"]

        return f"{language}_{word}_{dimension}"


if __name__ == "__main__":
    # reproducibility
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # args
    parser = argparse.ArgumentParser(description="Run MCQ experiment with Qwen Omni")
    parser.add_argument("--model", '-m', type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to the Qwen Omni model")
    parser.add_argument("--input_type", "-i", type=str, choices=["original", "ipa", "audio", "original_and_audio", "ipa_and_audio"])
    parser.add_argument("--word_group", "-w", type=str, choices=["common", "rare", "constructed"])
    parser.add_argument("--sem_dim", "-s", type=str, help="Specific semantic dimension to filter by")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")

    args = parser.parse_args()

    input_data_path = f"data/prompts/semantic_dimension/semantic_dimension_binary_{args.input_type}-{args.word_group}.json"
    output_dir = "results/experiments/semantic_dimension"

    experiment = QwenOmniMCQExperiment(
        model_path=args.model,
        data_path=input_data_path,
        output_dir=output_dir,
        max_tokens=args.max_tokens,
        input_type=args.input_type,
        word_group=args.word_group,
        sem_dim=args.sem_dim,
    )

    experiment.run_mcq_experiment()