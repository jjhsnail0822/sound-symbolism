import argparse
import json
import os
import random
import re
import itertools

import numpy as np
import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# interpretability
qwen_token_1 = 16
qwen_token_2 = 17

local_hidden_states = {}

distance_results = {}
hooks = []


def save_hidden_for_each_layer(layer_id):
    def inner(module, input, output):
        # only save the first token (in the ideal case, the token is the very first one!)
        if layer_id not in local_hidden_states:
            hidden = output[0][0, -1].detach().float().cpu().numpy()
            local_hidden_states[layer_id] = hidden

    return inner


class QwenOmniMCQExperiment:
    def __init__(
            self,
            model_path: str,
            output_dir: str,
            max_tokens: int,
            word_group: str = "common",
            is_debug: bool = False,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.word_group = word_group

        # Load Qwen Omni model
        print(f"Loading Qwen Omni model from {self.model_path}")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()
        self.thinker_lm_head = self.model.thinker.lm_head
        self.thinker_norm = self.model.thinker.model.norm
        for layer_id, module in enumerate(self.model.thinker.model.layers):
            hook_func = save_hidden_for_each_layer(layer_id)
            hook = module.register_forward_hook(hook_func)
            hooks.append(hook)

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

    @torch.no_grad()
    def run_mcq_experiment(self):
        # Load all MCQ data
        input_types = ["original", "ipa", "audio", "original_and_audio", "ipa_and_audio"]
        all_mcq_data = {}
        num_questions = -1

        for itype in input_types:
            data_path = f"data/prompts/semantic_dimension/semantic_dimension_binary_{itype}-{self.word_group}.json"
            if not os.path.exists(data_path):
                print(f"Warning: Data file not found, skipping: {data_path}")
                continue
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_mcq_data[itype] = data
                if num_questions == -1:
                    num_questions = len(data)
                elif num_questions != len(data):
                    raise ValueError("All prompt files must have the same number of questions.")
        
        if not all_mcq_data:
            print("No data files found. Exiting.")
            return
            
        # num_questions = min(num_questions, 100) # for debugging
        print(f"Loaded data for {list(all_mcq_data.keys())}. Running for {num_questions} questions.")

        # output_path
        model_name = os.path.basename(self.model_path)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # distance results path
        results_dir = "./results/layer_divergence"
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"distances_{self.word_group}_{model_name}.json")

        # Run experiment
        print(f"Running MCQ experiment on {num_questions} questions...")
        
        for query_idx in tqdm(range(num_questions)):
            example_key = None
            question_hidden_states = {}  # Store hidden states for the current question across all input types
            
            for input_type, mcq_data in all_mcq_data.items():
                query = mcq_data[query_idx]
                if example_key is None:
                    example_key = self._get_example_key(query)
                    distance_results[example_key] = {}

                local_hidden_states.clear()

                # input
                inputs = self.get_input_tensors(query, input_type)

                # inference (generate)
                text_ids = self.model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    max_new_tokens=self.max_tokens,
                )
                
                # get correctness
                full_text = self.processor.batch_decode(
                    text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                if "assistant\n" in full_text:
                    model_answer = full_text.split("assistant\n")[-1].strip()
                else:
                    model_answer = "" # Or handle as an error
                answer_match = re.search(r'\d+', model_answer)
                extracted_answer = answer_match.group(0) if answer_match else "0"
                try:
                    is_correct = int(extracted_answer) == query['meta_data']['answer']
                except (ValueError, TypeError):
                    is_correct = False

                # interpretability
                input_length = inputs["input_ids"].shape[-1]
                pure_out = text_ids[0, input_length:]

                # save only for the ideal output
                if pure_out.shape[-1] == 3:
                    distance_results[example_key][input_type] = {
                        "is_correct": is_correct,
                        "model_answer": extracted_answer
                    }
                    question_hidden_states[input_type] = local_hidden_states.copy()

            # Calculate and save distances after processing all input types for the current question
            if example_key and example_key in distance_results:
                self._calculate_and_save_distances(distance_results[example_key], question_hidden_states)

        # interpretability
        for hook in hooks:
            hook.remove()

        self.save_distance_results(results_path)
        print(f"Distance results saved to: {results_path}")

        # Clean up
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_input_tensors(self, query, input_type):
        conversation = self._get_conversation(query, input_type)

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

    def _calculate_and_save_distances(self, result_dict, hidden_states_dict):
        # This function calculates the cosine distance between hidden states of all input type pairs.
        if len(hidden_states_dict) < 2:
            return # Not enough data to compare

        num_layers = len(next(iter(hidden_states_dict.values())))
        distances = {"distances_between_input_types": {}}

        for layer_id in range(num_layers):
            layer_distances = {}
            # Iterate over all unique pairs of input types
            for (type1, hs1_all_layers), (type2, hs2_all_layers) in itertools.combinations(hidden_states_dict.items(), 2):
                pair_key = f"{type1}_vs_{type2}"
                
                hs1 = hs1_all_layers.get(layer_id)
                hs2 = hs2_all_layers.get(layer_id)

                if hs1 is None or hs2 is None:
                    continue

                # Calculate cosine similarity and convert to cosine distance
                cos_sim = np.dot(hs1, hs2) / (np.linalg.norm(hs1) * np.linalg.norm(hs2))
                cos_dist = 1 - cos_sim

                layer_distances[pair_key] = {
                    "cosine_distance": float(cos_dist),
                }

                # print(f"Layer {layer_id}, {pair_key}: Cosine Distance = {cos_dist:.4f}")
            distances["distances_between_input_types"][f"layer_{layer_id}"] = layer_distances
        
        # Add the calculated distances to the main results dictionary for the current example
        result_dict.update(distances)

    def collect_already_done(self, results_file_path):
        existing_results = {}
        if os.path.exists(results_file_path):
            print("Found existing results file. Resuming experiment.")
            with open(results_file_path, 'r', encoding='utf-8') as f:
                try:
                    saved_data = json.load(f)
                    for res in saved_data.get('results', []):
                        example_key = json.dumps(res['meta_data'], sort_keys=True)
                        existing_results[example_key] = res
                    print(f"Loaded {len(existing_results)} existing results.")
                except json.JSONDecodeError:
                    print("Warning: Could not decode JSON from results file. Starting from scratch.")
        return existing_results

    def save_distance_results(self, results_path):
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(distance_results, f, ensure_ascii=False, indent=4)

    def save_output(self, all_results, results_filename):
        correct_count = sum(1 for r in all_results if r["is_correct"])
        total_count = len(all_results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        results_dict = {
            "model": self.model_path,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "results": all_results,
        }
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)

    def _get_conversation(self, query, input_type):
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

        if 'audio' in input_type and '<AUDIO>' in query['question']:
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
    parser.add_argument("--word_group", "-w", type=str, choices=["natural", "constructed"], required=True)
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")

    args = parser.parse_args()

    output_dir = "results/experiments/layer_divergence"

    if args.word_group == "natural":
        # Run for 'common' and 'rare', then combine results
        print("Running experiment for 'natural' word group (common + rare)...")
        
        # Run for common
        print("\n--- Running for 'common' words ---")
        experiment_common = QwenOmniMCQExperiment(
            model_path=args.model,
            output_dir=output_dir,
            max_tokens=args.max_tokens,
            word_group="common",
        )
        experiment_common.run_mcq_experiment()
        common_results = distance_results.copy()
        distance_results.clear() # Clear global results for the next run

        # Run for rare
        print("\n--- Running for 'rare' words ---")
        experiment_rare = QwenOmniMCQExperiment(
            model_path=args.model,
            output_dir=output_dir,
            max_tokens=args.max_tokens,
            word_group="rare",
        )
        experiment_rare.run_mcq_experiment()
        rare_results = distance_results.copy()
        distance_results.clear()

        # Combine results
        print("\n--- Combining results for 'natural' ---")
        combined_results = {**common_results, **rare_results}
        
        # Save combined results
        results_dir = "./results/layer_divergence"
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"distances_natural_{args.model.split('/')[-1]}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=4)
        print(f"Combined distance results for 'natural' saved to: {results_path}")

    else: # For 'constructed'
        experiment = QwenOmniMCQExperiment(
            model_path=args.model,
            output_dir=output_dir,
            max_tokens=args.max_tokens,
            word_group=args.word_group,
        )

        experiment.run_mcq_experiment()
