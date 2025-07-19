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

# interpretability
qwen_token_1 = 16
qwen_token_2 = 17

local_hidden_states = {}
local_mlp_states = {}
local_attn_states = {}

global_hidden_states = {}
global_mlp_states = {}
global_attn_states = {}

global_logit_lens = {}
hooks = []


def save_hidden_for_each_layer(layer_id, dict_store):
    def inner(module, input, output):
        # only save the first token (in the ideal case, the token is the very first one!)
        if layer_id not in dict_store:
            import pdb; pdb.set_trace()
            hidden = output[0][0, -1].detach().float().cpu().numpy().tolist()
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
            is_debug: bool = False,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.input_type = input_type
        self.word_group = word_group

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
        mcq_data = mcq_data[:1000]

        # output_path
        model_name = os.path.basename(self.model_path)
        os.makedirs(self.output_dir, exist_ok=True)
        results_file_path = f"{self.output_dir}/{self.data_path.split('/')[-1].replace('.json', '')}_{model_name}.json"
        print(f"Results will be saved to: {results_file_path}")

        # logit lens path
        logit_lens_dir = "./results/logit_lens"
        os.makedirs(logit_lens_dir, exist_ok=True)
        logit_lens_path = os.path.join(logit_lens_dir, f"{self.input_type}_{self.word_group}.json")

        # check existing results
        existing_results = self.collect_already_done(results_file_path)

        # Run experiment
        print(f"Running MCQ experiment on {len(mcq_data)} questions...")
        all_results = []
        for query_idx, query in enumerate(tqdm(mcq_data)):
            # validate
            local_hidden_states.clear()
            local_mlp_states.clear()
            local_attn_states.clear()
            
            query_key = json.dumps(query['meta_data'], sort_keys=True)
            if query_key in existing_results:
                existing_result = existing_results[query_key]
                if existing_result.get("model_answer") != "0":
                    all_results.append(existing_result)
                    # continue

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

            result = {
                "meta_data": query['meta_data'],
                "model_answer": extracted_answer,
                "full_response": model_answer,
                "is_correct": is_correct
            }
            all_results.append(result)

            if len(all_results) % 10 == 0:
                self.save_output(all_results, results_file_path)

            # interpretability
            input_length = inputs["input_ids"].shape[-1]
            pure_out = text_ids[0, input_length:]

            # save only for the ideal output
            if pure_out.shape[-1] == 3:
                example_key = self._get_example_key(query)
                global_hidden_states[example_key] = local_hidden_states.copy()
                global_mlp_states[example_key] = local_mlp_states.copy()
                global_attn_states[example_key] = local_attn_states.copy()


            # save out
        self.save_output(all_results, results_file_path)
        print(f"Results saved to: {results_file_path}")

        # interpretability
        for hook in hooks:
            hook.remove()

        self.save_logit_lens(logit_lens_path)
        
        results = {
            "hidden_states": global_hidden_states,
            "mlp_states": global_mlp_states,
            "attn_states": global_attn_states,
        }
        with open(logit_lens_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

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

    def save_logit_lens(self, logit_lens_path):
        with open(logit_lens_path, 'w', encoding='utf-8') as f:
            json.dump(global_logit_lens, f, ensure_ascii=False, indent=4)

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

    def _logit_lens_for_all_layers(self, local_hidden_states):
        logit_lens_for_all_layers = {}
        for layer_id, hidden_state in local_hidden_states.items():
            print(f"Processing layer: {layer_id}")
            logit_lens = self._logit_lens(hidden_state)
            logit_lens_for_all_layers[layer_id] = logit_lens

        print("======" * 20)

        return logit_lens_for_all_layers

    def _logit_lens(self, hidden_state):
        hidden_state = torch.tensor(hidden_state, dtype=torch.bfloat16).to(self.model.device)

        normalized = self.thinker_norm(hidden_state)
        logits = self.thinker_lm_head(normalized)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        # choice
        choice_logits = logits[qwen_token_1: qwen_token_2 + 1].tolist()
        choice_prob = prob[qwen_token_1: qwen_token_2 + 1].tolist()

        # top
        top_prob = torch.topk(prob, 1, dim=-1).values[0].tolist()
        top_token_idx = torch.topk(logits, 1, dim=-1).indices[0].tolist()
        top_word = self.processor.tokenizer.decode(top_token_idx)
        top_logit = logits[top_token_idx].tolist()


        output = {
            "choice": {
                "logit": choice_logits,
                "prob": choice_prob,
            },
            "top": {
                "logit": top_logit,
                "prob": top_prob,
                "token_idx": top_token_idx,
                "word": top_word,
            },
        }

        print(f"top prob      : {top_prob}")
        print(f"top token idx : {top_token_idx}")
        print(f"top word      : {top_word}")
        print("")
        print(f"choice prob   : {choice_prob}")

        print("=========================================")

        return output

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
    )

    experiment.run_mcq_experiment()
