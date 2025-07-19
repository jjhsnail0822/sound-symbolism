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
local_hidden_states = {}
global_hidden_states = {}

global_logit_lens = {}
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
            data_path: str,
            output_dir: str,
            max_tokens: int,
            input_type: str,
            word_group: str,
            dim: int,
            num_examples: int,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.input_type = input_type
        self.word_group = word_group
        self.dim = dim
        self.num_examples = num_examples

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
            hook_func = save_hidden_for_each_layer(layer_id)
            hook = module.register_forward_hook(hook_func)
            hooks.append(hook)

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

    @torch.no_grad()
    def run_mcq_experiment(self):
        # Load MCQ data
        print(f"Loading MCQ data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            mcq_data = json.load(f)
        print(f"Loaded {len(mcq_data)} questions.")
        # mcq_data = mcq_data[:100]

        # output_path
        model_name = os.path.basename(self.model_path)
        os.makedirs(self.output_dir, exist_ok=True)
        results_file_path = f"{self.output_dir}/{self.data_path.split('/')[-1].replace('.json', '')}_{model_name}.json"
        print(f"Results will be saved to: {results_file_path}")

        # logit lens path
        logit_lens_dir = "./results/logit_lens/"
        os.makedirs(logit_lens_dir, exist_ok=True)
        logit_lens_path = os.path.join(logit_lens_dir, f"prompts.json")

        # Run experiment
        print(f"Running MCQ experiment on {len(mcq_data)} questions...")
        
        
        
        
        prompts = [
  "Classify the review: 1. Helpful 2. Not Helpful — \"This review gave me all the info I needed.\" Output should be: 1. or 2.",
  "Is this message urgent? 1. Yes 2. No — \"Server down. Immediate action required.\" Output should be: 1. or 2.",
  "Label the emotion: 1. Happy 2. Sad — \"I just got promoted today!\" Output should be: 1. or 2.",
  "Classify the content: 1. News 2. Advertisement — \"Buy one, get one free — limited time only!\" Output should be: 1. or 2.",
  "Is the sentence polite? 1. Yes 2. No — \"Move your stuff, it's in the way.\" Output should be: 1. or 2.",
  "Label this feedback: 1. Constructive 2. Unhelpful — \"You suck at this. Do better.\" Output should be: 1. or 2.",
  "Is the tone positive? 1. Yes 2. No — \"I had a wonderful time at the concert.\" Output should be: 1. or 2.",
  "Classify this question: 1. Factual 2. Opinion-based — \"Is chocolate better than vanilla?\" Output should be: 1. or 2.",
  "Label the message: 1. Customer Inquiry 2. Complaint — \"I still haven’t received my order.\" Output should be: 1. or 2.",
  "Identify the intent: 1. Request 2. Statement — \"Can you resend the report by 5 PM?\" Output should be: 1. or 2."
]

        for prompt_idx, prompt in enumerate(tqdm(prompts)):
            local_hidden_states.clear()
            
            # input
            inputs = self.get_input_tensors(prompt)

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
            

            # interpretability
            gen_token_idx = inputs["input_ids"].shape[-1]
            logit_lens_for_all_layers = self._logit_lens_for_all_layers(local_hidden_states, gen_token_idx)
            # logit_lens_for_all_layers["model_answer"] = extracted_answer
            logit_lens_for_all_layers["prompt"] = prompt
            logit_lens_for_all_layers["full_response"] = model_answer
            import pdb; pdb.set_trace()
            
            print(f"prompt: {prompt}")
            print(f"full response: {full_text}")
            print(f"model answer: {model_answer}")
            print(f"extracted answer: {extracted_answer}")
            print("=" * 100)
            global_logit_lens[prompt_idx] = logit_lens_for_all_layers

        # interpretability
        for hook in hooks:
            hook.remove()

        self.save_logit_lens(logit_lens_path)

        # Clean up
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_input_tensors(self, prompt):
        conversation = self._get_conversation_2(prompt)

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
    
    def _get_is_correct(self, item, extracted_answer):
        answer = item["answer"]
        dim1, dim2 = item["dimension"].split("-")
        
        if self.dim == 1:  # binary
            expected = {"1": dim1, "2": dim2}
        elif self.dim == 2:  # ternary
            expected = {"2": dim1, "1": dim2}
        else:
            return False  # or raise an error if needed

        return expected.get(extracted_answer) == answer
        
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

    def _get_conversation(self, item):
        conversation = []

        word = item["word"]
        ipa = item["ipa"]
        language = item["language"]
        # word_group = item["word_group"]
        # answer = item["answer"]
        dim1, dim2 = item["dimension"].split("-")

        # system content
        system_content = {
            "role": "system",
            "content": [
                    {"type": "text",
                     "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        }
        conversation.append(system_content)

        # first user content
        user_content = {
            "role": "user",
            "content": [],
        }

        user_content_first_text = f"Given a [WORD] with its pronunciation audio, does the given semantic feature align well with the word based on auditory impression?\n\n[WORD]"
        if "original" in self.input_type:
            user_content_first_text += f"\n{word}"
        elif "ipa" in self.input_type:
            user_content_first_text += f"\n{ipa}"
        if "audio" in self.input_type:
            user_content_first_text += f"\n<AUDIO>"
        user_content["content"].append({"type": "text", "text": user_content_first_text})

        # audio path
        audio_path = ""
        if "audio" in self.input_type:
            if language == 'art':
                audio_path = f'data/processed/art/tts/{word}.wav'
            else:
                audio_path = f'data/processed/nat/tts/{language}/{word}.wav'
            user_content["content"].append({"type": "audio", "audio": audio_path})

        # second user content
        dim = dim1 if self.dim == 1 else dim2
        user_content_second_text = f"\n\n[SEMANTIC DIMENSION]\n{dim} \n\n[OPTIONS]\n1: close \n2: far \nAnswer with the number only. (1-2)"
        user_content["content"].append({"type": "text", "text": user_content_second_text})

        conversation.append(user_content)

        return conversation

    def _get_conversation_2(self, prompt):
        conversation = []

        # system content
        system_content = {
            "role": "system",
            "content": [
                    {"type": "text",
                     "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        }
        conversation.append(system_content)

        # first user content
        user_content = {
            "role": "user",
            "content": [],
        }

        user_content["content"].append({
            "type": "text",
            "text": prompt
        })

        conversation.append(user_content)

        return conversation

    def _logit_lens_for_all_layers(self, local_hidden_states, gen_token_idx):
        logit_lens_for_all_layers = {}
        for layer_id, hidden_state in local_hidden_states.items():
            print(f"Processing layer: {layer_id}")
            logit_lens = self._logit_lens(hidden_state, gen_token_idx)
            logit_lens_for_all_layers[layer_id] = logit_lens

        print("======" * 20)

        return logit_lens_for_all_layers

    def _logit_lens(self, hidden_state, gen_token_idx):
        hidden_state = torch.tensor(hidden_state, dtype=torch.bfloat16).to(self.model.device)

        normalized = self.thinker_norm(hidden_state)
        logits = self.thinker_lm_head(normalized)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        # choice
        choice_logits = logits[gen_token_idx:gen_token_idx+1].tolist()
        choice_prob = prob[gen_token_idx:gen_token_idx+1].tolist()

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
        print(f"gen token idx : {gen_token_idx}")

        print("=========================================")

        return output

    def _get_example_key(self, query):
        word = query['word']
        language = query['language']
        dimension = query["dimension"]

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
    parser.add_argument("--dim",'-d', default=1, help="Dimension to run the experiment on")
    parser.add_argument("--num_examples", "-n", type=int, default=100, help="Number of examples to run")

    args = parser.parse_args()

    word_group = args.word_group
    input_data_path = f"{word_group}.json"
    output_dir = f"results/experiments/semantic_dimension/perturb/{args.input_type}"

    experiment = QwenOmniMCQExperiment(
        model_path=args.model,
        data_path=input_data_path,
        output_dir=output_dir,
        max_tokens=args.max_tokens,
        input_type=args.input_type,
        word_group=word_group,
        dim=args.dim,
        num_examples=args.num_examples,
    )

    experiment.run_mcq_experiment()
