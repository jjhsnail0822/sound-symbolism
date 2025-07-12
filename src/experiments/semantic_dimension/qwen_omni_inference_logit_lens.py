import argparse
import json
import os
import re

import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# interpretability
qwen_token_1 = 16
qwen_token_2 = 17
global_hidden_states = {}
local_hidden_states = {}
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
            exp_name: str,
            max_tokens: int,
            temperature: float = 0.0,
            retry_failed_answers: bool = False,
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.exp_name = exp_name
        self.retry_failed_answers = retry_failed_answers

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

    def run_mcq_experiment(self):
        # Load MCQ data
        print(f"Loading MCQ data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            mcq_data = json.load(f)
        print(f"Loaded {len(mcq_data)} questions.")
        mcq_data = mcq_data[:10]

        # Prepare file for saving results
        model_name = os.path.basename(self.model_path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        results_filename = f"{self.output_dir}/{self.data_path.split('/')[-1].replace('.json', '')}_{model_name}.json"
        print(f"Results will be saved to: {results_filename}")

        # Load existing results if file exists
        existing_results = {}
        if os.path.exists(results_filename):
            print("Found existing results file. Resuming experiment.")
            with open(results_filename, 'r', encoding='utf-8') as f:
                try:
                    saved_data = json.load(f)
                    # Create a dictionary for quick lookup
                    for res in saved_data.get('results', []):
                        # Assuming meta_data is unique for each question
                        key = json.dumps(res['meta_data'], sort_keys=True)
                        existing_results[key] = res
                    print(f"Loaded {len(existing_results)} existing results.")
                except json.JSONDecodeError:
                    print("Warning: Could not decode JSON from results file. Starting from scratch.")

        all_results = []

        # Run experiment
        print(f"Running MCQ experiment on {len(mcq_data)} questions...")

        # Process each question
        for query_idx, query in tqdm(enumerate(mcq_data)):
            query_key = json.dumps(query['meta_data'], sort_keys=True)

            # --- Logic to skip or retry ---

            word = query['meta_data']['word']
            language = query['meta_data']['language']
            if 'audio' in self.exp_name.lower():  # audio experiment
                if '<AUDIO>' in query['question']:  # word -> meaning
                    question_first_part = query['question'].split("<AUDIO>")[0]
                    question_second_part = query['question'].split("<AUDIO>")[1]
                    if language == 'art':
                        audio_path = f'data/processed/art/tts/{word}.wav'
                    else:
                        audio_path = f'data/processed/nat/tts/{language}/{word}.wav'
                    conversation = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text",
                                 "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
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
                # else: # meaning -> word
                #     question_parts = re.split(r'<AUDIO: .*?>', query['question'])
                #     option_audio_paths = []
                #     for option in query['options_info']:
                #         option_audio_paths.append(f'data/processed/nat/tts/{option["language"]}/{option["text"]}.wav')
                #         # check if the audio file exists
                #         if not os.path.exists(option_audio_paths[-1]):
                #             raise FileNotFoundError(f"Audio file not found: {option_audio_paths[-1]}")

                #     # Build content list dynamically
                #     content = [{"type": "text", "text": question_parts[0]}]
                #     for i in range(len(option_audio_paths)):
                #         content.append({"type": "audio", "audio": option_audio_paths[i]})
                #         if i + 1 < len(question_parts):
                #             content.append({"type": "text", "text": question_parts[i + 1]})

                #     conversation = [
                #         {
                #             "role": "system",
                #             "content": [
                #                 {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                #             ],
                #         },
                #         {
                #             "role": "user",
                #             "content": content,
                #         },
                #     ]
            else:  # text experiment
                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text",
                             "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query['question']},
                        ],
                    },
                ]

            # Prepare inputs
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

            # Generate response
            with torch.no_grad():
                text_ids = self.model.generate(
                    **inputs,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    max_new_tokens=self.max_tokens,
                )
            # Decode response
            full_text = \
            self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # interpretability
            input_length = inputs["input_ids"].shape[-1]
            # print(f'input : {inputs["input_ids"]}')
            # print(f'output : {text_ids}')
            pure_out = text_ids[0, input_length:]

            # save only for the ideal output
            if pure_out.shape[-1] == 3:
                logit_lens_for_all = self._logit_lens_for_all_layers(local_hidden_states)
                dimension = query["meta_data"]["dimension"]
                key = self._get_example_key(language, word, dimension)
                import pdb; pdb.set_trace()
                global_logit_lens[key] = logit_lens_for_all

                global_hidden_states[query_idx] = local_hidden_states.copy()

            local_hidden_states.clear()

            # print('Full text:', full_text)
            # Extract only the assistant's response
            if "assistant\n" in full_text:
                model_answer = full_text.split("assistant\n")[-1].strip()
            else:
                raise ValueError(f"Unexpected format in model output: {full_text}")

            # Extract first integer as answer
            answer_match = re.search(r'\d+', model_answer)
            if answer_match:
                extracted_answer = answer_match.group(0)
            else:
                extracted_answer = None

            # Handle cases where the model output is empty or None
            if extracted_answer is None:
                print(f"Warning: Model output is empty for query: {query['question'][:50]}...")
                extracted_answer = "0"

            # Check correctness
            try:
                is_correct = int(extracted_answer) == query['meta_data']['answer']
            except ValueError:
                print(f"Warning: Model output '{extracted_answer}' is not a valid integer. Marking as incorrect.")
                is_correct = False

            # print(f"Model answer: {extracted_answer}, Correct answer: {query['answer']}, Is correct: {is_correct}")

            # Store result
            result = {
                "meta_data": query['meta_data'],
                "model_answer": extracted_answer,
                "full_response": model_answer,
                "is_correct": is_correct
            }
            all_results.append(result)

            # --- Save results every 10 queries ---
            if len(all_results) % 10 == 0:
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

        # interpretability
        for hook in hooks:
            hook.remove()

        # --- Final save for any remaining results ---
        correct_count = sum(1 for r in all_results if r["is_correct"])
        total_count = len(all_results)
        accuracy = correct_count / total_count if total_count > 0 else 0

        print(f"Experiment completed. Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")

        # Save results
        results_dict = {
            "model": self.model_path,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "results": all_results,
        }

        # Save results to file
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)

        logit_lens_path = "./logit_lens.json"
        with open(logit_lens_path, 'w', encoding='utf-8') as f:
            json.dump(global_logit_lens, f, ensure_ascii=False, indent=4)

        print(f"Results saved to: {results_filename}")

        # Clean up
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return results_dict, results_filename

    def _logit_lens_for_all_layers(self, local_hidden_states):
        logit_lens_for_all_layers = {}
        for layer_id, hidden_state in local_hidden_states.items():
            print(f"Processing layer: {layer_id}")
            logit_lens = self._run_logit_lens(hidden_state)
            logit_lens_for_all_layers[layer_id] = logit_lens

        print("======" * 20)

        return logit_lens_for_all_layers

    def _run_logit_lens(self, hidden_state):
        hidden_state = torch.tensor(hidden_state, dtype=torch.bfloat16).to(self.model.device)

        normalized = self.thinker_norm(hidden_state)
        logits = self.thinker_lm_head(normalized)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        # top
        top_prob = torch.topk(prob, 1, dim=-1).values[0].tolist()
        top_token_idx = torch.topk(logits, 1, dim=-1).indices[0].tolist()
        top_word = self.processor.tokenizer.decode(top_token_idx)

        # choice
        choice_prob = prob[qwen_token_1: qwen_token_2 + 1].tolist()

        output = {
            "top": {
                "prob": top_prob,
                "token_idx": top_token_idx,
                "word": top_word,
            },
            "choice": {
                "prob": choice_prob,
            }
        }

        print(f"top prob      : {top_prob}")
        print(f"top token idx : {top_token_idx}")
        print(f"top word      : {top_word}")
        print("")
        print(f"choice prob   : {choice_prob}")

        print("=========================================")

        return output

    def _get_example_key(self, language, word, dimension):
        """
        Generate a unique key for the example based on language, word, and dimension.
        This will be used to store logit lens data.
        """
        return f"{language}_{word}_{dimension}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCQ experiment with Qwen Omni")
    parser.add_argument("--model", '-m', type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to the Qwen Omni model")
    parser.add_argument("--data", '-d', type=str, required=True, help="Path to the MCQ data JSON file")
    parser.add_argument("--output", '-o', type=str, default="results/experiments/semantic_dimension",
                        help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--retry-failed", action='store_true',
                        help="Retry questions where the model previously answered '0'")

    args = parser.parse_args()

    experiment = QwenOmniMCQExperiment(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        exp_name=args.exp_name,
        retry_failed_answers=args.retry_failed,
    )

    results, results_filename = experiment.run_mcq_experiment()
