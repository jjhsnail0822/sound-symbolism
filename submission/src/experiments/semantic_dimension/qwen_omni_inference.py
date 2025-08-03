import json
import argparse
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
import os
import re
import torch

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
            attn_implementation="flash_attention_2",
        )
        self.model.disable_talker()
        
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)

    def run_mcq_experiment(self):
        # Load MCQ data
        print(f"Loading MCQ data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            mcq_data = json.load(f)
        print(f"Loaded {len(mcq_data)} questions.")
        
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
        for query in tqdm(mcq_data):
            query_key = json.dumps(query['meta_data'], sort_keys=True)

            # --- Logic to skip or retry ---
            if query_key in existing_results:
                existing_result = existing_results[query_key]
                # If not retrying, or if retrying but this one was not a failure, skip
                if not self.retry_failed_answers or existing_result.get("model_answer") != "0":
                    all_results.append(existing_result)
                    continue

            word = query['meta_data']['word']
            language = query['meta_data']['language']
            if 'audio' in self.exp_name.lower(): # audio experiment
                if '<AUDIO>' in query['question']: # word -> meaning
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
            else: # text experiment
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
            full_text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
        
        print(f"Results saved to: {results_filename}")

        # Clean up
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return results_dict, results_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCQ experiment with Qwen Omni")
    parser.add_argument("--model", '-m', type=str, default="Qwen/Qwen2.5-Omni-7B", help="Path to the Qwen Omni model")
    parser.add_argument("--data", '-d', type=str, required=True, help="Path to the MCQ data JSON file")
    parser.add_argument("--output", '-o', type=str, default="results/experiments/semantic_dimension", help="Directory to save results")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--retry-failed", action='store_true', help="Retry questions where the model previously answered '0'")
    
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