from mcq_experiment import MCQExperiment
import argparse
import json

OUTPUT_DIR = "analysis/experiments/understanding/ipa_word_meaning_pair_matching"

langs = ['en', 'fr', 'ja', 'ko']
data_paths = [
    'dataset/3_questions/nat/understanding/ipa_unmasked_word_to_meaning_mcq-{language}.json',
    'dataset/3_questions/nat/understanding/ipa_masked_meaning_to_word_mcq-{language}.json'
]

parser = argparse.ArgumentParser(description="Run MCQ experiment")
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="Qwen/Qwen2.5-7B-Instruct",
    help="Model name to run the experiment on",
)
parser.add_argument(
    "--gpu",
    "-g",
    type=int,
    default=1,
    help="Number of GPUs to use for the experiment",
)
parser.add_argument(
    "--api",
    action="store_true",
    help="Use OpenAI API instead of local model",
)
parser.add_argument(
    "--thinking",
    action="store_true",
    help="Enable thinking mode for Qwen3",
)
args = parser.parse_args()

all_brief_results = []
for lang in langs:
    for data_path in data_paths:
        formatted_path = data_path.format(language=lang)
        print(f"Running experiment for {formatted_path} using model {args.model}")
        experiment = MCQExperiment(
            model_path=args.model,
            data_path=formatted_path,
            output_dir=OUTPUT_DIR,
            use_api=args.api,
            tensor_parallel_size=args.gpu,
            max_tokens=32,
            max_model_len=4096,
            temperature=0.0,
            thinking=args.thinking,
        )
        results_dict, results_filename = experiment.run_mcq_experiment()
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {results_filename}")
        all_brief_results.append({
            "model": args.model,
            "accuracy": results_dict["accuracy"],
            "correct_count": results_dict["correct_count"],
            "total_count": results_dict["total_count"],
            "data_path": formatted_path,
            "thinking": args.thinking,
        })
# Save all brief results to a single file
all_results_filename = f"{OUTPUT_DIR}/all_results_{args.model.replace('/', '_')}{'-thinking' if args.thinking else ''}.json"
with open(all_results_filename, 'w', encoding='utf-8') as f:
    json.dump(all_brief_results, f, ensure_ascii=False, indent=4)
print(f"All results saved to {all_results_filename}")