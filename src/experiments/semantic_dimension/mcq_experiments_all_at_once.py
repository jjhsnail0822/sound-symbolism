from qwen_omni_inference import QwenOmniMCQExperiment
import argparse
import json

parser = argparse.ArgumentParser(description="Run MCQ experiment")
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="Qwen/Qwen3-8B",
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
parser.add_argument(
    "--experiment",
    "-e",
    type=str,
    default="semantic_dimension_original",
    help="Experiment name to run",
    choices=["semantic_dimension_original",
             "semantic_dimension_romanized",
             "semantic_dimension_ipa",
             "semantic_dimension_audio",
             "semantic_dimension_binary_original",
             "semantic_dimension_binary_romanized",
             "semantic_dimension_binary_ipa",
             "semantic_dimension_binary_audio",
    ]
)
args = parser.parse_args()

word_groups = ['common', 'rare',]

if args.experiment == "semantic_dimension_original":
    data_paths = [
        'data/prompts/semantic_dimension/semantic_dimension_original-{word_group}.json',
    ]
    OUTPUT_DIR = f"results/experiments/semantic_dimension/original"
elif args.experiment == "semantic_dimension_romanized":
    data_paths = [
        'data/prompts/semantic_dimension/semantic_dimension_romanized-{word_group}.json',
    ]
    OUTPUT_DIR = f"results/experiments/semantic_dimension/romanized"
elif args.experiment == "semantic_dimension_ipa":
    data_paths = [
        'data/prompts/semantic_dimension/semantic_dimension_ipa-{word_group}.json',
    ]
    OUTPUT_DIR = f"results/experiments/semantic_dimension/ipa"
elif args.experiment == "semantic_dimension_audio":
    data_paths = [
        'data/prompts/semantic_dimension/semantic_dimension_audio-{word_group}.json',
    ]
    OUTPUT_DIR = f"results/experiments/semantic_dimension/audio"
elif args.experiment == "semantic_dimension_binary_original":
    data_paths = [
        'data/prompts/semantic_dimension/semantic_dimension_binary_original-{word_group}.json',
    ]
    OUTPUT_DIR = f"results/experiments/semantic_dimension/binary/original"
elif args.experiment == "semantic_dimension_binary_romanized":
    data_paths = [
        'data/prompts/semantic_dimension/semantic_dimension_binary_romanized-{word_group}.json',
    ]
    OUTPUT_DIR = f"results/experiments/semantic_dimension/binary/romanized"
elif args.experiment == "semantic_dimension_binary_ipa":
    data_paths = [
        'data/prompts/semantic_dimension/semantic_dimension_binary_ipa-{word_group}.json',
    ]
    OUTPUT_DIR = f"results/experiments/semantic_dimension/binary/ipa"
elif args.experiment == "semantic_dimension_binary_audio":
    data_paths = [
        'data/prompts/semantic_dimension/semantic_dimension_binary_audio-{word_group}.json',
    ]
    OUTPUT_DIR = f"results/experiments/semantic_dimension/binary/audio"
else:
    raise ValueError(f"Unsupported experiment: {args.experiment}")

all_brief_results = []
for word_group in word_groups:
    for data_path in data_paths:
        formatted_path = data_path.format(word_group=word_group)
        print(f"Running experiment for {formatted_path} using model {args.model}")
        if 'omni' in args.model.lower():
            experiment = QwenOmniMCQExperiment(
                model_path=args.model,
                data_path=formatted_path,
                output_dir=OUTPUT_DIR,
                max_tokens=32,
                temperature=0.0,
                exp_name=args.experiment,
            )
        else:
            raise ValueError("Unsupported model type. Only Qwen Omni models are supported in this script.")
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