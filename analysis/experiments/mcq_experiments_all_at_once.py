from mcq_experiment import MCQExperiment
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
    default="word_meaning_pair_matching",
    help="Experiment name to run",
    choices=["word_meaning_pair_matching", 
             "ipa_word_meaning_pair_matching",
             "ipa_word_meaning_pair_matching_no_dialogue",
             "non_en_word_meaning_pair_matching",
             "non_en_word_meaning_pair_matching_no_dialogue"],
)
args = parser.parse_args()

OUTPUT_DIR = f"analysis/experiments/understanding/{args.experiment}"

langs = ['en', 'fr', 'ja', 'ko']

if args.experiment == "word_meaning_pair_matching":
    data_paths = [
        'dataset/3_questions/nat/understanding-unmasked_word_to_meaning_mcq-{language}.json',
        'dataset/3_questions/nat/understanding-masked_meaning_to_word_mcq-{language}.json'
    ]
elif args.experiment == "ipa_word_meaning_pair_matching":
    data_paths = [
    'dataset/3_questions/nat/understanding_ipa_pair_matching/ipa_unmasked_word_to_meaning_mcq-{language}.json',
    'dataset/3_questions/nat/understanding_ipa_pair_matching/ipa_masked_meaning_to_word_mcq-{language}.json'
    ]
elif args.experiment == "ipa_word_meaning_pair_matching_no_dialogue":
    langs = ['en', 'fr', 'ja', 'ko']
    data_paths = [
        'dataset/3_questions/nat/understanding_ipa_pair_matching_no_dialogue/ipa_unmasked_word_to_meaning_mcq_no_dialogue-{language}.json',
        'dataset/3_questions/nat/understanding_ipa_pair_matching_no_dialogue/ipa_masked_meaning_to_word_mcq_no_dialogue-{language}.json'
    ]
elif args.experiment == "non_en_word_meaning_pair_matching":
    langs = ['fr', 'ja', 'ko', 'cross_language']
    data_paths = [
        'dataset/3_questions/nat/understanding_non_en_pair_matching/non_en_unmasked_word_to_meaning_mcq-{language}.json',
        'dataset/3_questions/nat/understanding_non_en_pair_matching/non_en_masked_meaning_to_word_mcq-{language}.json'
    ]
elif args.experiment == "non_en_word_meaning_pair_matching_no_dialogue":
    langs = ['fr', 'ja', 'ko', 'cross_language']
    data_paths = [
        'dataset/3_questions/nat/understanding_non_en_pair_matching_no_dialogue/non_en_unmasked_word_to_meaning_mcq_no_dialogue-{language}.json',
        'dataset/3_questions/nat/understanding_non_en_pair_matching_no_dialogue/non_en_masked_meaning_to_word_mcq_no_dialogue-{language}.json'
    ]

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