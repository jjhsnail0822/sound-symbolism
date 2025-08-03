import json
import random
import argparse
import os

random.seed(42)

WORD_GROUPS = ["common", "rare", "constructed"]
LANGUAGES = ["en", "fr", "ja", "ko"]

def generate_datasets(word_group, task, experiment_name, input_dir, input_dir_constructed, output_dir, prompts_file):
    WORD_GROUP = word_group
    TASK = task
    EXPERIMENT_NAME = experiment_name
    PROMPT_ROLE = "user_prompt" # Assuming the prompt structure uses this key
    AUDIO_TOKEN = "<AUDIO>"

    print(f"\n--- Generating dataset for: word_group='{WORD_GROUP}', task='{TASK}', experiment='{EXPERIMENT_NAME}' ---")

    print(f"Loading prompts from: {prompts_file}")
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            if TASK not in prompts or EXPERIMENT_NAME not in prompts[TASK] or PROMPT_ROLE not in prompts[TASK][EXPERIMENT_NAME]:
                print(f"Error: Prompt not found for task='{TASK}', experiment='{EXPERIMENT_NAME}', role='{PROMPT_ROLE}' in {prompts_file}. Skipping.")
                return
    except FileNotFoundError:
        print(f"Error: Prompts file not found at {prompts_file}. Skipping.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {prompts_file}. Skipping.")
        return
    except KeyError:
         print(f"Error: Invalid prompt structure in {prompts_file} for task='{TASK}', experiment='{EXPERIMENT_NAME}', role='{PROMPT_ROLE}'. Skipping.")
         return
    
    # Load the semantic dimension gt
    if word_group == "constructed":
        gt_file_path = os.path.join(input_dir_constructed, f"semantic_dimension_binary_gt.json")
    else: # common or rare
        if 'binary' in EXPERIMENT_NAME:
            gt_file_path = os.path.join(input_dir, f"semantic_dimension_binary_gt.json")
        else:
            gt_file_path = os.path.join(input_dir, f"semantic_dimension_gt.json")
    try:
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            semantic_dimension_gt = json.load(f)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_file_path}. Skipping.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {gt_file_path}. Skipping.")
        return

    result_data = []
    processed_count = 0
    skipped_count = 0

    # We use words in current word group from the semantic dimension gt
    for lang in semantic_dimension_gt:
        for subject_word_info in semantic_dimension_gt[lang]:
            for dimension in subject_word_info.get('dimensions', []):
                if WORD_GROUP != 'constructed' and subject_word_info.get('word_group') != WORD_GROUP:
                    # Skip words that do not belong to the current word group
                    continue
                word_text = subject_word_info.get('word')
                subject_lang = subject_word_info.get('language')
                dimension_list = dimension.split('-')
                dimension1 = dimension_list[0]
                dimension2 = dimension_list[1]
                answer = subject_word_info['dimensions'][dimension]['answer']

                if not word_text or not subject_lang:
                    skipped_count += 1
                    # print(f"Debug: Skipping subject word due to missing info: word='{word_text}', lang='{subject_lang}', meaning='{meaning}'")
                    continue
                try:
                    prompt_template = prompts[TASK][EXPERIMENT_NAME][PROMPT_ROLE]
                except KeyError:
                    print(f"Critical Error: Prompt template lookup failed unexpectedly for task='{TASK}', experiment='{EXPERIMENT_NAME}'. Stopping generation for this group.")
                    return # Stop for this specific group

                word_for_prompt = ""
                if "original" in EXPERIMENT_NAME:
                    word_for_prompt = word_text
                elif "romanized" in EXPERIMENT_NAME:
                    word_for_prompt = subject_word_info.get('romanization')
                    if 'romanization' not in subject_word_info:
                        print(f"Warning: 'romanization' field not found for word '{word_text}' in EXPERIMENT_NAME '{EXPERIMENT_NAME}'. Using the word itself as a fallback.")
                elif "ipa" in EXPERIMENT_NAME:
                    # Assuming 'ipa' field exists in subject_word_info.
                    # If not, it will default to word_text.
                    # Make sure your subject_word_info contains an 'ipa' field for IPA experiments.
                    word_for_prompt = subject_word_info.get('ipa')
                    if 'ipa' not in subject_word_info:
                        print(f"Warning: 'ipa' field not found for word '{word_text}' in EXPERIMENT_NAME '{EXPERIMENT_NAME}'. Using the word itself as a fallback.")
                elif "audio" in EXPERIMENT_NAME:
                    word_for_prompt = ""
                else:
                    # Fallback for experiment names not explicitly handled, though ideally all should be.
                    print(f"Warning: Experiment name '{EXPERIMENT_NAME}' does not explicitly define word type. Defaulting to word text for 'word' field in prompt.")
                    word_for_prompt = word_text

                try:
                    question = prompt_template.format(
                        word=word_for_prompt,
                        audio=AUDIO_TOKEN,
                        dimension1=dimension1,
                        dimension2=dimension2,
                    )
                except KeyError as e:
                    print(f"Error: Prompt template for '{EXPERIMENT_NAME}' is missing a required key: {e}. Prompt: '{prompt_template}'. Skipping word '{word_text}'.")
                    skipped_count += 1
                    continue

                if answer == dimension1:
                    number_answer = 1
                elif answer == dimension2:
                    number_answer = 2
                elif answer == 'neither':
                    number_answer = 3
                else:
                    raise ValueError('Error: An illegal answer was detected.')
                result_entry = {
                    "question": question,
                    "answer": answer,
                    "meta_data": {
                        "language": subject_word_info.get('language'), 
                        "word": word_text,
                        "dimension": dimension,
                        "answer": number_answer, # the correct answer for the question
                    }
                }

                result_data.append(result_entry)
                processed_count += 1

    output_filename = f"{EXPERIMENT_NAME}-{word_group}.json"
    output_file_path = os.path.join(output_dir, output_filename)

    print(f"Saving {len(result_data)} generated questions to: {output_file_path}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving results to {output_file_path}: {e}")
        return

    print(f"Successfully generated {processed_count} MCQ questions for {EXPERIMENT_NAME} in {word_group}. Skipped {skipped_count} entries.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Word <-> Meaning MCQ questions using predefined distractors.")
    parser.add_argument("--input_dir", "-i", type=str, default="data/processed/nat/semantic_dimension",
                        help="Directory containing the semantic dimension ground truth file.")
    parser.add_argument("--input_dir_constructed", "-ic", type=str, default="data/processed/art/semantic_dimension",
                        help="Directory containing the semantic dimension ground truth file for constructed words.")
    parser.add_argument("--output_dir", "-o", type=str, default="data/prompts/semantic_dimension",
                        help="Directory to save the generated question files.")
    parser.add_argument("--prompts_file", "-p", type=str, default="data/prompts/prompts.json",
                        help="Path to the JSON file containing prompts.")
    args = parser.parse_args()

    # Define the experiments to generate
    experiments = {
        "semantic_dimension": [
            "semantic_dimension_binary_original",
            "semantic_dimension_binary_ipa",
            "semantic_dimension_binary_audio",
        ]
    }

    for task, exp_list in experiments.items():
        for word_group in WORD_GROUPS:
            for exp_name in exp_list:
                generate_datasets(
                    word_group=word_group,
                    task=task,
                    experiment_name=exp_name,
                    input_dir=args.input_dir,
                    input_dir_constructed=args.input_dir_constructed,
                    output_dir=args.output_dir,
                    prompts_file=args.prompts_file
                )

    print("\n--- All dataset generation finished ---")
