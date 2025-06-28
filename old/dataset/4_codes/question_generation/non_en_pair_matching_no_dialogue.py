import json
import random
import argparse
import os

random.seed(42)

# Define languages explicitly for clarity when handling cross_language case
ALL_LANGS = ["fr", "ja", "ko"]
LANGUAGES_TO_PROCESS = ALL_LANGS + ["cross_language"] # Add cross_language option

def generate_datasets(language, task, experiment_name, input_dir, output_dir, prompts_file):
    LANGUAGE = language
    TASK = task
    EXPERIMENT_NAME = experiment_name
    PROMPT_ROLE = "user_prompt" # Assuming the prompt structure uses this key

    # Determine if options should be meanings or words (IPA) based on experiment name
    IS_OPTION_MEANING = "word_to_meaning" in EXPERIMENT_NAME
    # Determine the main subject of the question (word or meaning)
    IS_SUBJECT_WORD = "meaning_to_word" in EXPERIMENT_NAME

    MAX_OPTION = 4

    print(f"\n--- Generating dataset for: lang='{LANGUAGE}', task='{TASK}', experiment='{EXPERIMENT_NAME}' ---")

    # --- Load Clustered Data ---
    # Determine the correct input file based on language and scope assumed by the filename structure
    if LANGUAGE == "cross_language":
        # Assuming the cross-language file was generated with 'cross_language' scope
        clustered_file_path = os.path.join(input_dir, f'crosslingual_clustered_cross_language.json')
    else:
        # Assuming individual language files were generated with 'same_language' scope
        clustered_file_path = os.path.join(input_dir, f'crosslingual_clustered_{LANGUAGE}.json')

    print(f"Loading clustered data from: {clustered_file_path}")
    try:
        with open(clustered_file_path, 'r', encoding='utf-8') as f:
            clustered_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Clustered data file not found at {clustered_file_path}. Skipping.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {clustered_file_path}. Skipping.")
        return

    # --- Create word to meaning map ---
    word_to_meaning_map = {}
    for cluster in clustered_data:
        for word_info in cluster.get('words', []):
            word = word_info.get('word')
            meaning = word_info.get('en_meaning')
            if word and meaning:
                word_to_meaning_map[word] = meaning

    # --- Load Prompts ---
    print(f"Loading prompts from: {prompts_file}")
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            # Validate prompt existence
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


    def generate_options(subject_word_info, is_option_meaning, word_meaning_map):
        """Generates MCQ options (meanings or IPAs) and identifies the correct answer."""
        correct_ipa = subject_word_info.get('ipa', '').replace(" ", "")
        correct_meaning = subject_word_info.get('en_meaning')
        distractors_info = subject_word_info.get('distractors', [])

        if not correct_ipa or not correct_meaning:
             print(f"Warning: Missing IPA or meaning for word '{subject_word_info.get('word')}'. Skipping option generation.")
             return None, None, None

        # Determine the correct option text based on whether options are meanings or IPAs
        right_option_text = correct_meaning if is_option_meaning else correct_ipa

        # Extract distractor texts (meanings or IPAs)
        distractor_options_texts = []
        for distractor in distractors_info:
            dist_text = None
            if is_option_meaning:
                # Try getting meaning directly from distractor, fallback to map
                dist_text = distractor.get('en_meaning')
                if not dist_text:
                    dist_word = distractor.get('word')
                    if dist_word:
                        dist_text = word_meaning_map.get(dist_word)
            else: # Options are IPAs
                dist_text = distractor.get('ipa', '').replace(" ", "")

            if dist_text and dist_text != right_option_text: # Ensure distractor text exists and is different
                 distractor_options_texts.append(dist_text)

        # Ensure enough unique distractors, pad if necessary (or handle error)
        num_needed_distractors = MAX_OPTION - 1
        if len(set(distractor_options_texts)) < num_needed_distractors:
            # This case needs a strategy: skip, pad with placeholders, or find more distractors elsewhere.
            # For now, we'll skip if not enough unique distractors are available.
            print(f"Warning: Not enough unique distractors for word '{subject_word_info.get('word')}' (found {len(set(distractor_options_texts))}, need {num_needed_distractors}). Skipping.")
            return None, None, None
            # Alternative: Pad with placeholders if allowed by the task design
            # while len(distractor_options_texts) < num_needed_distractors:
            #     distractor_options_texts.append("FILLER_OPTION")

        # Select the required number of unique distractors
        unique_distractors = list(set(distractor_options_texts))
        random.shuffle(unique_distractors)
        selected_distractors = unique_distractors[:num_needed_distractors]

        # Combine right option with distractors and shuffle
        options = [right_option_text] + selected_distractors
        answer_indices = list(range(MAX_OPTION))
        random.shuffle(answer_indices) # Shuffle indices 0 to MAX_OPTION-1

        shuffled_options = [options[i] for i in answer_indices]

        # Find the new index of the correct answer (which was originally at index 0)
        try:
            answer_idx_plus_one = answer_indices.index(0) + 1
        except ValueError:
             print(f"Error: Could not find original answer index after shuffling for word '{subject_word_info.get('word')}'. Skipping.")
             return None, None, None


        # Create the formatted option string
        option_string = ""
        for i, option in enumerate(shuffled_options):
            option_string += f"{i + 1}: {option}\n"
        option_string = option_string.strip()

        return option_string, answer_idx_plus_one, correct_meaning # Return correct_meaning for prompt formatting


    # --- Generate MCQ questions ---
    result_data = []
    processed_count = 0
    skipped_count = 0

    # Iterate through clusters and words from the loaded data
    for cluster in clustered_data:
        cluster_id = cluster.get('cluster_id')
        if cluster_id is None: continue

        for subject_word_info in cluster.get('words', []):
            # Basic validation for the word entry
            word_text = subject_word_info.get('word')
            ipa = subject_word_info.get('ipa')
            meaning = subject_word_info.get('en_meaning')
            distractors = subject_word_info.get('distractors')

            # Skip if essential information or distractors are missing
            if not word_text or not ipa or not meaning or distractors is None:
                # print(f"Debug: Skipping word due to missing info: {word_text}, IPA: {ipa}, Meaning: {meaning}, Distractors: {distractors is not None}")
                skipped_count += 1
                continue

            # Generate options based on the experiment type
            option_string, answer, _ = generate_options(subject_word_info, IS_OPTION_MEANING, word_to_meaning_map)

            if option_string is None or answer is None:
                skipped_count += 1
                continue # Skip if options couldn't be generated

            # Prepare prompt variables
            prompt_word_ipa = ipa.replace(" ", "")
            prompt_meaning = meaning

            # Select the correct prompt template
            try:
                prompt_template = prompts[TASK][EXPERIMENT_NAME][PROMPT_ROLE]
            except KeyError:
                 print(f"Critical Error: Prompt template lookup failed unexpectedly for task='{TASK}', experiment='{EXPERIMENT_NAME}'. Stopping generation.")
                 return # Should not happen if initial check passed, but good safety measure.


            # Format the question using the prompt template
            # Ensure the prompt template only uses available variables
            try:
                 question = prompt_template.format(
                     word=prompt_word_ipa, # Use 'word' for IPA in prompt
                     meaning=prompt_meaning,
                     options=option_string,
                     MAX_OPTION=MAX_OPTION
                     # Removed: dialogue, MASKING_WORD
                 )
            except KeyError as e:
                 print(f"Error: Prompt template for '{EXPERIMENT_NAME}' is missing a required key: {e}. Prompt: '{prompt_template}'. Skipping word '{word_text}'.")
                 skipped_count += 1
                 continue


            # Create the result data entry
            result_data.append({
                "question": question,
                "answer": answer,
                "meta_data": {
                    "language": subject_word_info.get('language', LANGUAGE), # Use language from word data if available (cross-lang case)
                    "word": word_text,
                    "meaning": meaning,
                    "ipa": prompt_word_ipa,
                    "cluster_id": cluster_id,
                    # Removed dialogue-specific metadata
                }
            })
            processed_count += 1

    # --- Save the result data ---
    output_filename = f"{EXPERIMENT_NAME}-{LANGUAGE}.json"
    output_file_path = os.path.join(output_dir, output_filename)

    print(f"Saving {len(result_data)} generated questions to: {output_file_path}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving results to {output_file_path}: {e}")
        return

    print(f"Successfully generated {processed_count} MCQ questions for {EXPERIMENT_NAME} in {LANGUAGE}. Skipped {skipped_count} entries.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Word <-> Meaning MCQ questions without dialogues.")
    parser.add_argument("--input_dir", "-i", type=str, default="dataset/1_preprocess/nat",
                        help="Directory containing clustered input files (e.g., crosslingual_clustered_fr.json).")
    parser.add_argument("--output_dir", "-o", type=str, default="dataset/3_questions/nat/understanding_non_en_pair_matching_no_dialogue",
                        help="Directory to save the generated question files.")
    parser.add_argument("--prompts_file", "-p", type=str, default="analysis/experiments/prompts.json",
                        help="Path to the JSON file containing prompts.")
    args = parser.parse_args()

    # Define the experiments to generate
    experiments = {
        "understanding": [ # Assuming 'matching' is the task name in prompts.json
            "non_en_unmasked_word_to_meaning_mcq_no_dialogue",
            "non_en_masked_meaning_to_word_mcq_no_dialogue"
        ]
    }

    for task, exp_list in experiments.items():
        for lang in LANGUAGES_TO_PROCESS:
            for exp_name in exp_list:
                generate_datasets(
                    language=lang,
                    task=task,
                    experiment_name=exp_name,
                    input_dir=args.input_dir,
                    output_dir=args.output_dir,
                    prompts_file=args.prompts_file
                )

    print("\n--- All dataset generation finished ---")
