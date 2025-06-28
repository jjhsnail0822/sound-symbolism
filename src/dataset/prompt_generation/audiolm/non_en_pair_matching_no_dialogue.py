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
    AUDIO_TOKEN = "<AUDIO>"

    # Determine if options should be meanings or words based on experiment name
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

    # --- Generate word to cluster mapping ---
    word_to_cluster = {}
    for cluster in clustered_data:
        for word_info in cluster.get('words', []):
            word = word_info.get('word')
            if word:
                word_to_cluster[word] = cluster.get('cluster_id')

    def generate_options(subject_word_info, random_clusters, is_option_meaning=False):
        """Generates MCQ options (meanings or words) and identifies the correct answer."""
        # Get correct answer based on option type
        if is_option_meaning:
            right_option = subject_word_info.get('en_meaning')
            right_option_lang = 'en'
        else:
            right_option = subject_word_info.get('word')
            right_option_lang = subject_word_info.get('language', LANGUAGE)
        
        if not right_option:
            print(f"Warning: Missing {'meaning' if is_option_meaning else 'word'} for word '{subject_word_info.get('word')}'. Skipping option generation.")
            return None, None, None, None

        # Generate distractor options from random clusters
        random_options = []
        random_options_info = []  # Store language info for each option
        for cluster_id in random_clusters:
            cluster = next((c for c in clustered_data if c['cluster_id'] == cluster_id), None)
            if cluster:
                cluster_words = cluster.get('words', [])
                if cluster_words:
                    random_word_info = random.choice(cluster_words)
                    if is_option_meaning:
                        random_option = random_word_info.get('en_meaning')
                        random_option_lang = 'en'
                        # Don't store language info for meanings
                    else:
                        random_option = random_word_info.get('word')
                        random_option_lang = random_word_info.get('language', LANGUAGE)
                        random_options_info.append({
                            'text': random_option,
                            'language': random_option_lang
                        })
                    
                    if random_option and random_option != right_option:
                        random_options.append(random_option)

        # Ensure we have enough options
        if len(random_options) < MAX_OPTION - 1:
            print(f"Warning: Not enough distractor options for word '{subject_word_info.get('word')}'. Skipping.")
            return None, None, None, None

        # Take only the required number of distractors
        random_options = random_options[:MAX_OPTION - 1]
        if not is_option_meaning:
            random_options_info = random_options_info[:MAX_OPTION - 1]
        
        # Combine right option with distractors
        options = [right_option] + random_options
        
        if is_option_meaning:
            # Don't create options_info for meanings
            options_info = None
        else:
            options_info = [{
                'text': right_option,
                'language': right_option_lang
            }] + random_options_info
        
        answer_indices = list(range(MAX_OPTION))
        
        # Shuffle the options and answer index together
        random.shuffle(answer_indices)
        shuffled_options = [options[i] for i in answer_indices]
        
        if not is_option_meaning:
            shuffled_options_info = [options_info[i] for i in answer_indices]
        else:
            shuffled_options_info = None
            
        answer = answer_indices.index(0) + 1
        
        # Create the option string
        option_string = ""
        for i, option in enumerate(shuffled_options):
            if is_option_meaning:
                option_string += f"{i + 1}: {option}\n"
            else:
                option_string += f"{i + 1}: <AUDIO: {option}>\n"
        option_string = option_string.strip()
        
        return option_string, answer, subject_word_info.get('en_meaning'), shuffled_options_info

    # --- Generate MCQ questions ---
    result_data = []
    processed_count = 0
    skipped_count = 0

    # Iterate through clusters and words from the loaded data
    for cluster in clustered_data:
        cluster_id = cluster.get('cluster_id')
        if cluster_id is None: 
            continue

        for subject_word_info in cluster.get('words', []):
            # Basic validation for the word entry
            word_text = subject_word_info.get('word')
            meaning = subject_word_info.get('en_meaning')

            # Skip if essential information is missing
            if not word_text or not meaning:
                skipped_count += 1
                continue

            # Choose clusters randomly, not including the current cluster
            available_clusters = [c['cluster_id'] for c in clustered_data if c['cluster_id'] != cluster_id]
            if len(available_clusters) < MAX_OPTION - 1:
                print(f"Warning: Not enough clusters available for distractors. Skipping word '{word_text}'.")
                skipped_count += 1
                continue
            
            random_clusters = random.sample(available_clusters, MAX_OPTION - 1)

            # Generate options based on the experiment type
            option_string, answer, meaning_text, options_info = generate_options(subject_word_info, random_clusters, IS_OPTION_MEANING)

            if option_string is None or answer is None:
                skipped_count += 1
                continue # Skip if options couldn't be generated

            # Select the correct prompt template
            try:
                prompt_template = prompts[TASK][EXPERIMENT_NAME][PROMPT_ROLE]
            except KeyError:
                 print(f"Critical Error: Prompt template lookup failed unexpectedly for task='{TASK}', experiment='{EXPERIMENT_NAME}'. Stopping generation.")
                 return

            # Format the question using the prompt template with AUDIO_TOKEN
            try:
                 question = prompt_template.format(
                     word=AUDIO_TOKEN,  # Use AUDIO_TOKEN instead of IPA
                     meaning=meaning_text,
                     options=option_string,
                     MAX_OPTION=MAX_OPTION
                 )
            except KeyError as e:
                 print(f"Error: Prompt template for '{EXPERIMENT_NAME}' is missing a required key: {e}. Prompt: '{prompt_template}'. Skipping word '{word_text}'.")
                 skipped_count += 1
                 continue

            # Create the result data entry
            result_entry = {
                "question": question,
                "answer": answer,
                "meta_data": {
                    "language": subject_word_info.get('language', LANGUAGE), # Use language from word data if available (cross-lang case)
                    "word": word_text,
                    "meaning": meaning,
                    "cluster_id": cluster_id,
                }
            }
            
            # Only add options_info if it's not None (i.e., for word options only)
            if options_info is not None:
                result_entry["options_info"] = options_info
                
            result_data.append(result_entry)
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
    parser = argparse.ArgumentParser(description="Generate Word <-> Meaning MCQ questions without dialogues using original words.")
    parser.add_argument("--input_dir", "-i", type=str, default="data/processed/nat/clustering",
                        help="Directory containing clustered input files (e.g., crosslingual_clustered_fr.json).")
    parser.add_argument("--output_dir", "-o", type=str, default="data/prompts/understanding/non_en_pair_matching/audiolm",
                        help="Directory to save the generated question files.")
    parser.add_argument("--prompts_file", "-p", type=str, default="data/prompts/prompts.json",
                        help="Path to the JSON file containing prompts.")
    args = parser.parse_args()

    # Define the experiments to generate
    experiments = {
        "understanding": [
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
