import json
import random
import argparse
import os

random.seed(42)

WORD_GROUPS = ["common", "rare"]

def generate_datasets(word_group, task, experiment_name, input_dir, output_dir, prompts_file):
    WORD_GROUP = word_group
    TASK = task
    EXPERIMENT_NAME = experiment_name
    PROMPT_ROLE = "user_prompt" # Assuming the prompt structure uses this key
    AUDIO_TOKEN = "<AUDIO>"

    IS_OPTION_MEANING = "word_to_meaning" in EXPERIMENT_NAME
    MAX_OPTION = 4

    print(f"\n--- Generating dataset for: word_group='{WORD_GROUP}', task='{TASK}', experiment='{EXPERIMENT_NAME}' ---")

    clustered_file_path = os.path.join(input_dir, f'crosslingual_clustered_{WORD_GROUP}.json')
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

    word_to_cluster = {}
    word_details_map = {} # For easy lookup of word details like en_meaning, language
    for cluster_item in clustered_data:
        for word_info_obj in cluster_item.get('words', []):
            word_str = word_info_obj.get('word')
            lang_str = word_info_obj.get('language') # Get language
            if word_str and lang_str: # Ensure both word and language exist
                word_lang_key = (word_str, lang_str) # Create composite key
                word_to_cluster[word_lang_key] = cluster_item.get('cluster_id')
                word_details_map[word_lang_key] = word_info_obj # Store full details
            else:
                print(f"Warning: Skipping word_info_obj in word_details_map/word_to_cluster creation due to missing 'word' or 'language': {word_info_obj}")

    def generate_options_from_predefined(subject_word_info, local_is_option_meaning, experiment_name):
        subject_word_text = subject_word_info.get('word')
        subject_word_lang = subject_word_info.get('language')

        if local_is_option_meaning:
            correct_option_text = subject_word_info.get('en_meaning')
            correct_option_lang = 'en' # Meaning is in English
            if not correct_option_text:
                print(f"Warning: Missing 'en_meaning' for subject word '{subject_word_text}' (lang: {subject_word_lang}). Cannot generate options.")
                return None, None, None, None
        else: # Options are words
            if "ipa" in experiment_name:
                correct_option_text = subject_word_info.get('ipa')
                if not correct_option_text:
                    print(f"Warning: Subject word '{subject_word_text}' (lang: {subject_word_lang}) missing 'ipa' field for experiment '{experiment_name}'. Using original word.")
                    correct_option_text = subject_word_text
            elif "romanized" in experiment_name:
                correct_option_text = subject_word_info.get('romanization')
                if not correct_option_text:
                    print(f"Warning: Subject word '{subject_word_text}' (lang: {subject_word_lang}) missing 'romanization' field for experiment '{experiment_name}'. Using original word.")
                    correct_option_text = subject_word_text
            else: # original or audio (where options are words, not <AUDIO> tokens yet)
                correct_option_text = subject_word_text
            
            correct_option_lang = subject_word_lang

            if not correct_option_text or not correct_option_lang: # correct_option_text could be None if ipa/romanization was missing and original word was also missing (edge case)
                print(f"Warning: Missing 'word' (or its derived form for {experiment_name}) or 'language' for subject word '{subject_word_text}' (lang: {subject_word_lang}). Cannot generate options.")
                return None, None, None, None

        predefined_distractor_list = subject_word_info.get('distractors')
        if not predefined_distractor_list:
            print(f"Warning: 'distractors' field missing or empty for word '{subject_word_text}' (lang: {subject_word_lang}). Skipping.")
            return None, None, None, None

        distractor_choices_texts = []
        distractor_choices_info_list = [] 

        for dist_item in predefined_distractor_list:
            if len(distractor_choices_texts) >= MAX_OPTION - 1:
                break

            dist_option_text_val = None
            dist_option_lang_val = None
            original_dist_word_for_info = None # For options_info

            if local_is_option_meaning:
                dist_word_str = dist_item.get('word')
                dist_lang_str = dist_item.get('language')
                if not dist_word_str or not dist_lang_str:
                    print(f"Warning: Distractor item missing 'word' or 'language': {dist_item} for subject '{subject_word_text}' (lang: {subject_word_lang}). Skipping this distractor.")
                    continue

                dist_word_key = (dist_word_str, dist_lang_str)
                dist_word_details = word_details_map.get(dist_word_key)
                if not dist_word_details or not dist_word_details.get('en_meaning'):
                    print(f"Warning: Could not find 'en_meaning' for distractor word '{dist_word_str}' (lang: {dist_lang_str}) (target: '{subject_word_text}' lang: '{subject_word_lang}'). Skipping this distractor.")
                    continue
                dist_option_text_val = dist_word_details.get('en_meaning')
                dist_option_lang_val = 'en'
            else: # Options are words
                dist_word_str_for_map = dist_item.get('word')
                dist_lang_str_for_map = dist_item.get('language')

                if not dist_word_str_for_map or not dist_lang_str_for_map:
                    print(f"Warning: Distractor item missing 'word' or 'language' for word option: {dist_item} for subject '{subject_word_text}' (lang: {subject_word_lang}). Skipping this distractor.")
                    continue
                
                dist_word_key_for_map = (dist_word_str_for_map, dist_lang_str_for_map)
                dist_word_details_for_option = word_details_map.get(dist_word_key_for_map)

                if not dist_word_details_for_option:
                    print(f"Warning: Could not find details for distractor word '{dist_word_str_for_map}' (lang: {dist_lang_str_for_map}) in word_details_map. Skipping this distractor.")
                    continue
                
                original_dist_word_for_info = dist_word_details_for_option.get('word') # Get original word for info

                if "ipa" in experiment_name:
                    dist_option_text_val = dist_word_details_for_option.get('ipa')
                    if not dist_option_text_val:
                        print(f"Warning: Distractor word '{dist_word_str_for_map}' (lang: {dist_lang_str_for_map}) missing 'ipa' field for experiment '{experiment_name}'. Using original word.")
                        dist_option_text_val = original_dist_word_for_info # Use the already fetched original word
                elif "romanized" in experiment_name:
                    dist_option_text_val = dist_word_details_for_option.get('romanization')
                    if not dist_option_text_val:
                        print(f"Warning: Distractor word '{dist_word_str_for_map}' (lang: {dist_lang_str_for_map}) missing 'romanization' field for experiment '{experiment_name}'. Using original word.")
                        dist_option_text_val = original_dist_word_for_info # Use the already fetched original word
                else: # original or audio (where options are words)
                    dist_option_text_val = original_dist_word_for_info # Use the already fetched original word
                
                dist_option_lang_val = dist_word_details_for_option.get('language')

                if not dist_option_text_val or not dist_option_lang_val: # dist_option_text_val could be None if original word was missing
                    print(f"Warning: Distractor item missing 'word' (or its derived form for {experiment_name}) or 'language' for word option: {dist_item} for subject '{subject_word_text}' (lang: {subject_word_lang}). Skipping this distractor.")
                    continue
            
            if dist_option_text_val and dist_option_text_val != correct_option_text:
                distractor_choices_texts.append(dist_option_text_val)
                if not local_is_option_meaning: 
                    distractor_choices_info_list.append({
                        'text': dist_option_text_val, 
                        'language': dist_option_lang_val,
                        'original_word': original_dist_word_for_info # Add original word here
                    })
        
        if len(distractor_choices_texts) < MAX_OPTION - 1:
            print(f"Warning: Not enough unique distractors ({len(distractor_choices_texts)}) from predefined list for '{subject_word_text}' (lang: {subject_word_lang}, experiment: {experiment_name}) (need {MAX_OPTION - 1}). Skipping.")
            return None, None, None, None

        all_options_texts_list = [correct_option_text] + distractor_choices_texts
        
        all_options_full_info_list = None
        if not local_is_option_meaning: 
            correct_original_word = subject_word_info.get('word') # Original word for the correct option
            all_options_full_info_list = [{
                'text': correct_option_text, 
                'language': correct_option_lang,
                'original_word': correct_original_word # Add original word for correct option
            }] + distractor_choices_info_list

        correct_answer_original_idx = 0 
        shuffled_indices = list(range(len(all_options_texts_list))) # Use actual length
        random.shuffle(shuffled_indices)

        final_shuffled_texts_list = [all_options_texts_list[i] for i in shuffled_indices]
        
        final_shuffled_full_info_list = None
        if not local_is_option_meaning and all_options_full_info_list: # Check all_options_full_info_list is not None
            final_shuffled_full_info_list = [all_options_full_info_list[i] for i in shuffled_indices]

        final_answer_idx_1_based = shuffled_indices.index(correct_answer_original_idx) + 1

        option_string_output = ""
        for i, text_opt_val in enumerate(final_shuffled_texts_list):
            prefix = f"{i + 1}: "
            if local_is_option_meaning: 
                option_string_output += f"{prefix}{text_opt_val}\n"
            else: 
                if "audio" in experiment_name: 
                     option_string_output += f"{prefix}<AUDIO: {text_opt_val}>\n"
                else: 
                     option_string_output += f"{prefix}{text_opt_val}\n"

        option_string_output = option_string_output.strip()
        
        subject_meaning_for_prompt = subject_word_info.get('en_meaning') 
        
        return option_string_output, final_answer_idx_1_based, subject_meaning_for_prompt, final_shuffled_full_info_list

    result_data = []
    processed_count = 0
    skipped_count = 0

    for cluster_data_item in clustered_data:
        for subject_word_info in cluster_data_item.get('words', []):
            word_text = subject_word_info.get('word')
            subject_lang = subject_word_info.get('language')
            meaning = subject_word_info.get('en_meaning')

            if not word_text or not subject_lang or not meaning:
                skipped_count += 1
                # print(f"Debug: Skipping subject word due to missing info: word='{word_text}', lang='{subject_lang}', meaning='{meaning}'")
                continue
            
            subject_word_key = (word_text, subject_lang)
            current_cluster_id = word_to_cluster.get(subject_word_key) 

            # Use the new function that relies on predefined distractors
            option_string, answer, meaning_text_for_prompt, options_info_for_result = \
                generate_options_from_predefined(subject_word_info, IS_OPTION_MEANING, EXPERIMENT_NAME)

            if option_string is None or answer is None:
                skipped_count += 1
                # The warning is printed inside generate_options_from_predefined
                # print(f"Skipping word '{word_text}' (lang: {subject_lang}) as options could not be generated.")
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
                if IS_OPTION_MEANING: # e.g., word_to_meaning_audio
                    word_for_prompt = AUDIO_TOKEN
                else: # e.g., meaning_to_word_audio
                    word_for_prompt = f"<AUDIO: {word_text}>"
            else:
                # Fallback for experiment names not explicitly handled, though ideally all should be.
                print(f"Warning: Experiment name '{EXPERIMENT_NAME}' does not explicitly define word type. Defaulting to word text for 'word' field in prompt.")
                word_for_prompt = word_text

            try:
                 question = prompt_template.format(
                     word=word_for_prompt, 
                     meaning=meaning_text_for_prompt, # This is subject_word_info.get('en_meaning')
                     options=option_string,
                     MAX_OPTION=MAX_OPTION
                 )
            except KeyError as e:
                 print(f"Error: Prompt template for '{EXPERIMENT_NAME}' is missing a required key: {e}. Prompt: '{prompt_template}'. Skipping word '{word_text}'.")
                 skipped_count += 1
                 continue

            result_entry = {
                "question": question,
                "answer": answer,
                "meta_data": {
                    "language": subject_word_info.get('language'), 
                    "word": word_text,
                    "meaning": meaning, # This is the original meaning of the subject word for metadata
                    "cluster_id": current_cluster_id,
                }
            }
            
            if options_info_for_result is not None:
                result_entry["options_info"] = options_info_for_result
                
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
    parser.add_argument("--input_dir", "-i", type=str, default="data/processed/nat/clustering",
                        help="Directory containing clustered input files (e.g., crosslingual_clustered_common.json).")
    parser.add_argument("--output_dir", "-o", type=str, default="data/prompts/word_meaning_matching_without_auditory_impression",
                        help="Directory to save the generated question files.")
    parser.add_argument("--prompts_file", "-p", type=str, default="data/prompts/prompts.json",
                        help="Path to the JSON file containing prompts.")
    args = parser.parse_args()

    # Define the experiments to generate
    experiments = {
        "word_meaning_matching_without_auditory_impression": [
            "word_to_meaning_original",
            "meaning_to_word_original",
            "word_to_meaning_romanized",
            "meaning_to_word_romanized",
            "word_to_meaning_ipa",
            "meaning_to_word_ipa",
            "word_to_meaning_audio",
            "meaning_to_word_audio"
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
                    output_dir=args.output_dir,
                    prompts_file=args.prompts_file
                )

    print("\n--- All dataset generation finished ---")
