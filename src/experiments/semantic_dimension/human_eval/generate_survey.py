import json
import random
from collections import defaultdict
import os

def load_and_group_data(input_files_config):
    """Load all input files and group data by dimension and source file."""
    # dimension -> source_file -> list of questions
    data_by_dim_and_source = defaultdict(lambda: defaultdict(list))
    
    for file_info in input_files_config:
        filepath = file_info.get("path")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found. Skipping.")
            continue
        except json.JSONDecodeError:
            print(f"Error: File '{filepath}' is not a valid JSON. Skipping.")
            continue
        
        for item in data:
            dimension = item.get("meta_data", {}).get("dimension")
            if dimension:
                data_by_dim_and_source[dimension][filepath].append(item)
                
    return data_by_dim_and_source

def main():
    """Main function to read config and generate survey JSON for Label Studio."""
    # 1. Load config file
    try:
        # Set config.json path relative to script location
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(script_dir, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'.")
        return

    input_files_config = config.get("input_files", [])
    output_file = config.get("output_file", "output.json")
    # total_questions is not used anymore, but kept for compatibility
    # total_questions_target = config.get("total_questions") 
    sampling_config = config.get("sampling_config", {})

    if not all(key in config for key in ["input_files", "output_file", "sampling_config"]):
        print("Error: Required keys missing in config.json (input_files, output_file, sampling_config).")
        return

    # 2. Load and group all data by dimension and file
    data_by_dim_and_source = load_and_group_data(input_files_config)
    
    all_final_questions = []
    
    # 3. Sampling for each dimension
    for dimension, num_words_to_sample in sampling_config.items():
        print(f"--- Processing dimension: {dimension} (Target words: {num_words_to_sample}) ---")
        
        if dimension not in data_by_dim_and_source:
            print(f"Warning: Dimension '{dimension}' not found in any input file. Skipping.")
            continue

        questions_for_dim = []
        
        # 4. For each file, sample words according to ratio
        for file_info in input_files_config:
            filepath = file_info.get("path")
            ratio = file_info.get("ratio")
            
            if filepath not in data_by_dim_and_source[dimension]:
                print(f"Info: No data for dimension '{dimension}' in file '{filepath}'.")
                continue

            questions_from_file = data_by_dim_and_source[dimension][filepath]
            unique_words = sorted(list(set(item['meta_data']['word'] for item in questions_from_file)))
            
            # Calculate number of words to sample from this file according to ratio
            num_words_for_file = round(num_words_to_sample * ratio)
            if num_words_for_file == 0 and num_words_to_sample > 0:
                 print(f"Info: Calculated 0 words to sample for '{filepath}' in '{dimension}' due to low ratio/target. Skipping.")
                 continue

            if len(unique_words) < num_words_for_file:
                print(f"Warning: In '{filepath}', unique words for '{dimension}' ({len(unique_words)}) is less than target ({num_words_for_file}). Using all available.")
                sampled_words = unique_words
            else:
                sampled_words = random.sample(unique_words, num_words_for_file)
            
            # Add all questions for the sampled words
            for question in questions_from_file:
                if question['meta_data']['word'] in sampled_words:
                    questions_for_dim.append(question)
            
            print(f"Added {len(sampled_words)} words for '{dimension}' from '{filepath}'.")

        all_final_questions.extend(questions_for_dim)

    # Update the 'answer' field and add the 'audio' key
    for question in all_final_questions:
        if 'meta_data' in question:
            meta = question['meta_data']
            if 'answer' in meta:
                question['answer'] = meta['answer']
            
            if 'language' in meta and 'word' in meta:
                language = meta['language']
                word = meta['word']
                question['audio'] = f"tts/{language}/{word}.wav"

    # 5. Shuffle and save final results
    random.shuffle(all_final_questions)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_final_questions, f, indent=4, ensure_ascii=False)

    print("\n--- Done ---")
    print(f"Total {len(all_final_questions)} questions generated.")
    print(f"Results saved to '{output_file}'.")


if __name__ == "__main__":
    main()