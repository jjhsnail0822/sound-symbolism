import json
import random
from collections import defaultdict
import os
import argparse

parser = argparse.ArgumentParser(description="Generate a survey")
parser.add_argument("--num_participants", "-n", required=True, type=int, help="Number of participants")
args = parser.parse_args()

def generate_survey(seed):
    random.seed(seed)  # For reproducibility
    
    while True:
        path_common = "data/prompts/semantic_dimension/semantic_dimension_binary_audio-common.json"
        path_rare = "data/prompts/semantic_dimension/semantic_dimension_binary_audio-rare.json"
        path_constructed = "data/prompts/semantic_dimension/semantic_dimension_binary_audio-constructed.json"
        output_file = f"data/human_eval/label_studio_survey_{seed}.json"
        output_file_csv = f"data/human_eval/label_studio_survey_{seed}.csv"

        sampling_dimensions = {
                "beautiful-ugly" : 8,
                "strong-weak" : 8,
                "big-small" : 8,
                "fast-slow" : 8,
                "sharp-round" : 8,
                "realistic-fantastical" : 8,
                "ordinary-unique" : 8,
                "simple-complex" : 8,
                "abrupt-continuous" : 8,
                "exciting-calming" : 8,
                "hard-soft" : 8,
                "happy-sad" : 8,
                "harsh-mellow" : 8,
                "heavy-light" : 8,
                "inhibited-free" : 8,
                "masculine-feminine" : 8,
                "solid-nonsolid" : 8,
                "tense-relaxed" : 8,
                "dangerous-safe" : 8
            }

        raw_data_natural = []
        raw_data_constructed = []
        with open(path_common, 'r', encoding='utf-8') as f:
            raw_data_natural.extend(json.load(f))
        with open(path_rare, 'r', encoding='utf-8') as f:
            raw_data_natural.extend(json.load(f))
        with open(path_constructed, 'r', encoding='utf-8') as f:
            raw_data_constructed.extend(json.load(f))

        # Split the data by dimension and language
        data_by_dim_and_source = defaultdict(lambda: defaultdict(list))

        for item in raw_data_natural:
            dimension = item.get("meta_data", {}).get("dimension")
            if dimension and dimension in sampling_dimensions:
                data_by_dim_and_source[dimension][item['meta_data']['language']].append(item)

        for item in raw_data_constructed:
            dimension = item.get("meta_data", {}).get("dimension")
            if dimension and dimension in sampling_dimensions:
                data_by_dim_and_source[dimension]['art'].append(item)

        data_prepared = {}
        used_words = set() # Track used words (lang, word) across all dimensions

        # For each dimension, sample 4 questions with answer 1 and 4 with answer 2.
        # Try to maintain a balance between natural and constructed languages.
        for dimension, sources in data_by_dim_and_source.items():
            if dimension not in sampling_dimensions:
                continue

            # --- Start of the dimension-specific retry loop ---
            # Try to sample for this dimension until successful.
            # This prevents restarting the entire survey generation if one dimension fails.
            while True:
                questions = []
                
                # Create a temporary copy of used_words to revert if this attempt fails.
                temp_used_words = used_words.copy()

                # Separate items by answer, source type, and language for natural languages
                natural_by_lang_and_answer = defaultdict(lambda: defaultdict(list))
                art_by_answer = defaultdict(list)

                natural_languages = [lang for lang in sources.keys() if lang != 'art']
                random.shuffle(natural_languages) # Ensure random selection of languages

                for lang, items in sources.items():
                    random.shuffle(items) # Shuffle items for random sampling
                    for item in items:
                        answer = item['meta_data']['answer']
                        if lang == 'art':
                            art_by_answer[answer].append(item)
                        else:
                            natural_by_lang_and_answer[lang][answer].append(item)

                # We need 4 questions for each answer (1 and 2)
                # Let's try to get 2 from natural (from 2 different languages) and 2 from 'art' for each answer.
                
                sampled_questions_1 = []
                sampled_questions_2 = []
                
                # Sample for answer 1
                langs_for_1 = [lang for lang in natural_languages if natural_by_lang_and_answer[lang][1]]
                used_langs_for_1 = []
                for lang in langs_for_1:
                    if len(sampled_questions_1) >= 2:
                        break
                    for i, item in enumerate(natural_by_lang_and_answer[lang][1]):
                        word_tuple = (item['meta_data']['language'], item['meta_data']['word'])
                        if word_tuple not in temp_used_words:
                            sampled_questions_1.append(item)
                            temp_used_words.add(word_tuple)
                            used_langs_for_1.append(lang)
                            # We don't pop from the list here to allow retries with the full dataset
                            break
                
                # Sample from 'art' for answer 1
                art_candidates_1 = [item for item in art_by_answer[1] if (item['meta_data']['language'], item['meta_data']['word']) not in temp_used_words]
                num_art_needed = 4 - len(sampled_questions_1)
                num_art_to_sample = min(num_art_needed, len(art_candidates_1))
                if num_art_to_sample > 0:
                    sampled_art = random.sample(art_candidates_1, num_art_to_sample)
                    sampled_questions_1.extend(sampled_art)
                    for item in sampled_art:
                        temp_used_words.add((item['meta_data']['language'], item['meta_data']['word']))

                # Sample for answer 2
                langs_for_2 = [lang for lang in natural_languages if natural_by_lang_and_answer[lang][2] and lang not in used_langs_for_1]
                for lang in langs_for_2:
                    if len(sampled_questions_2) >= 2:
                        break
                    for i, item in enumerate(natural_by_lang_and_answer[lang][2]):
                        word_tuple = (item['meta_data']['language'], item['meta_data']['word'])
                        if word_tuple not in temp_used_words:
                            sampled_questions_2.append(item)
                            temp_used_words.add(word_tuple)
                            # We don't pop from the list here
                            break

                # Sample from 'art' for answer 2
                art_candidates_2 = [item for item in art_by_answer[2] if (item['meta_data']['language'], item['meta_data']['word']) not in temp_used_words]
                num_art_needed = 4 - len(sampled_questions_2)
                num_art_to_sample = min(num_art_needed, len(art_candidates_2))
                if num_art_to_sample > 0:
                    sampled_art = random.sample(art_candidates_2, num_art_to_sample)
                    sampled_questions_2.extend(sampled_art)
                    for item in sampled_art:
                        temp_used_words.add((item['meta_data']['language'], item['meta_data']['word']))

                questions.extend(sampled_questions_1)
                questions.extend(sampled_questions_2)

                # --- Validation for the current dimension ---
                natural_lang_count = sum(1 for q in questions if q['meta_data']['language'] != 'art')
                total_questions_count = len(questions)

                if natural_lang_count == 4 and total_questions_count == 8:
                    # Success: commit the used words and break the loop for this dimension
                    used_words = temp_used_words
                    break
                # Failure: The loop will continue, trying a different random sample for this dimension.
            
            # Shuffle the questions to mix natural and constructed
            random.shuffle(questions)
            
            data_prepared[dimension] = questions

        # Update the 'answer' field and add the 'audio' key
        for dimension, questions in data_prepared.items():
            for question in questions:
                question['answer'] = question['meta_data']['answer']
                language = question['meta_data']['language']
                word = question['meta_data']['word']
                question['audio'] = f"/data/local-files/?d=tts/{language}/{word}.wav"

        all_final_questions = []
        for dimension, questions in data_prepared.items():
            if questions:
                all_final_questions.extend(questions)

        random.shuffle(all_final_questions)

        # Save the final questions to JSON
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        is_valid = True
        # 1. Check if the total number of questions is 152
        if len(all_final_questions) != 152:
            print(f"Seed {seed}: Failed validation. Total questions {len(all_final_questions)} != 152. Retrying...")
            is_valid = False
        
        # 2. Check if each dimension has exactly 4 questions of natural language
        if is_valid:
            for dimension, questions in data_prepared.items():
                natural_lang_count = sum(1 for q in questions if q['meta_data']['language'] != 'art')
                if natural_lang_count != 4:
                    print(f"Seed {seed}: Failed validation in dimension '{dimension}'. Natural language questions {natural_lang_count} != 4. Retrying...")
                    is_valid = False
                    break
        
        if is_valid:
            print(f"Seed {seed}: Survey generated successfully with {len(all_final_questions)} questions.")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_final_questions, f, ensure_ascii=False, indent=4)
            break


for i in range(args.num_participants):
    generate_survey(seed=i)  # Generate surveys with different seeds for variety