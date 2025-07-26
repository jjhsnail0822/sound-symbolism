import json
import random
from collections import defaultdict
import os

random.seed(0)  # For reproducibility

path_common = "data/prompts/semantic_dimension/semantic_dimension_binary_audio-common.json"
path_rare = "data/prompts/semantic_dimension/semantic_dimension_binary_audio-rare.json"
path_constructed = "data/prompts/semantic_dimension/semantic_dimension_binary_audio-constructed.json"
output_file = "data/human_eval/label_studio_survey.json"
output_file_csv = "data/human_eval/label_studio_survey.csv"

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

    questions = []
    
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
            if word_tuple not in used_words:
                sampled_questions_1.append(item)
                used_words.add(word_tuple)
                used_langs_for_1.append(lang)
                natural_by_lang_and_answer[lang][1].pop(i)
                break
    
    num_art_needed = 4 - len(sampled_questions_1)
    for i, item in enumerate(art_by_answer[1]):
        if len(sampled_questions_1) >= 4:
            break
        word_tuple = (item['meta_data']['language'], item['meta_data']['word'])
        if word_tuple not in used_words:
            sampled_questions_1.append(item)
            used_words.add(word_tuple)

    # Sample for answer 2
    langs_for_2 = [lang for lang in natural_languages if natural_by_lang_and_answer[lang][2] and lang not in used_langs_for_1]
    for lang in langs_for_2:
        if len(sampled_questions_2) >= 2:
            break
        for i, item in enumerate(natural_by_lang_and_answer[lang][2]):
            word_tuple = (item['meta_data']['language'], item['meta_data']['word'])
            if word_tuple not in used_words:
                sampled_questions_2.append(item)
                used_words.add(word_tuple)
                natural_by_lang_and_answer[lang][2].pop(i)
                break

    num_art_needed = 4 - len(sampled_questions_2)
    for i, item in enumerate(art_by_answer[2]):
        if len(sampled_questions_2) >= 4:
            break
        word_tuple = (item['meta_data']['language'], item['meta_data']['word'])
        if word_tuple not in used_words:
            sampled_questions_2.append(item)
            used_words.add(word_tuple)

    questions.extend(sampled_questions_1)
    questions.extend(sampled_questions_2)
    
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

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_final_questions, f, ensure_ascii=False, indent=4)

with open(output_file_csv, 'w', encoding='utf-8') as f:
    f.write("id,prompt,answer,audio\n")
    for i, question in enumerate(all_final_questions):
        question_text = question.get('question', '')
        answer = question.get('answer', '')
        audio = question.get('audio', '')
        f.write(f"{i+1},{question_text},{answer},{audio}\n")