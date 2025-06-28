import json
import random
import re
import os

random.seed(42)

def generate_datasets(language, task, experiment_name):
    LANGUAGE = language
    TASK = task
    EXPERIMENT_NAME = experiment_name
    PROMPT_ROLE = "user_prompt"
    IS_OPTION_MEANING = True if EXPERIMENT_NAME == "unmasked_word_to_meaning_mcq_no_dialogue" else False # whether the options are meaning or word
    MASKING = False if EXPERIMENT_NAME == "unmasked_word_to_meaning_mcq_no_dialogue" else True # whether to mask the subject word
    MASKING_WORD = "[__]"
    AUDIO_TOKEN = "<AUDIO>"
    MAX_OPTION = 4
    NONE_OF_THE_OTHERS = False # whether to substitute an answer with "none of the others" option
    ONE_DIALOGUE = False # whether to use only one dialogue for each word

    with open(f'data/dialogues/{LANGUAGE}.json', 'r', encoding='utf-8') as f:
        dialogues = json.load(f)

    with open(f'data/processed/nat/clustering/{LANGUAGE}_clustered.json', 'r', encoding='utf-8') as f:
        clustered_words = json.load(f)

    with open('data/prompts/prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    def generate_options(subject_word, random_clusters, is_option_meaning=False):
        # choose random meanings from the clusters
        random_options = []
        random_options_info = []  # Store language info for each option
        for cluster_id in random_clusters:
            cluster = next((c for c in clustered_words if c['cluster_id'] == cluster_id), None)
            if cluster:
                random_word_info = random.choice(cluster['words'])
                if is_option_meaning:
                    random_option = random_word_info['meaning']
                    if isinstance(random_option, list):
                        random_option = random_option[0]
                    # Don't store language info for meanings
                else:
                    random_option = random_word_info['word']
                    random_option_lang = LANGUAGE  # words are in the current language
                    random_options_info.append({
                        'text': random_option,
                        'language': random_option_lang
                    })
                
                random_options.append(random_option)
                continue
            raise ValueError(f"Cluster ID {cluster_id} not found in clustered words.")
        
        meaning = subject_word['meaning'][0] if isinstance(subject_word['meaning'], list) else subject_word['meaning']
        if is_option_meaning:
            right_option = meaning
            # Don't create options_info for meanings
            options_info = None
        else:
            right_option = subject_word['word']
            right_option_lang = LANGUAGE
            options_info = [{
                'text': right_option,
                'language': right_option_lang
            }] + random_options_info
            
        if NONE_OF_THE_OTHERS:
            right_option = "None of the others"
            
        options = [right_option] + random_options
        
        answer_idx = [i for i in range(MAX_OPTION)]
        # shuffle the options and answer index together
        random.shuffle(answer_idx)
        shuffled_options = [options[i] for i in answer_idx]
        
        if not is_option_meaning:
            shuffled_options_info = [options_info[i] for i in answer_idx]
        else:
            shuffled_options_info = None
            
        answer = answer_idx.index(0) + 1
        
        # create the option string
        option_string = ""
        for i, option in enumerate(shuffled_options):
            if is_option_meaning:
                option_string += f"{i + 1}: {option}\n"
            else:
                option_string += f"{i + 1}: <AUDIO: {option}>\n"
        option_string = option_string.strip()
        return option_string, answer, meaning, shuffled_options_info

    # Generate word to cluster mapping
    word_to_cluster = {}
    for cluster in clustered_words:
        for word in cluster['words']:
            word_to_cluster[word['word']] = cluster['cluster_id']

    # Generate MCQ questions
    result_data = []
    for subject_word in dialogues:
        for dialogue in subject_word['dialogues'][:1]: # only use the first dialogue
            if not dialogue['meta_data']['success']:
                print(f"Skipping dialogue for: {subject_word['word']}, num: {dialogue['meta_data']['dialogue_num']}")
                continue
            if ONE_DIALOGUE and dialogue['meta_data']['dialogue_num'] != 1:
                continue
            dialogue_text = ""
            for utterance in dialogue['dialogue']:
                if utterance['index'] == dialogue['meta_data']['ss_idx']:
                    pattern = r'\b' + re.escape(subject_word['word']) + r'\b'
                    if MASKING:
                        replaced_text = re.sub(pattern, MASKING_WORD, utterance['text'], flags=re.IGNORECASE)
                        if MASKING_WORD not in replaced_text:
                            replaced_text = utterance['text'].replace(subject_word['word'], MASKING_WORD)
                        dialogue_text += f"{utterance['speaker']}: {replaced_text}\n"
                    else: # keep the subject word
                        replaced_text = re.sub(pattern, '[' + subject_word['word'] + ']', utterance['text'], flags=re.IGNORECASE)
                        if '[' + subject_word['word'] + ']' not in replaced_text:
                            replaced_text = utterance['text'].replace(subject_word['word'], '[' + subject_word['word'] + ']')
                        dialogue_text += f"{utterance['speaker']}: {replaced_text}\n"
                else:
                    dialogue_text += f"{utterance['speaker']}: {utterance['text']}\n"
            dialogue_text = dialogue_text.strip()
            word_text = subject_word['word']
            # choose clusters randomly 0 to len(clustered_words), not including the current cluster
            random_clusters = random.sample([c['cluster_id'] for c in clustered_words if c['cluster_id'] != word_to_cluster[word_text]], MAX_OPTION - 1)
            # generate options
            option_string, answer, meaning_text, options_info = generate_options(subject_word, random_clusters, is_option_meaning=IS_OPTION_MEANING)
            
            # create the result data
            result_entry = {
                "question": prompts[TASK][EXPERIMENT_NAME][LANGUAGE][PROMPT_ROLE].format(word=AUDIO_TOKEN, meaning=meaning_text, options=option_string, MAX_OPTION=MAX_OPTION, MASKING_WORD=MASKING_WORD),
                "answer": answer,
                "meta_data": {
                    "language": LANGUAGE,
                    "word": word_text,
                    "meaning": subject_word['meaning'],
                    "cluster_id": word_to_cluster[word_text],
                }
            }
            
            # Only add options_info if it's not None (i.e., for word options only)
            if options_info is not None:
                result_entry["options_info"] = options_info
                
            result_data.append(result_entry)

    # Save the result data to a JSON file
    if not os.path.exists('data/prompts/understanding/pair_matching/audiolm'):
        os.makedirs('data/prompts/understanding/pair_matching/audiolm')
    with open(f'data/prompts/understanding/pair_matching/audiolm/{EXPERIMENT_NAME}-{LANGUAGE}.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
        
    print(f"Generated {len(result_data)} MCQ questions.")
    return

if __name__ == "__main__":
    for lang in ["en", "fr", "ja", "ko"]:
        generate_datasets(language=lang, task="understanding", experiment_name="unmasked_word_to_meaning_mcq_no_dialogue")
        generate_datasets(language=lang, task="understanding", experiment_name="masked_meaning_to_word_mcq_no_dialogue")
