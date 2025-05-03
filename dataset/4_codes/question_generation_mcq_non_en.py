import json
import random
import re

random.seed(42)

LANGUAGES = ["fr", "ja", "ko", "cross_language"]

def generate_datasets(language, task, experiment_name):
    LANGUAGE = language
    TASK = task
    EXPERIMENT_NAME = experiment_name
    PROMPT_ROLE = "user_prompt"
    IS_OPTION_MEANING = True if EXPERIMENT_NAME == "non_en_unmasked_word_to_meaning_mcq" else False # whether the options are meaning or word
    MASKING = False if EXPERIMENT_NAME == "non_en_unmasked_word_to_meaning_mcq" else True # whether to mask the subject word
    MASKING_WORD = "[__]"
    MAX_OPTION = 4
    NONE_OF_THE_OTHERS = False # whether to substitute an answer with "none of the others" option
    ONE_DIALOGUE = False # whether to use only one dialogue for each word

    if LANGUAGE == "cross_language":
        dialogues = []
        for lang in LANGUAGES:
            if lang == "cross_language": continue
            with open(f'dataset/2_dialogue/nat/{lang}2en.json', 'r', encoding='utf-8') as f:
                lang_dialogues = json.load(f)
            dialogues.extend(lang_dialogues)
    else:
        with open(f'dataset/2_dialogue/nat/{LANGUAGE}2en.json', 'r', encoding='utf-8') as f:
            dialogues = json.load(f)

    with open(f'dataset/1_preprocess/nat/crosslingual_clustered_{LANGUAGE}.json', 'r', encoding='utf-8') as f:
        clustered_words = json.load(f)

    with open('analysis/experiments/prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    def generate_options(distractor_options, is_option_meaning=False):
        meaning = subject_word['en_meaning']
        right_option = meaning if is_option_meaning else subject_word['ipa'].replace(" ", "")
        if NONE_OF_THE_OTHERS:
            right_option = "None of the others"
        options = [right_option] + distractor_options
        answer_idx = [i for i in range(MAX_OPTION)]
        # shuffle the options and answer index together
        random.shuffle(answer_idx)
        shuffled_options = [options[i] for i in answer_idx]
        answer = answer_idx.index(0) + 1
        # create the option string
        option_string = ""
        for i, option in enumerate(shuffled_options):
            option_string += f"{i + 1}: {option}\n"
        option_string = option_string.strip()
        return option_string, answer, meaning

    # Generate word to cluster mapping
    word_to_cluster = {}
    for cluster in clustered_words:
        for word in cluster['words']:
            word_to_cluster[word['word']] = cluster['cluster_id']

    # Generate word to IPA mapping
    word_details_map = {}
    for cluster in clustered_words:
        cluster_id = cluster.get('cluster_id')
        if cluster_id is None: continue
        for word_info in cluster.get('words', []):
            word_text = word_info.get('word')
            ipa = word_info.get('ipa')
            en_meaning = word_info.get('en_meaning')
            distractors = word_info.get('distractors', [])
            if word_text: # Only map if word text exists
                # Store IPA (cleaned) and cluster_id. Handle missing IPA.
                word_details_map[word_text] = {
                    'ipa': ipa.replace(" ", "") if ipa else None, # Store cleaned IPA or None
                    'cluster_id': cluster_id,
                    'en_meaning': en_meaning,
                    'distractors': distractors
                }
    
    # Generate word to IPA mapping for the dialogues
    for subject_word in dialogues:
        word_text = subject_word['word']
        if word_text in word_details_map:
            subject_word['ipa'] = word_details_map[word_text]['ipa']
            subject_word['cluster_id'] = word_details_map[word_text]['cluster_id']
            subject_word['en_meaning'] = word_details_map[word_text]['en_meaning']
            subject_word['distractors'] = word_details_map[word_text]['distractors']
        else:
            print(f"Warning: {word_text} not found in clustered words.")

    # Generate MCQ questions
    result_data = []
    for subject_word in dialogues:
        for dialogue in subject_word['dialogues']:
            if not dialogue['meta_data']['success']:
                print(f"Skipping dialogue for: {subject_word['word']}, num: {dialogue['meta_data']['dialogue_num']}")
                continue
            if ONE_DIALOGUE and dialogue['meta_data']['dialogue_num'] != 1:
                continue
            dialogue_text = ""
            for utterance in dialogue['dialogue']:
                if utterance['index'] == dialogue['meta_data']['ss_idx']:
                    # pattern = [...]
                    pattern = r"\[.+\]"
                    if MASKING:
                        replaced_text = re.sub(pattern, MASKING_WORD, utterance['text'], flags=re.IGNORECASE)
                        if MASKING_WORD not in replaced_text:
                            replaced_text = utterance['text'].replace(subject_word['word'], MASKING_WORD)
                        dialogue_text += f"{utterance['speaker']}: {replaced_text}\n"
                    else: # change the word to ipa
                        replaced_text = re.sub(pattern, '[' + subject_word['ipa'].replace(" ", "") + ']', utterance['text'], flags=re.IGNORECASE)
                        if '[' + subject_word['ipa'].replace(" ", "") + ']' not in replaced_text:
                            replaced_text = utterance['text'].replace(subject_word['word'], '[' + subject_word['ipa'].replace(" ", "") + ']')
                        dialogue_text += f"{utterance['speaker']}: {replaced_text}\n"
                else:
                    dialogue_text += f"{utterance['speaker']}: {utterance['text']}\n"
            dialogue_text = dialogue_text.strip()
            word_text = subject_word['word']
            # get distractor options from the word
            if IS_OPTION_MEANING:
                distractor_words = subject_word['distractors']
                distractor_options = [word_details_map[word['word']]['en_meaning'] for word in distractor_words if word['word'] in word_details_map]
            else:
                distractor_words = subject_word['distractors']
                distractor_options = [distractor_word['ipa'].replace(" ", "") for distractor_word in distractor_words]
            # generate options
            option_string, answer, meaning_text = generate_options(distractor_options, is_option_meaning=IS_OPTION_MEANING)
            # create the prompt
            question = prompts[TASK][EXPERIMENT_NAME][PROMPT_ROLE].format(word=subject_word['ipa'].replace(" ",""), meaning=meaning_text, dialogue=dialogue_text, options=option_string, MAX_OPTION=MAX_OPTION, MASKING_WORD=MASKING_WORD)
            # create the result data
            result_data.append({
                "question": question,
                "answer": answer,
                "meta_data": {
                    "language": LANGUAGE,
                    "word": subject_word['word'],
                    "meaning": subject_word['en_meaning'],
                    "ipa": subject_word['ipa'].replace(" ", ""),
                    "cluster_id": word_to_cluster[subject_word['word']],
                    "num_utterances": dialogue['meta_data']['num_utterances'],
                    "dialogue_num": dialogue['meta_data']['dialogue_num'],
                    "ss_idx": dialogue['meta_data']['ss_idx'],
                }
            })

    # Save the result data to a JSON file
    with open(f'dataset/3_questions/nat/understanding_non_en_pair_matching/{EXPERIMENT_NAME}-{LANGUAGE}.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
        
    print(f"Generated {len(result_data)} MCQ questions for {experiment_name} in {language}.")
    return

if __name__ == "__main__":
    for lang in LANGUAGES:
        generate_datasets(language=lang, task="understanding", experiment_name="non_en_unmasked_word_to_meaning_mcq")
        generate_datasets(language=lang, task="understanding", experiment_name="non_en_masked_meaning_to_word_mcq")
