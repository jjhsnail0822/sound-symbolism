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
    IS_OPTION_MEANING = True if EXPERIMENT_NAME == "ipa_unmasked_word_to_meaning_mcq_no_dialogue" else False # whether the options are meaning or word
    MASKING = False if EXPERIMENT_NAME == "ipa_unmasked_word_to_meaning_mcq_no_dialogue" else True # whether to mask the subject word
    MASKING_WORD = "[__]"
    MAX_OPTION = 4
    NONE_OF_THE_OTHERS = False # whether to substitute an answer with "none of the others" option
    ONE_DIALOGUE = False # whether to use only one dialogue for each word

    with open(f'data/dialogues/{LANGUAGE}.json', 'r', encoding='utf-8') as f:
        dialogues = json.load(f)

    with open(f'data/processed/nat/clustering/{LANGUAGE}_clustered.json', 'r', encoding='utf-8') as f:
        clustered_words = json.load(f)

    with open('data/prompts/prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    def generate_options(random_clusters, is_option_meaning=False):
        # choose random meanings from the clusters
        random_options = []
        for cluster_id in random_clusters:
            cluster = next((c for c in clustered_words if c['cluster_id'] == cluster_id), None)
            if cluster:
                random_meaning = random.choice(cluster['words'])['meaning'] if is_option_meaning else random.choice(cluster['words'])['ipa'].replace(" ", "")
                if isinstance(random_meaning, list):
                    random_meaning = random_meaning[0]
                random_options.append(random_meaning)
                continue
            raise ValueError(f"Cluster ID {cluster_id} not found in clustered words.")
        meaning = subject_word['meaning'][0] if isinstance(subject_word['meaning'], list) else subject_word['meaning']
        right_option = meaning if is_option_meaning else subject_word['ipa'].replace(" ", "")
        if NONE_OF_THE_OTHERS:
            right_option = "None of the others"
        options = [right_option] + random_options
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
            if word_text: # Only map if word text exists
                # Store IPA (cleaned) and cluster_id. Handle missing IPA.
                word_details_map[word_text] = {
                    'ipa': ipa.replace(" ", "") if ipa else None, # Store cleaned IPA or None
                    'cluster_id': cluster_id
                    # Add meaning if needed later, but dialogues already has it
                    # 'meaning': word_info.get('meaning')
                }
    
    # Generate word to IPA mapping for the dialogues
    for subject_word in dialogues:
        word_text = subject_word['word']
        if word_text in word_details_map:
            subject_word['ipa'] = word_details_map[word_text]['ipa']
            subject_word['cluster_id'] = word_details_map[word_text]['cluster_id']
        else:
            print(f"Warning: {word_text} not found in clustered words.")

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
                    else: # change the word to ipa
                        replaced_text = re.sub(pattern, '[' + subject_word['ipa'].replace(" ", "") + ']', utterance['text'], flags=re.IGNORECASE)
                        if '[' + subject_word['ipa'].replace(" ", "") + ']' not in replaced_text:
                            replaced_text = utterance['text'].replace(subject_word['word'], '[' + subject_word['ipa'].replace(" ", "") + ']')
                        dialogue_text += f"{utterance['speaker']}: {replaced_text}\n"
                else:
                    dialogue_text += f"{utterance['speaker']}: {utterance['text']}\n"
            dialogue_text = dialogue_text.strip()
            word_text = subject_word['word']
            # choose clusters randomly 0 to len(clustered_words), not including the current cluster
            random_clusters = random.sample([c['cluster_id'] for c in clustered_words if c['cluster_id'] != word_to_cluster[word_text]], MAX_OPTION - 1)
            # generate options
            option_string, answer, meaning_text = generate_options(random_clusters, is_option_meaning=IS_OPTION_MEANING)
            # create the prompt
            question = prompts[TASK][EXPERIMENT_NAME][LANGUAGE][PROMPT_ROLE].format(word=subject_word['ipa'].replace(" ",""), meaning=meaning_text, options=option_string, MAX_OPTION=MAX_OPTION, MASKING_WORD=MASKING_WORD)
            # create the result data
            result_data.append({
                "question": question,
                "answer": answer,
                "meta_data": {
                    "language": LANGUAGE,
                    "word": subject_word['word'],
                    "meaning": subject_word['meaning'],
                    "ipa": subject_word['ipa'].replace(" ", ""),
                    "cluster_id": word_to_cluster[subject_word['word']],
                }
            })

    # Save the result data to a JSON file
    if not os.path.exists('data/prompts/understanding/pair_matching/ipa'):
        os.makedirs('data/prompts/understanding/pair_matching/ipa')
    with open(f'data/prompts/understanding/pair_matching/ipa/{EXPERIMENT_NAME}-{LANGUAGE}.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
        
    print(f"Generated {len(result_data)} MCQ questions for {experiment_name} in {language}.")
    return

if __name__ == "__main__":
    for lang in ["en", "fr", "ja", "ko"]:
        generate_datasets(language=lang, task="understanding", experiment_name="ipa_unmasked_word_to_meaning_mcq_no_dialogue")
        generate_datasets(language=lang, task="understanding", experiment_name="ipa_masked_meaning_to_word_mcq_no_dialogue")
