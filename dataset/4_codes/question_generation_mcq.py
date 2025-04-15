import json
import random

LANGUAGE = "ja"
TASK = "understanding"
EXPERIMENT_NAME = "unmasked_word_to_meaning_mcq"
PROMPT_ROLE = "user_prompt"
MASKING = False
MASKING_WORD = "[__]"
MAX_OPTION = 4

random.seed(42)

with open(f'dataset/2_dialogue/nat/{LANGUAGE}.json', 'r', encoding='utf-8') as f:
    dialogues = json.load(f)

with open(f'dataset/1_preprocess/nat/{LANGUAGE}_clustered.json', 'r', encoding='utf-8') as f:
    clustered_words = json.load(f)

with open('analysis/experiments/prompts.json', 'r', encoding='utf-8') as f:
    prompts = json.load(f)

# Generate word to cluster mapping
word_to_cluster = {}
for cluster in clustered_words:
    for word in cluster['words']:
        word_to_cluster[word['word']] = cluster['cluster_id']

# Generate MCQ questions
result_data = []
for subject_word in dialogues:
    for dialogue in subject_word['dialogues']:
        if not dialogue['meta_data']['success']:
            print(f"Skipping dialogue for: {subject_word['word']}, num: {dialogue['meta_data']['dialogue_num']}")
            continue
        dialogue_text = ""
        for utterance in dialogue['dialogue']:
            if MASKING and utterance['index'] == dialogue['meta_data']['ss_idx']: # mask the subject word
                    dialogue_text += f"{utterance['speaker']}: {utterance['text'].replace(subject_word['word'], MASKING_WORD)}\n"
            else: # keep the subject word
                dialogue_text += f"{utterance['speaker']}: {utterance['text']}\n"
        dialogue_text = dialogue_text.strip()
        word_text = subject_word['word']
        # choose clusters randomly 0 to len(clustered_words), not including the current cluster
        random_clusters = random.sample([c['cluster_id'] for c in clustered_words if c['cluster_id'] != word_to_cluster[word_text]], MAX_OPTION - 1)
        # choose random meanings from the clusters
        random_meanings = []
        for cluster_id in random_clusters:
            cluster = next((c for c in clustered_words if c['cluster_id'] == cluster_id), None)
            if cluster:
                random_meaning = random.choice(cluster['words'])['meaning']
                if isinstance(random_meaning, list):
                    random_meaning = random_meaning[0]
                random_meanings.append(random_meaning)
                continue
            raise ValueError(f"Cluster ID {cluster_id} not found in clustered words.")
        answer_meaning = subject_word['meaning'][0] if isinstance(subject_word['meaning'], list) else subject_word['meaning']
        options = [answer_meaning] + random_meanings
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
        # create the prompt
        question = prompts[TASK][EXPERIMENT_NAME][LANGUAGE][PROMPT_ROLE].format(word=word_text, dialogue=dialogue_text, options=option_string, MAX_OPTION=MAX_OPTION)
        # create the result data
        result_data.append({
            "question": question,
            "answer": answer,
            "meta_data": {
                "language": LANGUAGE,
                "word": word_text,
                "meaning": subject_word['meaning'],
                "cluster_id": word_to_cluster[word_text],
                "num_utterances": dialogue['meta_data']['num_utterances'],
                "dialogue_num": dialogue['meta_data']['dialogue_num'],
                "ss_idx": dialogue['meta_data']['ss_idx'],
            }
        })

# Save the result data to a JSON file
with open(f'dataset/3_questions/nat/{TASK}-{EXPERIMENT_NAME}-{LANGUAGE}.json', 'w', encoding='utf-8') as f:
    json.dump(result_data, f, ensure_ascii=False, indent=4)
    
print(f"Generated {len(result_data)} MCQ questions.")
