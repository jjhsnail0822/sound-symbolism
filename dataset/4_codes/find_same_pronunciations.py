import json
import itertools

LANG = 'ko'

def find_same_pronunciations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    ipa_map = {}
    for rec in records:
        ipa = rec.get('ipa', '').strip()
        word = rec.get('word')
        if ipa and word:
            ipa_map.setdefault(ipa, []).append(word)
    # for each IPA with more than one word, print all unique pairs
    with open(f'dataset/1_preprocess/nat/same_pronunciations_{LANG}.txt', 'w', encoding='utf-8') as f:
        for ipa, words in ipa_map.items():
            if len(words) > 1:
                for w1, w2 in itertools.combinations(words, 2):
                    f.write(f"{w1}\t{w2}\t{ipa}\n")

if __name__ == "__main__":
    ipa_file = f'dataset/1_preprocess/nat/{LANG}_ipa.json'
    find_same_pronunciations(ipa_file)