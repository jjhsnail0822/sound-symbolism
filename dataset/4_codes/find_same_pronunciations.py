import json
import itertools
import argparse

LANG = 'ko'
# LANG = 'fr'

def find_same_pronunciations(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default='ko', help='Language to process')
    parser.add_argument("--input", type=str, help="Input file path")
    args = parser.parse_args()
    ipa_file = f'dataset/1_preprocess/nat/{args.lang}_ipa.json'
    # ipa_file = f'/scratch2/sheepswool/workspace/sound-symbolism/dataset/1_preprocess/nat/{args.lang}_ipa_filtered.json'
    find_same_pronunciations(ipa_file)