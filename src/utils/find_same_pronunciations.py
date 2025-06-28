import json
import itertools
import argparse

def find_same_pronunciations(input_path, lang):
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    ipa_map = {}
    for rec in records:
        ipa = rec.get('ipa', '').strip()
        word = rec.get('word')
        if ipa and word:
            ipa_map.setdefault(ipa, []).append(word)
    # for each IPA with more than one word, print all unique pairs
    with open(f'data/processed/nat/same_pronunciations_{lang}.txt', 'w', encoding='utf-8') as f:
        for ipa, words in ipa_map.items():
            if len(words) > 1:
                for w1, w2 in itertools.combinations(words, 2):
                    f.write(f"{w1}\t{w2}\t{ipa}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", "-l", type=str, default='en', help='Language to process')
    args = parser.parse_args()
    ipa_file = f'data/processed/nat/{args.lang}.json'
    find_same_pronunciations(ipa_file, args.lang)