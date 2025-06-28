import os
import json

def extract_segments():
    base_dir = 'dataset/1_preprocess/nat'
    langs = ['en', 'fr', 'ja', 'ko']
    # map each segment to the first word it appears in
    segments = {}

    for lang in langs:
        json_path = os.path.join(base_dir, f'{lang}.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        for rec in records:
            ipa = rec.get('ipa', '')
            word = rec.get('word', '')
            for seg in ipa.split():
                if seg and seg not in segments:
                    segments[seg] = word

    for seg in sorted(segments):
        print(f'"{seg}"\t{segments[seg]}')

if __name__ == '__main__':
    extract_segments()