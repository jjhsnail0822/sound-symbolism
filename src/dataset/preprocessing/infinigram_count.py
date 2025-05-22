import requests
import json
from tqdm import tqdm

langs = ['en', 'fr', 'ja', 'ko']

def get_infinigram_count(query):
    payload = {
        'index': 'v4_olmo-2-0325-32b-instruct_llama',
        'query_type': 'count',
        'query': query,
    }
    while True:
        try:
            result = requests.post('https://api.infini-gram.io/', json=payload).json()
            if 'count' in result:
                break
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying...")
            continue
    return result['count']

for lang in langs:
    with open(f'data/processed/nat/{lang}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Processing {lang} data...")
    for item in tqdm(data):
        item['infinigram_count'] = get_infinigram_count(item['word'])
    with open(f'data/processed/nat/{lang}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Finished processing {lang} data.")
