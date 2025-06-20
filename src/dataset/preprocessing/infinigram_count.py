import requests
import json
from tqdm import tqdm

langs = ['en', 'fr', 'ja', 'ko']
datasets = [
    'v4_olmo-mix-1124_llama',
    'v4_dolma-v1_7_llama',
    'v4_rpj_llama_s4',
    'v4_piletrain_llama',
    'v4_c4train_llama',
]

def get_infinigram_count(query, dataset):
    payload = {
        'index': dataset,
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
        item['infinigram_count'] = {}
        for dataset in datasets:
            item['infinigram_count'][dataset] = get_infinigram_count(item['word'], dataset)
        item['infinigram_count']['total'] = sum(item['infinigram_count'].values())
        with open(f'data/processed/nat/{lang}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Finished processing {lang} data.")
