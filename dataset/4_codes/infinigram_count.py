import requests
import json
from tqdm import tqdm

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

with open('dataset/1_preprocess/nat/crosslingual_with_en/crosslingual_clustered.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for cluster in tqdm(data):
    for word in cluster['words']:
        word['infinigram_count'] = get_infinigram_count(word['word'])
        print(f"Word: {word['word']}, Count: {word['infinigram_count']}")

with open('dataset/1_preprocess/nat/crosslingual_with_en/crosslingual_clustered_with_count.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)