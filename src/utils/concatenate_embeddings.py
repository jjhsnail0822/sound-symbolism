import pickle

langs = ['en', 'fr', 'ja', 'ko']

all_embeddings = []
for lang in langs:
    with open(f'data/processed/nat/{lang}_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
        for item in embeddings:
            item['language'] = lang

    all_embeddings.extend(embeddings)

with open(f'data/processed/nat/all_embeddings.pkl', 'wb') as f:
    pickle.dump(all_embeddings, f)