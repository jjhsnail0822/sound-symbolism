import argparse
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
from pathlib import Path
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Get embeddings for meanings')
parser.add_argument('-l', '--language', type=str, default='ko', help='Language code (default: ko)', choices=['ko', 'en', 'ja', 'fr'])
parser.add_argument('-d', '--dimension', type=int, default=256, help='Dimension of the embeddings (default: 256)')
args = parser.parse_args()

LANGUAGE = args.language
DIMENSION = args.dimension

# Load environment variables from .env.local file
env_path = Path('.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text):
    """
    Get the embedding for a given text using OpenAI's API.
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large",
        dimensions=DIMENSION,
    )
    return response.data[0].embedding

with open(f'dataset/1_preprocess/nat/{LANGUAGE}.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

embeddings = []
for word in tqdm(data):
    if not word['found']:
        continue
    # is meaning a list?
    if isinstance(word['meaning'], list):
        meaning = word['meaning'][0]
    else:
        meaning = word['meaning']
    # get embedding
    embedding = get_embedding(meaning)
    embeddings.append({
        'word': word['word'],
        'meaning': meaning,
        'embedding': embedding
    })

# Save the embeddings to a file
with open(f'dataset/1_preprocess/nat/{LANGUAGE}_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)



########################


# LANGUAGE = 'en'

# with open(f'dataset/1_preprocess/nat/{LANGUAGE}.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# with open(f'dataset/1_preprocess/nat/{LANGUAGE}_embeddings_old.pkl', 'rb') as f:
#     pickle_data = pickle.load(f)

# # if pickle_data['word'] not in data['word'], drop it
# new_pickle_data = []
# for word in data:
#     for item in pickle_data:
#         if word['word'] == item['word']:
#             new_pickle_data.append(item)
#             break

# # Save the new embeddings to a file
# with open(f'dataset/1_preprocess/nat/{LANGUAGE}_embeddings.pkl', 'wb') as f:
#     pickle.dump(new_pickle_data, f)
