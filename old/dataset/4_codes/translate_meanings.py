from dotenv import load_dotenv
import json
from openai import OpenAI
import os
from pathlib import Path
from tqdm import tqdm

LANGUAGES = ['en', 'fr', 'ja', 'ko']
DIMENSION = 256
MODEL_NAME = 'gpt-4.1'

# Load environment variables from .env.local file
env_path = Path('.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

with open('analysis/experiments/prompts.json', 'r', encoding='utf-8') as f:
    prompts = json.load(f)
    if 'meaning_translation' not in prompts or 'user_prompt' not in prompts['meaning_translation']:
        raise KeyError("Prompt structure 'meaning_translation.user_prompt' not found in prompts.json")
    BASE_PROMPT_TEMPLATE = prompts['meaning_translation']['user_prompt']

def translate(word, meaning):
    prompt = BASE_PROMPT_TEMPLATE.format(word=word, definition=meaning, target_language="English")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    translated_meaning = response.choices[0].message.content.strip()
    if translated_meaning == "":
        raise ValueError(f"Translation failed for word: {word} with meaning: {meaning}")
    print(f"Translated meaning for word '{word}': {translated_meaning}")
    return translated_meaning

for LANGUAGE in LANGUAGES:
    # Load the original data
    with open(f'dataset/1_preprocess/nat/{LANGUAGE}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Translate meanings
    for item in tqdm(data, desc=f'Translating meanings for {LANGUAGE}'):
        word = item['word']
        meaning = item['meaning']
        if isinstance(meaning, list):
            meaning = meaning[0]
        if LANGUAGE == 'en':
            item['en_meaning'] = meaning
            continue
        translated_meaning = translate(word, meaning)
        item['en_meaning'] = translated_meaning
    
    # Save the translated data
    with open(f'dataset/1_preprocess/nat/{LANGUAGE}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Translated meanings saved to dataset/1_preprocess/nat/{LANGUAGE}.json")