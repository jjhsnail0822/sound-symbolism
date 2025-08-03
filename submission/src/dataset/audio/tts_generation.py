# melotts works on python 3.9 environment

import json
from melo.api import TTS
from tqdm import tqdm

langs = ['en', 'fr', 'ja', 'ko']
melo_langs = {'en': 'EN', 'fr': 'FR', 'ja': 'JP', 'ko': 'KR'}
melo_ids = {'en': 'EN-Default', 'fr': 'FR', 'ja': 'JP', 'ko': 'KR'}
OUTPUT_BASE_PATH = 'data/processed/nat/tts'

class TTSGenerator:
    def __init__(self, language):
        self.language = language
        self.melo_lang = melo_langs[language]
        self.melo_id = melo_ids[language]
        self.device = 'auto'
        self.model = TTS(language=self.melo_lang, device=self.device)
        self.speaker_ids = self.model.hps.data.spk2id
        self.output_path = f"{OUTPUT_BASE_PATH}/{self.language}"
        self.speed = 1.0

    def generate(self, text):
        self.model.tts_to_file(text, self.speaker_ids[self.melo_id], f'{self.output_path}/{text}.wav', speed=self.speed)
        return

for lang in langs:
    with open(f'data/processed/nat/{lang}.json', 'r') as f:
        data = json.load(f)
    tts = TTSGenerator(lang)
    for item in tqdm(data):
        tts.generate(item['word'])