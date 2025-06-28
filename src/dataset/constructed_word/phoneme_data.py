import json
from pathlib import Path

class PhonemeData:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        self.feature_to_score = self.load_json('feature_to_score.json')
        self.ipa_to_alphabet  = self.load_json('ipa_to_alphabet.json')
        self.ipa_to_feature = self.load_json('ipa_to_feature.json')
        self.ipa_to_word = self.load_json('ipa_to_word_en_US.json')        
        self.phonemes = self.load_json('phonemes.json')
    
    def load_json(self, file_name: str):
        with open(self.data_dir / file_name, 'r', encoding='utf-8') as file:
            return json.load(file)
