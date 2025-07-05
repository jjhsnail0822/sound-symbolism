import json
from pathlib import Path

class PhonemeData:
    def __init__(self, data_dir: str, auto_verify=True):
        self.data_dir = Path(data_dir) / 'resources'
        
        self.feature_to_score = self._load_json('feature_to_score.json')
        self.ipa_to_alphabet  = self._load_json('ipa_to_alphabet.json')
        self.ipa_to_feature = self._load_json('ipa_to_feature.json')
        self.ipa_to_word = self._load_json('ipa_to_word.json')        
        self.phonemes = self._load_json('phonemes.json')

        if auto_verify:
            self.verify()
    
    def _load_json(self, file_name: str):
        with open(self.data_dir / file_name, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def verify(self):
        # Feature
        features1 = set(list(self.feature_to_score.values())[0].keys())
        features2 = set(self.ipa_to_feature.values())
        features2.discard('back') 
        if features1 != features2:
            raise AssertionError("Feature sets don't match")
        
        # Phonemes (IPA)
        ipa1 = set(self.ipa_to_alphabet.keys())
        ipa2 = set(self.ipa_to_feature.keys())
        if ipa1 != ipa2:
            raise AssertionError("IPA symbol sets don't match")

        for lang, ipa_to_word in self.ipa_to_word.items():
            lang = lang.lower()
            ipa_words = list(ipa_to_word.keys())
            ipa_set = set(''.join(ipa_words))
            if ('o' in ipa_set) and ('w' in ipa_set) :
                ipa_set.add('ow')
            if ('e' in ipa_set )and ('j' in ipa_set) :
                ipa_set.add('ej')
            
            diff = (ipa1 - ipa_set)
            if diff:
                if lang == 'en_uk' and diff == {'ow'}:
                    pass
                elif lang == 'ko' and diff == {'ɑ', 'ð', 'z', 'v', 'f'}:
                    pass
                elif lang == 'ja' and diff == {'ʃ', 'f', 'ɑ', 'ow', 'ð'}:
                    pass
                elif lang == 'fr_fr' and diff == {'ð'}:
                    pass
                else:       
                    print(lang, ipa1-ipa_set)
                    print(ipa1)
                    print(ipa_set)
                    raise AssertionError("Some phonemes are different with word dictionary")

        print("[INFO] All consistency tests passed")
