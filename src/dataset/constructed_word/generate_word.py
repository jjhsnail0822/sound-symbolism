import argparse 
import unicodedata
from typing import List
from pathlib import Path
from itertools import product

import pandas as pd
from phoneme_data import PhonemeData

class CVCVWordGenerator:
    def __init__(self, phoneme_data:PhonemeData, ensure_nonword:bool=False):
        self.phoneme_data = phoneme_data
        self.ensure_nonword = ensure_nonword

        self.consonants = phoneme_data.phonemes['consonants']
        self.vowels = phoneme_data.phonemes['vowels']


    def __call__(self):
        # IPA format
        cvcv_ipa_list = self.generate_cvcv_words() 
        
        # Alphabet Format
        cvcv_alphabet_list = [self.make_alphabet_word(cvcv) for cvcv in cvcv_ipa_list] 

        # Convert to list of strings
        cvcv_ipa_list = [''.join(cvcv) for cvcv in cvcv_ipa_list]
        cvcv_alphabet_list = [''.join(cvcv) for cvcv in cvcv_alphabet_list]

        return list(zip(cvcv_ipa_list, cvcv_alphabet_list))

    def generate_cvcv_words(self) -> List[List[str]]:
        """
        Generate CVCV words in IPA format, using the consonants and vowels defined in IPA_PHONEMES.
        """
        cvcv_list = product(self.consonants, self.vowels, self.consonants, self.vowels)

        result = []
        for cvcv in cvcv_list:
            if self.ensure_nonword:

                filter_condition = (not self._is_exists(cvcv)) and self._is_valid(cvcv)
            else:
                filter_condition = self._is_valid(cvcv)
            
            if filter_condition:
                result.append(list(cvcv))
        print(f'[INFO] Filtered {len(result)} valid CVCV combinations.')
        return result
    
    # Make new column for Alphabet-written words(from IPA)
    def make_alphabet_word(self, cvcv_ipa) -> List[str]:
        c1, v1, c2, v2 = cvcv_ipa
        
        aph_result = {'c1': self.phoneme_data.ipa_to_alphabet[c1],
                'v1': self.phoneme_data.ipa_to_alphabet[v1],
                'c2': self.phoneme_data.ipa_to_alphabet[c2],
                'v2': self.phoneme_data.ipa_to_alphabet[v2]}

        # ===Change c1===
        # ipa: `ɡi`, alphabet: `gi` -> `ghi`
        if c1 + v1 == 'ɡi':
            aph_result['c1'] = 'gh'

        
        # ===Change v1===
        # ipa `ʃow`, alphabet `shoe` -> `show``
        if v1 + c1 == 'ʃow':
            aph_result['v1'] = 'ow'
        # ipa 'dow', alphabet 'dow' -> 'dough'
        elif c1 + v1 == 'dow':
            aph_result['v1'] = 'ough'
        # ipa 'own', alphabet 'own' -> 'oan'
        elif v1 + c2 == 'own':
            aph_result['v1'] = 'oa'
        
        # ===Change c2===
        # ipa: `ɡi`, alphabet: `gi` -> `ggi`
        if c2 + v2 == 'ɡi':
            aph_result['c2'] = 'gg'

        # ===Change v2===
        # ipa 'ðow', alphabet 'thoe' -> 'though'
        if c2 + v2 == 'ðow':
            aph_result['c2'] = '-th'
            aph_result['v2'] = 'ough'
        # ipa 'kow', alphabet 'koe' -> 'o'
        elif c2 + v2 == 'kow':
            aph_result['v2'] = 'o'
        # ipa 'ow', alphabet 'oe' -> 'ow' at the end of word
        elif v2 == 'ow':
            aph_result['v2'] = 'ow'
        # ipa 'ði, alphabet 'thee' -> 'they'
        elif c2 + v2 == 'ði':
            aph_result['v2'] = 'ey'
        # ipa 'ðej', alphabet 'thay' -> 'they'
        elif c2 + v2  == 'ðej':
            aph_result['c2'] = '-th'
            aph_result['v2'] = 'ey'
        # ipa 'ej', alphabet 'ay' -> 'aye' at the end of word
        elif v2 == 'ej':
            aph_result['v2'] = 'aye'
        
        return [aph_result['c1'], aph_result['v1'], aph_result['c2'], aph_result['v2']]
    
    def _is_exists(self, cvcv):
        cvcv = [unicodedata.normalize('NFKC', phoneme) for phoneme in cvcv]
        result = self.phoneme_data.ipa_to_word.get(''.join(cvcv), None)
        if result:
            print(f'[DEBUG] {"".join(cvcv)}: {result}')
        return True if result is not None else False

    def _is_valid(self, cvcv):
        cvcv = [unicodedata.normalize('NFKC', phoneme) for phoneme in cvcv]
        # words starting with "ð", rather than "θ", are less common in English
        if cvcv[0] == "ð":
            return False 
        if cvcv[-1] == unicodedata.normalize("NFKC", "ɑ"):
            # words ending with "ɑ" are less common in English
            return False
        
        return True


def main(args):
    phoneme_data = PhonemeData(args.data_dir)
    generator = CVCVWordGenerator(phoneme_data, args.ensure_nonword)
    words = generator()


    df = pd.DataFrame(words, columns=['ipa', 'alphabet'])
    output_dir = Path(args.data_dir) / 'outputs' 
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.ensure_nonword:
        df.to_csv( output_dir / 'constructed_nonwords.csv',
            index_label='index')
    else:
        df.to_csv( output_dir / 'constructed_words.csv',
    index_label='index')



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate CVCV words based on phonemes.")
    parser.add_argument('--data_dir', type=str, default='/data/constructed_words')
    parser.add_argument('--ensure_nonword', action='store_true')
    args = parser.parse_args()
    main(args)
