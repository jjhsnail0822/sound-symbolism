import os
import json
import unicodedata
import argparse 
from typing import List
from itertools import product

from phonemes import IPA_PHONEMES, IPA_TO_ALPHABET

class CVCVWordGenerator:
    def __init__(self, data_dir:str, ensure_nonword:bool=False):
        self.data_dir = data_dir
        self.ensure_nonword = ensure_nonword

        self.consonants = IPA_PHONEMES['consonants']
        self.vowels = IPA_PHONEMES['vowels']
        self.ipa_to_word = self._load_ipa_to_word()
    
    def _load_ipa_to_word(self):
        ipa_to_word_path = os.path.join(self.data_dir, 'ipa_to_word_en_US.json')
        with open(ipa_to_word_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __call__(self):
        # IPA format
        cvcv_ipa_list = self.generate_cvcv_words() 
        
        # Alphabet Format
        cvcv_alphabet_list = [self.make_alphabet_word(cvcv) for cvcv in cvcv_ipa_list] 

        # Convert to list of strings
        cvcv_ipa_list = [''.join(cvcv) for cvcv in cvcv_ipa_list]
        cvcv_alphabet_list = [''.join(cvcv) for cvcv in cvcv_alphabet_list]


        # Save the constructed words to a file
        self.save_to_file(cvcv_ipa_list, cvcv_alphabet_list)

        return zip(cvcv_ipa_list, cvcv_alphabet_list)



    def save_to_file(self, cvcv_ipa_list: List[str], cvcv_alphabet_list: List[str]):
        """
        Save the generated CVCV words in both IPA and Alphabet format to a CSV file.
        """
        fname = 'constructed_words_ensure_nonword.csv' if self.ensure_nonword else 'constructed_words.csv' 
        output_path = os.path.join(self.data_dir, fname)

        with open(output_path, 'w') as f:
            f.write("index,ipa,alphabet\n")
        
        with open(output_path, 'a') as f:
            for i, (ipa, alphabet) in enumerate(zip(cvcv_ipa_list, cvcv_alphabet_list)):
                f.write(f"{i},{ipa},{alphabet}\n")

        
    def generate_cvcv_words(self) -> List[List[str]]:
        """
        Generate CVCV words in IPA format, using the consonants and vowels defined in IPA_PHONEMES.
        """
        cvcv_list = product(self.consonants, self.vowels, self.consonants, self.vowels)

        result = []
        for cvcv in cvcv_list:
            if self.ensure_nonword:
                condition1 = (not self._is_exists(cvcv))
                condition2 = self._is_valid(cvcv)
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
        
        aph_result = {'c1': IPA_TO_ALPHABET[c1],
                  'v1': IPA_TO_ALPHABET[v1],
                  'c2': IPA_TO_ALPHABET[c2],
                  'v2': IPA_TO_ALPHABET[v2]}
        
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
        result = self.ipa_to_word.get(''.join(cvcv), None)
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
    generator = CVCVWordGenerator(args.data_dir, args.ensure_nonword)
    generator()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Generate CVCV words based on phonemes.")
    parser.add_argument('--data_dir', type=str, default='/data/constructed_words')
    parser.add_argument('--ensure_nonword', action='store_true')
    args = parser.parse_args()
    main(args)
