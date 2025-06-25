import os
import json
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from phoneme_data import PhonemeData

parser = ArgumentParser(description="Construct words based on phonetic features.")
parser.add_argument('--data_dir', type=str, default='sound-symbolism/data/constructed_words')
parser.add_argument('--csv_path', type=str, help='Path to `consturcted_words.csv` or `constructed_nonwords.csv` file.')

class ScoreAnnotator:
    def __init__(self, phoneme_data:PhonemeData, csv_path:str):
        self.phoneme_data = phoneme_data
        self.csv_path = Path(csv_path)

    def __call__(self):
        df = pd.read_csv(self.csv_path)
        for dimension in self.phoneme_data.feature_to_score.keys():
            column_name = 'score_' + dimension.lower()
            df[column_name] =  df['ipa'].apply(lambda x: self.get_score(x, dimension))

        return df

    def get_score(self, ipa, dimension):
        score = 0
        for char in ipa:
            feature = self.phoneme_data.ipa_to_feature.get(char, None)
            if feature == 'back':
                score -= self.phoneme_data.feature_to_score.get('front', 0)
            else: 
                score += self.phoneme_data.feature_to_score[dimension].get(feature, 0)
        
        return score / len(ipa) if ipa else 0

def main(args):
    phoneme_data = PhonemeData(args.data_dir)
    annotator = ScoreAnnotator(phoneme_data, args.csv_path)
    result_df = annotator()

    fname = Path(args.csv_path).stem
    ouput_dir = Path(args.data_dir) / 'outputs'
    ouput_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(ouput_dir / f'{fname}_annotated.csv', index=False)
    
    
if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
