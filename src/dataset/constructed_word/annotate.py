import os
import json
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from phonemes import IPA_TO_FEATURE

parser = ArgumentParser(description="Construct words based on phonetic features.")
parser.add_argument('--data_dir', type=str, default='sound-symbolism/data/constructed_words')
parser.add_argument('--csv_path', type=str, help='Path to `consturcted_words.csv` or `constructed_words_ensure_nonwords.csv` file.')

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def main(args):
    fname = os.path.basename(args.csv_path).replace('.csv', '')
    df = pd.read_csv(args.csv_path)

    data_dir = Path(args.data_dir)
    
    feature_to_score = load_json(data_dir / 'feature_to_score.json')

    def get_score(ipa, dimension):
        score = 0
        for char in ipa:
            feature = IPA_TO_FEATURE.get(char, None)
            if feature == 'back':
                score -= feature_to_score.get('front', 0)
            else: 
                score += feature_to_score[dimension].get(feature, 0)
        
        return score / len(ipa) if ipa else 0
    

    for dimension in feature_to_score.keys():
        df['score_'+ dimension.lower()] =  df['ipa'].apply(lambda x: get_score(x, dimension), by_row='compat')


    df.to_csv(data_dir / f'{fname}_w_scores.csv', index=False)
    print(f"[INFO] Scores calculated and saved to '{data_dir / 'constructed_words_with_scores.csv'}'.")


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
