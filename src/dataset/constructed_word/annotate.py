import json
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from phonemes import IPA_MAP, IPA_TO_FEATURE

parser = ArgumentParser(description="Construct words based on phonetic features.")
parser.add_argument('--data_dir', type=str, default='data/constructed_words')

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def main(args):
    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / 'constructed_words.csv')

    
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


    df.to_csv(data_dir / 'constructed_words_with_scores.csv', index=False)
    print(f"[INFO] Scores calculated and saved to '{data_dir / 'constructed_words_with_scores.csv'}'.")


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
