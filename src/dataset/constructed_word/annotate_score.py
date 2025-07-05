import json
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from phoneme_data import PhonemeData

parser = ArgumentParser(description="Construct words based on phonetic features.")
parser.add_argument('--data_dir', type=str, default='/sound-symbolism/data/processed/art/resources')
parser.add_argument('--output_dir', type=str, default='/sound-symbolism/data/processed/art/semantic_dimension')
parser.add_argument('--json_path', type=str, help='Path to `semantic_dimension_binary_gt.json`')
parser.add_argument('--threshold', type=float, default=0.171)

class ScoreAnnotator:
    def __init__(self, phoneme_data:PhonemeData, json_path:str):
        self.phoneme_data = phoneme_data
        self.json_path = Path(json_path)

    def annotate(self, threshold=0.171):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cnt = 0
        for word in data['art']:
            score_dict = {}
            for dimension in self.phoneme_data.feature_to_score:
                dim1, dim2 = dimension.lower().split('-')
                score = self.get_score(word['ipa'], dimension)
                if score > threshold:
                    answer = dim2
                    cnt += 1
                elif score < -threshold:
                    answer = dim1
                    cnt +=1
                else:
                    continue 

                score_dict[dimension.lower()] = {
                    "answer": answer,
                    "score": score
                }
            word.update({'dimensions': score_dict})
        print(f"[INFO] Meaningful Dimension Data Count: {cnt}/{25*len(data['art'])}")

        return data

    def get_score(self, ipa, dimension):
        score = 0
        phonemes = ipa.split(' ')
        for char in phonemes:
            feature = self.phoneme_data.ipa_to_feature[char]
            if feature == 'back':
                score -= self.phoneme_data.feature_to_score[dimension]['front']
            else: 
                score += self.phoneme_data.feature_to_score[dimension][feature]
        
        return score / len(phonemes) if phonemes else 0

def main(args):
    phoneme_data = PhonemeData(args.data_dir)
    annotator = ScoreAnnotator(phoneme_data, args.json_path)
    result = annotator.annotate(threshold=args.threshold)

    

    
    with open(args.json_path,'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print("[INFO] Saved to", args.json_path)

    
if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
