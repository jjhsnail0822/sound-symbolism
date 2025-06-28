import time
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import epitran
import re

class PhonemeDistributionAnalyzer:
    def __init__(self,
             model_path:str,
             language_code:str,
             top_k:int=3,
             temperature:float=0.0,
             ):
        self.model_path = model_path
        self.language_code = language_code
        self.top_k = top_k
        self.temperature = temperature

        self.model = LLM(model=self.model_path)    
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.epi = epitran.Epitran(language_code)
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=1, 
            top_k=self.top_k,
            logprobs=self.top_k,
            prompt_logprobs=0
        )
    
    def run(self, base_prompt:str, max_token: int = 10):
        base_prompt_encoded = self.tokenizer.encode(base_prompt)
        
        def explore_paths(current_token_ids:List[int], current_logprob:float, depth:int):
            completed_token_ids = []

            if depth >= max_token:
                completed_token_ids.append((current_token_ids, current_logprob))
                return completed_token_ids
            
            prompt_encoded = base_prompt_encoded + current_token_ids
            outputs = self._collect_topk_logprobs(prompt_encoded)

            
            for output in outputs:
                token_logprob = output['logprob']
                token_id = output['token_id']
                next_prob = current_logprob + token_logprob   # log prob 로 덧셈!
            

                if self._is_eos_token(token_id):
                    completed_token_ids.append((current_token_ids, current_logprob))
                else:
                    next_token_ids = current_token_ids + [token_id]
                    next_completed_token_ids = explore_paths(next_token_ids, next_prob, depth + 1)
                    completed_token_ids.extend(next_completed_token_ids)
            
            return completed_token_ids
        
        tic = time.time()
        all_token_ids = explore_paths([], 1.0, 0)
        toc = time.time()
        print(f"[INFO] Explore Words: {toc - tic}s")
        
        # Remove duplicates and sum probabilities
        word_probs = defaultdict(float)
        for token_ids, logprob in all_token_ids:
            word = self.tokenizer.decode(token_ids)
            # print(f"Generated word: '{word}'")
            prob = np.exp(logprob)
            word_probs[word] += prob

        all_combinations = []
        for word, total_prob in word_probs.items():
            # Word cleaning and validation
            cleaned_word = self._clean_word(word)
            if cleaned_word:
                ipa = self._convert_to_ipa(cleaned_word)
                if ipa:
                    all_combinations.append((cleaned_word, ipa, total_prob))
                    # print(f"Valid word: '{cleaned_word}' -> IPA: '{ipa}'")
        
        if all_combinations:
            phoneme_distribution = self._calculate_phoneme_distribution(all_combinations)
            print(f"[INFO] EOS-based Brute Force completed: {len(all_combinations)} words")
        else:
            phoneme_distribution = {}
            print("[WARNING] No valid words generated.")

        return all_combinations, phoneme_distribution

    def _clean_word(self, word: str) -> str:
        """Extract actual word from generated text"""
        # Remove backticks
        word = re.sub(r'`+', '', word)
        # Remove leading and trailing whitespace
        word = word.strip()
        # Remove special characters (keep only letters)
        if 'eng' in self.language_code:
            word = re.sub(r'[^a-zA-Z]', '', word)
        else:
            raise NotImplementedError
        return word.lower()


    def _is_eos_token(self, token_id):
        return (token_id == self.tokenizer.eos_token_id)

    def _collect_topk_logprobs(self, prompt_token_ids:List[str] ):
        response = self.model.generate(
            prompt_token_ids = prompt_token_ids,
            sampling_params=self.sampling_params)
        result = []
        for rp in response:
            if rp.outputs and rp.outputs[0].logprobs:
                token_logprobs=rp.outputs[0].logprobs[0]
                for token_id, logprob_data in token_logprobs.items():
                    token_text = logprob_data.decoded_token
                    logprob= logprob_data.logprob
                    prob=np.exp(logprob)

                    result.append({
                        'token_id':token_id,
                        'token_text':token_text,
                        'logprob':logprob,
                        'probability': prob
                    })

        return result

    def _convert_to_ipa(self, word: str) -> str:
        """Safe IPA conversion"""
        try:
            word = word.strip().lower()
            if not word or not word.isalpha():
                return ""
            
            ipa = self.epi.transliterate(word)
            return ipa
        except (IndexError, ValueError, AttributeError) as e:
            print(f"[WARNING] IPA conversion failed for '{word}': {e}")
            return ""
        except Exception as e:
            print(f"[ERROR] Unexpected error for '{word}': {e}")
            return ""

    def _calculate_phoneme_distribution(self, combinations: List[Tuple]) -> Dict[str, float]:
        """Calculate weighted phoneme distribution from combinations"""
        phoneme_weights = defaultdict(float)
        total_weight = sum(combo[2] for combo in combinations)  # Sum of all probabilities
        
        for word, ipa, prob in combinations:
            normalized_prob = prob / total_weight if total_weight > 0 else 0
            for phoneme in ipa:
                phoneme_weights[phoneme] += normalized_prob
        
        return dict(phoneme_weights)


def main(): 
    meaning = "Used as an emphatic expression, chiefly of disgust, but also (formerly) of surprise or approval."
    prompt = f"""Create a single invented word that represents the sound of a sneeze. 
The word should:
- Be 3-6 letters long
- Use only Roman letters (a-z)
- Sound like a sneeze when pronounced
- Be enclosed in backticks

Example format: `achoo`

Create the word for: {meaning}

Word: """
  
    analyzer = PhonemeDistributionAnalyzer(
        model_path='Qwen/Qwen3-4B', 
        language_code="eng-Latn",
        top_k=3,  
        temperature=0.0, 
    )
    
    combinations, phoneme_dist = analyzer.run(base_prompt=prompt, max_token=6)
    print("\n=== Final Results ===")
    print("Generated words:")
    for i, (word, ipa, prob) in enumerate(combinations):
        print(f"{i}. {word} [{ipa}] (probability: {prob:.4f})")
    
    print("\nPhoneme distribution:")
    for phoneme, weight in sorted(phoneme_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {phoneme}: {weight:.4f}")

if __name__ == '__main__':
    main()