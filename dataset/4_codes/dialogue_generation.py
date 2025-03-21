import os
import json
import argparse
import time
from tqdm import tqdm
import openai
import pandas as pd
import random
from typing import List, Dict, Any, Optional

# Bring api key from .env.local
api_key = os.getenv("OPENAI_API_KEY")

class DialogueGenerator:
    def __init__(self, language: str, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the dialogue generator
        
        Args:
            language (str): Language code (en/fr/ko/ja)
            api_key (str, optional): OpenAI API key. If None, will try to get from environment variable
            model (str): OpenAI model to use
        """
        self.language = language.lower()
        if self.language not in ['en', 'fr', 'ko', 'ja']:
            raise ValueError("Language must be one of 'en', 'fr', 'ko', 'ja'.")
        
        # Set paths
        self.input_path = os.path.join('../1_preprocess/nat', f"{self.language}.json")
        self.output_path = os.path.join('../2_dialogue/nat', f"{self.language}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Set OpenAI API key
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Set model
        self.model = model
        
        # Load language-specific templates
        self.templates = self._get_language_templates()
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter data to only include entries with meanings
            filtered_data = [item for item in data if item.get('meaning') and len(item.get('meaning', [])) > 0 and item.get('found', False)]
            
            print(f"Loaded {len(filtered_data)} entries with meanings out of {len(data)} total entries")
            return filtered_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def _get_language_templates(self) -> Dict[str, Dict[str, str]]:
        """Get language-specific templates for prompts"""
        templates = {
            'en': {
                'system_prompt': "You are a helpful assistant that creates natural dialogues between two people. The dialogue should include the onomatopoeic word '{word}' which means '{meaning}'. Make the dialogue sound natural and contextually appropriate.",
                'user_prompt': "Create a dialogue between two people that naturally incorporates the onomatopoeic word '{word}' which means '{meaning}'. The dialogue should be 5-8 exchanges long and feel natural. Make sure to highlight the onomatopoeic word in ALL CAPS when it appears in the dialogue.",
                'dialogue_format': "Person A: {}\nPerson B: {}"
            },
            'fr': {
                'system_prompt': "Vous êtes un assistant utile qui crée des dialogues naturels entre deux personnes. Le dialogue doit inclure le mot onomatopéique '{word}' qui signifie '{meaning}'. Faites en sorte que le dialogue semble naturel et approprié au contexte.",
                'user_prompt': "Créez un dialogue entre deux personnes qui incorpore naturellement le mot onomatopéique '{word}' qui signifie '{meaning}'. Le dialogue doit comporter 5 à 8 échanges et sembler naturel. Assurez-vous de mettre en évidence le mot onomatopéique en MAJUSCULES lorsqu'il apparaît dans le dialogue.",
                'dialogue_format': "Personne A: {}\nPersonne B: {}"
            },
            'ko': {
                'system_prompt': "당신은 두 사람 사이의 자연스러운 대화를 만드는 도우미입니다. 대화에는 '{meaning}'을(를) 의미하는 의성어/의태어 '{word}'가 포함되어야 합니다. 대화가 자연스럽고 문맥에 적절하게 들리도록 만드세요.",
                'user_prompt': "조건1: '{meaning}'을(를) 의미하는 의성어/의태어 '{word}'를 자연스럽게 포함하는 두 사람 사이의 대화를 만들어주세요.\n \
                    조건2: 대화는 {}번의 교환으로 이루어져야 하며 자연스럽게 느껴져야 합니다.\n \
                    조건3: 대화 내의 의성어/의태어는 <> 속에 입력해주세요.\n \
                    조건4: 의성어/의태어는 {}번째 발화에 나타나야 합니다.\n \
                    조건5: 대화 내에 포함되는 의성어/의태어를 직접적으로 유추할 수 있는 단어를 제외하세요.",
                'dialogue_format': "A: {}\nB: {}"
            },
            'ja': {
                'system_prompt': "あなたは二人の間で自然な対話を作成する役立つアシスタントです。対話には「{meaning}」を意味するオノマトペ「{word}」を含める必要があります。対話が自然で文脈に適切に聞こえるようにしてください。",
                'user_prompt': "「{meaning}」を意味するオノマトペ「{word}」を自然に取り入れた二人の間の対話を作成してください。対話は5〜8回のやり取りで構成され、自然に感じられるようにしてください。オノマトペが対話に現れるときは、大文字で強調してください。",
                'dialogue_format': "A: {}\nB: {}"
            }
        }
        return templates
    
    def generate_dialogue(self, word_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dialogue for a given word"""
        word = word_data.get('word', '')
        
        # Get the first meaning if it's a list, or use the meaning directly
        meaning = word_data.get('meaning', [])
        if isinstance(meaning, list) and meaning:
            meaning = meaning[0]
        elif not meaning:
            meaning = "an onomatopoeic word"
        
        # Format prompts
        system_prompt = self.templates[self.language]['system_prompt'].format(word=word, meaning=meaning)
        user_prompt = self.templates[self.language]['user_prompt'].format(word=word, meaning=meaning)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract dialogue from response
            dialogue = response.choices[0].message.content.strip()
            
            # Create result dictionary
            result = {
                'word': word,
                'meaning': meaning,
                'dialogue': dialogue,
                'language': self.language
            }
            
            return result
        except Exception as e:
            print(f"Error generating dialogue for word '{word}': {e}")
            # Return empty dialogue on error
            return {
                'word': word,
                'meaning': meaning,
                'dialogue': "",
                'language': self.language,
                'error': str(e)
            }
    
    def generate_all_dialogues(self, limit: Optional[int] = None, shuffle: bool = True) -> List[Dict[str, Any]]:
        """Generate dialogues for all words in the data"""
        if not self.data:
            print("No data available to generate dialogues")
            return []
        
        # Optionally shuffle the data
        data_to_process = self.data.copy()
        if shuffle:
            random.shuffle(data_to_process)
        
        # Limit the number of items to process if specified
        if limit and limit > 0:
            data_to_process = data_to_process[:limit]
        
        results = []
        
        # Process each word with progress bar
        for item in tqdm(data_to_process, desc=f"Generating {self.language} dialogues"):
            # Generate dialogue
            result = self.generate_dialogue(item)
            
            # Save individual dialogue to file
            self._save_dialogue_to_file(result)
            
            # Add to results list
            results.append(result)
            
            # Add a delay to avoid rate limiting
            time.sleep(1)
        
        return results
    
    def _save_dialogue_to_file(self, dialogue_data: Dict[str, Any]) -> None:
        """Save a single dialogue to a JSON file"""
        word = dialogue_data.get('word', 'unknown')
        # Replace characters that might be problematic in filenames
        safe_word = ''.join(c if c.isalnum() else '_' for c in word)
        
        # Create filename
        filename = os.path.join(self.output_path, f"{safe_word}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dialogue_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving dialogue for word '{word}' to file: {e}")
    
    def save_all_dialogues(self, dialogues: List[Dict[str, Any]]) -> None:
        """Save all dialogues to a single JSON file"""
        if not dialogues:
            print("No dialogues to save")
            return
        
        # Create filename for all dialogues
        filename = os.path.join(self.output_path, f"all_dialogues_{self.language}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dialogues, f, ensure_ascii=False, indent=4)
            print(f"All dialogues saved to {filename}")
        except Exception as e:
            print(f"Error saving all dialogues to file: {e}")
    
    def run(self, limit: Optional[int] = None, shuffle: bool = True) -> None:
        """Run the dialogue generation process"""
        print(f"Starting dialogue generation for language: {self.language}")
        
        # Generate dialogues
        dialogues = self.generate_all_dialogues(limit=limit, shuffle=shuffle)
        
        # Save all dialogues to a single file
        self.save_all_dialogues(dialogues)
        
        print(f"Dialogue generation completed. Generated {len(dialogues)} dialogues.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialogue Generation Tool')
    parser.add_argument('--language', '-l', required=True, choices=['en', 'fr', 'ko', 'ja'], 
                        help='Language code (en/fr/ko/ja)')
    parser.add_argument('--api-key', '-k', help='OpenAI API key (optional, can use environment variable)')
    parser.add_argument('--model', '-m', default="gpt-3.5-turbo", 
                        help='OpenAI model to use (default: gpt-3.5-turbo)')
    parser.add_argument('--limit', '-n', type=int, help='Limit the number of dialogues to generate')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle the data before processing')
    
    args = parser.parse_args()
    
    generator = DialogueGenerator(
        language=args.language,
        api_key=args.api_key,
        model=args.model
    )
    
    generator.run(limit=args.limit, shuffle=not args.no_shuffle) 