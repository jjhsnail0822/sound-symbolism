#!/usr/bin/env python3
# EXAMPLE: python ipa_check.py -l fr
import json
import argparse
import os
from tqdm import tqdm

class IPAChecker:
    def __init__(self, language):
        """
        Initialize the IPA checker
        
        Args:
            language (str): Language code (en/ja/ko/fr)
        """
        self.language = language.lower()
        if self.language not in ['en', 'ja', 'ko', 'fr']:
            raise ValueError("Language must be one of 'en', 'ja', 'ko', 'fr'.")
        
        # Set paths based on language
        self.base_path = "data/processed/nat"
        
        if self.language == 'fr':
            self.input_path = os.path.join(self.base_path, f"{self.language}_ipa_filtered.json")
            self.output_path = os.path.join(self.base_path, f"{self.language}_ipa_second_filtered.json")
        else:
            self.input_path = os.path.join(self.base_path, f"{self.language}_ipa.json")
            self.output_path = os.path.join(self.base_path, f"{self.language}_ipa_verified.json")
        
        # Create checkpoint file path
        self.checkpoint_path = os.path.join(self.base_path, f"{self.language}_ipa_checkpoint.json")
    
    def load_data(self):
        """Load data from JSON file"""
        try:
            # Check if checkpoint exists
            if os.path.exists(self.checkpoint_path):
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Loaded checkpoint with {len(data)} items.")
                return data
            
            # Otherwise load from original file
            with open(self.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded original data with {len(data)} items.")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
    
    def save_data(self, data, is_checkpoint=False):
        """Save data to JSON file"""
        try:
            # Save to checkpoint file if requested
            save_path = self.checkpoint_path if is_checkpoint else self.output_path
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if is_checkpoint:
                print(f"Checkpoint saved with {len(data)} items.")
            else:
                print(f"Data saved to {save_path} with {len(data)} items.")
            
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def check_ipa(self):
        """Check IPA for each word and update certainty"""
        # Load data
        data = self.load_data()
        if not data:
            print("No data to process.")
            return False
        
        # Count items already processed
        processed_count = sum(1 for item in data if 'ipa_certainty' in item)
        print(f"{processed_count} items already have ipa_certainty.")
        
        # Process items
        try:
            for i, item in enumerate(data[processed_count:], processed_count):
                word = item.get('word', '')
                ipa = item.get('ipa', '')
                
                # Skip items without IPA
                if not ipa:
                    print(f"\n[{i+1}/{len(data)}] Word: {word} - No IPA available")
                    item['ipa_certainty'] = 0
                    continue
                
                # Display word and IPA
                print(f"\n[{i+1}/{len(data)}] Word: {word}")
                print(f"IPA: {ipa}")
                
                # Get user input for certainty
                while True:
                    certainty = input("Is this IPA correct? (1: Yes, 0: No): ")
                    if certainty in ['0', '1']:
                        break
                    print("Invalid input. Please enter 0 or 1.")
                
                # Update item with certainty
                item['ipa_certainty'] = int(certainty)
                
                # Save checkpoint after each update
                self.save_data(data, is_checkpoint=True)
                
                # Also save to final output file
                self.save_data(data, is_checkpoint=False)
            
            print("\nAll items processed successfully!")
            return True
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            print(f"Progress saved. Processed {processed_count} out of {len(data)} items.")
            return False
        except Exception as e:
            print(f"\nError during processing: {e}")
            print(f"Progress saved. Processed {processed_count} out of {len(data)} items.")
            return False
    
    def run(self):
        """Run the IPA checking process"""
        print(f"Starting IPA checking for language: {self.language}")
        success = self.check_ipa()
        
        if success:
            print("IPA checking completed successfully.")
        else:
            print("IPA checking was not completed. You can resume later.")
        
        return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IPA Checking Tool')
    parser.add_argument('--language', '-l', required=True, choices=['en', 'fr', 'ko', 'ja'], 
                        help='Language code (en/fr/ko/ja)')
    
    args = parser.parse_args()
    
    checker = IPAChecker(
        language=args.language,
    )
    
    checker.run() 