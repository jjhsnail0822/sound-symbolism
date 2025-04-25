#!/usr/bin/env python3
# EXAMPLE: python filter_same_pronunciations.py
import json
import os
import itertools
from collections import defaultdict

class PronunciationFilter:
    def __init__(self):
        """Initialize the pronunciation filter"""
        # Set paths
        self.base_path = "/scratch2/sheepswool/workspace/sound-symbolism/dataset/1_preprocess/nat"
        self.input_path = os.path.join(self.base_path, "fr_ipa_filtered.json")
        self.output_path = os.path.join(self.base_path, "fr_ipa_second_filtered.json")
        self.checkpoint_path = os.path.join(self.base_path, "fr_ipa_pronunciation_checkpoint.json")
        
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
    
    def sort_by_word(self, data):
        """Sort data by word alphabetically"""
        return sorted(data, key=lambda x: x.get('word', '').lower())
    
    def find_same_pronunciations(self, data):
        """Find words with the same IPA pronunciation"""
        # Group words by IPA
        ipa_groups = defaultdict(list)
        for i, item in enumerate(data):
            ipa = item.get('ipa', '').strip()
            if ipa:  # Only consider items with IPA
                ipa_groups[ipa].append((i, item))
        
        # Filter groups with more than one word
        same_pronunciations = {ipa: items for ipa, items in ipa_groups.items() if len(items) > 1}
        
        return same_pronunciations
    
    def process_same_pronunciations(self, data, same_pronunciations):
        """Process words with the same pronunciation and let user choose which to keep"""
        # Create a set to track indices to remove
        indices_to_remove = set()
        
        # Process each group of words with the same pronunciation
        for ipa, items in same_pronunciations.items():
            print(f"\n{'='*80}")
            print(f"Found {len(items)} words with the same pronunciation: {ipa}")
            print(f"{'='*80}")
            
            # Display each word with its details
            for i, (idx, item) in enumerate(items, 1):
                word = item.get('word', '')
                meanings = item.get('meaning', [])
                
                print(f"\n[{i}] Word: {word}")
                print(f"    IPA: {ipa}")
                print(f"    Meanings:")
                for j, meaning in enumerate(meanings, 1):
                    print(f"      {j}. {meaning}")
            
            # Ask user which words to keep
            while True:
                try:
                    choices = input("\nEnter the numbers of words to keep (space-separated, e.g., '1 3'): ")
                    selected_indices = [int(x) - 1 for x in choices.split() if x.strip()]
                    
                    # Validate choices
                    if not selected_indices:
                        print("You must select at least one word to keep.")
                        continue
                    
                    if any(idx < 0 or idx >= len(items) for idx in selected_indices):
                        print(f"Invalid selection. Please enter numbers between 1 and {len(items)}.")
                        continue
                    
                    break
                except ValueError:
                    print("Invalid input. Please enter numbers separated by spaces.")
            
            # Mark indices to remove
            for i, (idx, _) in enumerate(items):
                if i not in selected_indices:
                    indices_to_remove.add(idx)
            
            print(f"Selected to keep: {', '.join(items[i][1].get('word', '') for i in selected_indices)}")
        
        # Create new data without the removed items
        filtered_data = [item for i, item in enumerate(data) if i not in indices_to_remove]
        
        print(f"\nRemoved {len(indices_to_remove)} duplicate pronunciation items.")
        print(f"Original count: {len(data)}, New count: {len(filtered_data)}")
        
        return filtered_data
    
    def run(self):
        """Run the pronunciation filter process"""
        print("Starting pronunciation filter for French words...")
        
        # Load data
        data = self.load_data()
        if not data:
            print("No data to process.")
            return False
        
        # Sort data by word alphabetically
        print("Sorting data by word alphabetically...")
        sorted_data = self.sort_by_word(data)
        
        # Save sorted data
        self.save_data(sorted_data)
        print("Sorted data saved.")
        
        # Find words with the same pronunciation
        print("Finding words with the same pronunciation...")
        same_pronunciations = self.find_same_pronunciations(sorted_data)
        
        if not same_pronunciations:
            print("No words with the same pronunciation found.")
            return True
        
        print(f"Found {len(same_pronunciations)} groups of words with the same pronunciation.")
        
        # Process words with the same pronunciation
        filtered_data = self.process_same_pronunciations(sorted_data, same_pronunciations)
        
        # Save filtered data
        self.save_data(filtered_data)
        print("Filtered data saved.")
        
        return True

if __name__ == "__main__":
    filter_tool = PronunciationFilter()
    success = filter_tool.run()
    
    if success:
        print("Pronunciation filtering completed successfully.")
    else:
        print("Error occurred during pronunciation filtering.") 