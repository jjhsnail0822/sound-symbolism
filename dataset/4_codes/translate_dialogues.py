import os
from pathlib import Path
from dotenv import load_dotenv
import json
import time
import re
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from tqdm import tqdm
import argparse

env_path = Path('.env.local')
load_dotenv(dotenv_path=env_path)

# Get API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)  # Added timeout to client

MODEL_NAME = "gpt-4.1"
INPUT_DIR = "dataset/2_dialogue/nat"
OUTPUT_DIR = "dataset/2_dialogue/nat"

# Define languages to process and their full names for the prompt
LANGUAGES_TO_PROCESS = {
    "fr": "French",
    "ja": "Japanese",
    "ko": "Korean"
}

# Retry mechanism configuration
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 15  # Increased delay for retries

# --- Prompt Loading ---
try:
    with open('analysis/experiments/prompts.json', 'r', encoding='utf-8') as f:
        prompts = json.load(f)
        if 'dialogue_translation' not in prompts or 'user_prompt' not in prompts['dialogue_translation']:
            raise KeyError("Prompt structure 'dialogue_translation.user_prompt' not found in prompts.json")
        BASE_PROMPT_TEMPLATE = prompts['dialogue_translation']['user_prompt']
except FileNotFoundError:
    raise FileNotFoundError("Error: prompts.json not found. Make sure the path 'analysis/experiments/prompts.json' is correct.")
except json.JSONDecodeError as e:
    raise json.JSONDecodeError(f"Error decoding prompts.json: {e.msg}", e.doc, e.pos)
except KeyError as e:
    raise KeyError(f"Error accessing prompt in prompts.json: {e}")

# --- Translation Function (Handles API call and basic parsing) ---
def translate_dialogue_api_call(dialogue_obj, source_language_name):
    """
    Makes the API call for translation and handles basic response cleaning/parsing.
    Separated to facilitate retries on specific errors.

    Returns:
        dict: Parsed translated dialogue object from API.
    Raises:
        RateLimitError, APIError, APITimeoutError, json.JSONDecodeError, ValueError (for empty response), Exception
    """
    dialogue_json_string = json.dumps(dialogue_obj, ensure_ascii=False, indent=4)
    prompt = BASE_PROMPT_TEMPLATE.format(
        source_language=source_language_name,
        json_dialogue_string=dialogue_json_string
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    translated_json_string = response.choices[0].message.content.strip()

    # Clean potential markdown code blocks
    if translated_json_string.startswith("```json"):
        translated_json_string = translated_json_string[7:]
    if translated_json_string.endswith("```"):
        translated_json_string = translated_json_string[:-3]
    translated_json_string = translated_json_string.strip()

    if not translated_json_string:
        # Raise an error for empty response to trigger retry
        raise ValueError("Received empty JSON string from API.")

    # Parse JSON - can raise json.JSONDecodeError
    translated_dialogue_obj = json.loads(translated_json_string)
    return translated_dialogue_obj

# --- Consistency Check Functions ---
def check_keys(original_utterances, translated_utterances):
    """Checks if keys in translated utterances match the original."""
    if not original_utterances or not translated_utterances:
        return False  # Cannot compare if one is empty
    if len(original_utterances) != len(translated_utterances):
        print("    Consistency Check Failed: Number of utterances mismatch.")
        return False

    original_keys = set(original_utterances[0].keys()) if original_utterances else set()
    for i, trans_utt in enumerate(translated_utterances):
        trans_keys = set(trans_utt.keys())
        if trans_keys != original_keys:
            print(f"    Consistency Check Failed: Key mismatch in utterance {i+1}. Expected {original_keys}, Got {trans_keys}")
            return False
    return True

def check_brackets(translated_utterances, ss_idx):
    """Checks bracket presence only in the target utterance."""
    bracket_pattern = re.compile(r"\[.*?\]")  # Simple check for brackets
    found_bracket_in_target = False
    found_bracket_elsewhere = False

    if not translated_utterances:  # Handle empty list
        print("    Consistency Check Info: Cannot check brackets, translated utterances list is empty.")
        return False

    for utterance in translated_utterances:
        text = utterance.get('text', '')
        has_brackets = bool(bracket_pattern.search(text))
        utterance_idx = utterance.get('index')

        if utterance_idx == ss_idx:
            if has_brackets:
                found_bracket_in_target = True
            else:
                # If brackets are MISSING in the target utterance, it's a failure
                print(f"    Consistency Check Failed: Brackets missing in target utterance (index {ss_idx}). Text: '{text}'")
                return False
        elif has_brackets:
            # If brackets are PRESENT in a non-target utterance, it's a failure
            found_bracket_elsewhere = True
            print(f"    Consistency Check Failed: Unexpected brackets found in non-target utterance (index {utterance_idx}). Text: '{text}'")
            return False  # Fail fast

    if not found_bracket_in_target:
        # This case is covered above, but as a safeguard
        print(f"    Consistency Check Failed: Brackets missing in target utterance (index {ss_idx}) after checking all.")
        return False

    # If we reach here, brackets were found ONLY in the target utterance
    return True

# --- Refactored Translation and Validation Function ---
def translate_and_validate_dialogue(
    original_dialogue_entry: dict,
    subject_word: str,
    source_language_name: str,
    original_utterances_for_check: list  # Pass the clean original utterances for key check
) -> Optional[list]:  # Returns list of translated utterances or None
    """
    Preprocesses, translates (with retries), and validates a single dialogue entry.

    Returns:
        The translated and validated dialogue list (value for the 'dialogue' key),
        or None if translation/validation fails after retries.
    """
    if 'dialogue' not in original_dialogue_entry or not isinstance(original_dialogue_entry['dialogue'], list):
        print(f"    Skipping dialogue: 'dialogue' key missing or not a list.")
        return None
    if 'meta_data' not in original_dialogue_entry or 'ss_idx' not in original_dialogue_entry['meta_data']:
        print(f"    Skipping dialogue: 'meta_data' or 'ss_idx' missing.")
        return None

    ss_idx = original_dialogue_entry['meta_data']['ss_idx']
    preprocessed_dialogue_list = []

    # --- Preprocessing Step ---
    try:
        # Use the passed original_dialogue_entry for preprocessing
        original_utterances = json.loads(json.dumps(original_dialogue_entry['dialogue']))  # Deep copy for modification
        for utterance in original_utterances:
            if utterance.get('index') == ss_idx:
                original_text = utterance.get('text', '')
                # Use word boundary regex
                pattern = r'\b' + re.escape(subject_word) + r'\b'
                replacement = '[' + subject_word + ']'
                replaced_text, num_replacements = re.subn(pattern, replacement, original_text, flags=re.IGNORECASE)
                # Fallback only if regex didn't replace AND word is present
                if num_replacements == 0 and subject_word.lower() in original_text.lower():
                    # Be cautious with simple replace, maybe log this occurrence
                    print(f"      Preprocessing Fallback: Regex failed for '{subject_word}' in '{original_text}'. Using simple replace.")
                    replaced_text = original_text.replace(subject_word, replacement)
                utterance['text'] = replaced_text
            preprocessed_dialogue_list.append(utterance)
    except Exception as preproc_e:
        print(f"    Error during preprocessing dialogue: {preproc_e}. Skipping.")
        return None  # Indicate failure

    # --- Translation with Retry Logic ---
    translated_utterances = None  # Initialize
    for attempt in range(MAX_RETRIES):
        dialogue_to_translate = {"dialogue": preprocessed_dialogue_list}
        try:
            # Make the API call
            translated_dialogue_part = translate_dialogue_api_call(dialogue_to_translate, source_language_name)

            # --- Consistency Checks ---
            if not translated_dialogue_part or 'dialogue' not in translated_dialogue_part:
                print("    Consistency Check Failed: API response missing 'dialogue' key.")
                raise ValueError("Invalid structure in translated response")

            current_translated_utterances = translated_dialogue_part['dialogue']

            # 1. Key Check
            if not check_keys(original_utterances_for_check, current_translated_utterances):
                raise ValueError("Consistency Check Failed: Keys mismatch.")

            # 2. Bracket Check
            if not check_brackets(current_translated_utterances, ss_idx):
                raise ValueError("Consistency Check Failed: Brackets mismatch.")

            # If all checks pass
            translated_utterances = current_translated_utterances  # Store successful result
            break  # Exit retry loop on success

        except RateLimitError as e:
            print(f"      Rate limit reached (Attempt {attempt + 1}). Waiting 60s. Error: {e}")
            time.sleep(60)
        except (APIError, APITimeoutError) as e:
            print(f"      API Error/Timeout (Attempt {attempt + 1}). Waiting {RETRY_DELAY_SECONDS}s. Error: {e}")
            time.sleep(RETRY_DELAY_SECONDS)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"      Parsing/Consistency/Empty Response Error (Attempt {attempt + 1}). Waiting {RETRY_DELAY_SECONDS}s. Error: {e}")
            time.sleep(RETRY_DELAY_SECONDS)
        except Exception as e:
            print(f"      An unexpected error occurred (Attempt {attempt + 1}). Waiting {RETRY_DELAY_SECONDS}s. Error: {e}")
            time.sleep(RETRY_DELAY_SECONDS)

    # --- After Retry Loop ---
    if translated_utterances is None:
        return None  # Indicate failure
    else:
        return translated_utterances  # Return the successfully translated list

# --- Main Processing Function ---
def main(skip_phase_1: bool):  # Add skip_phase_1 parameter
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_errors = 0
    global_translation_failures = 0  # Count dialogues that failed initial translation
    global_retranslation_failures = 0  # Count dialogues that failed re-translation

    # === Phase 1: Initial Translation ===
    if not skip_phase_1:
        print("--- Phase 1: Initial Translation ---")
        for lang_code, lang_name in LANGUAGES_TO_PROCESS.items():
            input_filepath = os.path.join(INPUT_DIR, f"{lang_code}.json")
            output_filename = f"{lang_code}2en.json"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)

            print(f"\n--- Processing language: {lang_name} ({lang_code}.json) ---")

            if not os.path.exists(input_filepath):
                print(f"  Input file not found: {input_filepath}. Skipping.")
                continue

            try:
                with open(input_filepath, 'r', encoding='utf-8') as f:
                    source_data = json.load(f)
                if not isinstance(source_data, list):
                    print(f"  Error: Expected a list in {input_filepath}, but got {type(source_data)}. Skipping.")
                    continue
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Error reading or parsing source JSON file {input_filepath}: {e}. Skipping language.")
                total_errors += 1
                continue

            current_lang_translated_data = []
            language_had_errors = False

            # Use tqdm for iterating through source data items
            for i, item in enumerate(tqdm(source_data, desc=f"Translating {lang_name}", unit="item")):
                subject_word = item.get('word')
                if not subject_word:
                    continue

                if 'dialogues' not in item or not isinstance(item['dialogues'], list):
                    continue

                translated_dialogues_for_item = []
                item_translation_failed = False  # Tracks if any dialogue within the item failed permanently

                for j, dialogue_entry in enumerate(item['dialogues']):
                    # Keep a clean copy of original utterances for key check
                    try:
                        original_utterances_for_check = json.loads(json.dumps(dialogue_entry.get('dialogue', [])))
                    except Exception as copy_e:
                        print(f"    Error copying original utterances for dialogue {j+1} in item {i+1}: {copy_e}. Skipping dialogue.")
                        item_translation_failed = True
                        language_had_errors = True
                        total_errors += 1
                        continue

                    # Call the refactored translation function
                    translated_utterances = translate_and_validate_dialogue(
                        dialogue_entry, subject_word, lang_name, original_utterances_for_check
                    )

                    if translated_utterances is not None:
                        # Reconstruct the full dialogue entry if translation succeeded
                        new_dialogue_entry = dialogue_entry.copy()
                        new_dialogue_entry['dialogue'] = translated_utterances
                        translated_dialogues_for_item.append(new_dialogue_entry)
                    else:
                        # If translation failed for this dialogue
                        item_translation_failed = True  # Mark item as failed
                        language_had_errors = True
                        global_translation_failures += 1  # Increment global failure count

                # --- After processing all dialogues for an item ---
                if not item_translation_failed:
                    translated_item = item.copy()
                    translated_item['dialogues'] = translated_dialogues_for_item
                    current_lang_translated_data.append(translated_item)

            # --- After processing all items for a language ---
            if current_lang_translated_data:
                print(f"\n  Saving initial translated data for {lang_name} to {output_filepath}...")
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        json.dump(current_lang_translated_data, f, ensure_ascii=False, indent=4)
                    print(f"  Successfully saved {len(current_lang_translated_data)} initially translated items to {output_filename}.")
                except Exception as e:
                    print(f"  Error writing initial output file {output_filepath}: {e}")
                    total_errors += 1
            elif not language_had_errors:
                print(f"  No data was translated successfully for {lang_name} in Phase 1. Output file not created.")
            else:
                print(f"  No initial data saved for {lang_name} due to processing errors.")
    else:
        print("--- Skipping Phase 1: Initial Translation ---")

    # === Phase 2: Validation and Re-translation ===
    print("\n--- Phase 2: Validation and Re-translation ---")

    for lang_code, lang_name in LANGUAGES_TO_PROCESS.items():
        output_filename = f"{lang_code}2en.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        input_filepath = os.path.join(INPUT_DIR, f"{lang_code}.json")  # Need original source again

        print(f"\n--- Validating and Re-translating: {lang_name} ({output_filename}) ---")

        if not os.path.exists(output_filepath):
            print(f"  Translated file not found: {output_filepath}. Skipping validation.")
            continue
        if not os.path.exists(input_filepath):
            print(f"  Original source file not found: {input_filepath}. Cannot validate. Skipping.")
            continue

        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                translated_data = json.load(f)
            with open(input_filepath, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            if not isinstance(translated_data, list) or not isinstance(source_data, list):
                print(f"  Error: Expected lists in {output_filepath} or {input_filepath}. Skipping validation.")
                continue
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Error reading or parsing JSON files for validation ({output_filepath} or {input_filepath}): {e}. Skipping language.")
            total_errors += 1
            continue

        # Create a map of original dialogues for quick lookup during validation
        original_dialogue_map = {}
        for item in source_data:
            word = item.get('word')
            if word and 'dialogues' in item:
                original_dialogue_map[word] = item.get('dialogues', [])

        needs_resave = False
        dialogues_to_retranslate_count = 0

        # Use tqdm for iterating through translated items
        for i, translated_item in enumerate(tqdm(translated_data, desc=f"Validating {lang_name}", unit="item")):
            subject_word = translated_item.get('word')
            if not subject_word or subject_word not in original_dialogue_map:
                print(f"    Warning: Cannot find original data for word '{subject_word}' in item {i+1}. Skipping validation for this item.")
                continue

            original_dialogues_list = original_dialogue_map[subject_word]
            translated_dialogues_list = translated_item.get('dialogues', [])

            if len(original_dialogues_list) != len(translated_dialogues_list):
                print(f"    Warning: Dialogue count mismatch for word '{subject_word}'. Original: {len(original_dialogues_list)}, Translated: {len(translated_dialogues_list)}. Skipping detailed validation for this item.")
                continue

            for j, translated_dialogue_entry in enumerate(translated_dialogues_list):
                # Find corresponding original dialogue (assuming order is preserved)
                if j >= len(original_dialogues_list):
                    print(f"    Warning: Index out of bounds when accessing original dialogue for word '{subject_word}', dialogue index {j+1}. Skipping validation.")
                    continue
                original_dialogue_entry = original_dialogues_list[j]
                original_utterances_for_check = original_dialogue_entry.get('dialogue', [])
                translated_utterances = translated_dialogue_entry.get('dialogue', [])
                ss_idx = original_dialogue_entry.get('meta_data', {}).get('ss_idx')

                if ss_idx is None or not original_utterances_for_check:
                    continue

                # Perform consistency checks on the already translated data
                keys_ok = check_keys(original_utterances_for_check, translated_utterances)
                brackets_ok = check_brackets(translated_utterances, ss_idx)

                if not keys_ok or not brackets_ok:
                    dialogues_to_retranslate_count += 1
                    print(f"  ! Consistency failed for word '{subject_word}', dialogue {j+1}. Attempting re-translation...")

                    # Attempt re-translation using the validation function
                    retranslated_utterances = translate_and_validate_dialogue(
                        original_dialogue_entry, subject_word, lang_name, original_utterances_for_check
                    )

                    if retranslated_utterances is not None:
                        print(f"    Re-translation successful for word '{subject_word}', dialogue {j+1}.")
                        # Update the dialogue in the loaded translated_data structure
                        translated_data[i]['dialogues'][j]['dialogue'] = retranslated_utterances
                        needs_resave = True
                    else:
                        print(f"    Re-translation FAILED for word '{subject_word}', dialogue {j+1} after retries.")
                        global_retranslation_failures += 1

        # --- After validating all items for a language ---
        if needs_resave:
            print(f"\n  Re-saving updated translated data for {lang_name} to {output_filepath}...")
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(translated_data, f, ensure_ascii=False, indent=4)
                print(f"  Successfully re-saved updated data for {lang_name}.")
                print(f"  ({dialogues_to_retranslate_count} dialogues were flagged for re-translation attempt)")
            except Exception as e:
                print(f"  Error re-writing updated output file {output_filepath}: {e}")
                total_errors += 1
        else:
            print(f"  No changes needed for {lang_name} after validation.")

    # --- Final Summary ---
    print("\n--- Translation and Validation process finished ---")
    if global_translation_failures > 0:
        print(f"Phase 1: Completed with {global_translation_failures} dialogues failing initial translation attempts.")
    if global_retranslation_failures > 0:
        print(f"Phase 2: Completed with {global_retranslation_failures} dialogues failing re-translation attempts after validation failure.")
    if total_errors > 0:
        print(f"Encountered {total_errors} other errors during processing (reading files, preprocessing, saving).")
    if total_errors == 0 and global_translation_failures == 0 and global_retranslation_failures == 0:
        print("Completed successfully with no persistent translation failures.")

if __name__ == "__main__":
    # Add argument parser to control skipping Phase 1
    parser = argparse.ArgumentParser(description="Translate dialogues and validate consistency.")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip the initial translation phase (Phase 1) and only run validation/re-translation (Phase 2).")
    args = parser.parse_args()

    main(skip_phase_1=args.skip_phase1)