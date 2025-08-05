import json
import argparse
import os

def remove_infinigram_info(file_path):
    """
    Removes the 'infinigram_count' key from each object in a JSON file
    and saves the modified data back to the same file.
    The JSON file can be a list of objects or a dictionary of lists of objects.

    Args:
        file_path (str): The path to the JSON file.
    """
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return

        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process data if it's a list of dictionaries
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'infinigram_count' in item:
                    del item['infinigram_count']
        # Process data if it's a dictionary of lists
        elif isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict) and 'infinigram_count' in item:
                            del item['infinigram_count']
        else:
            print("Error: JSON content is not in a supported format (list of objects or dict of lists of objects).")
            return

        # Write the modified data back to the same file
        with open(file_path, 'w', encoding='utf-8') as f:
            # Use indent=4 for pretty-printing the JSON
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Successfully removed 'infinigram_count' from {file_path}")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Please check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Set up the argument parser to get the file path from the command line
    parser = argparse.ArgumentParser(
        description="Remove 'infinigram_count' key from objects in a JSON file."
    )
    parser.add_argument(
        'file_path',
        type=str,
        help='The path to the JSON file to be processed.'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function with the provided file path
    remove_infinigram_info(args.file_path)