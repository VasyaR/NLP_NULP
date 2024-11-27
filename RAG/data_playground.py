import os
import pandas as pd
import json
from typing import Callable

# Define the base path
base_path = "/mnt/d/Semester7/NLP/RAG/Data"

# Construct the full paths
reddit_jokes_1_path = os.path.join(base_path, "reddit_jokes1.csv")
reddit_jokes_1_path_processed = os.path.join(base_path, "reddit_jokes1_processed.json")

hate_speech_path = os.path.join(base_path, "hate_speech.csv")
hate_speech_path_processed = os.path.join(base_path, "hate_speech_processed.json")

reddit_jokes_2_path = os.path.join(base_path, "reddit_jokes2.json")
reddit_jokes_2_processed_path = os.path.join(base_path, "reddit_jokes2_processed.json")

stupidstuff_path = os.path.join(base_path, "stupidstuff.json")
stupidstuff_path_processed = os.path.join(base_path, "stupidstuff_processed.json")

wocka_path = os.path.join(base_path, "wocka.json")
wocka_path_processed = os.path.join(base_path, "wocka_processed.json")

def csv_to_json(in_path: str, out_path: str, preprocess_function: Callable[[list], None] = None) -> None:
    # Read the CSV file
    df = pd.read_csv(in_path)

    # Convert the DataFrame to a list of dictionaries
    data = df.to_dict(orient='records') # orient='records' means that each row is converted to a dictionary

    # Preprocess the data
    if preprocess_function is not None:
        preprocess_function(data)

    # Save the list to a JSON file
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=4)

def preprocess_json(in_path: str, out_path: str, preprocess_function: Callable[[list], None]) -> None:
    # Read json file
    with open(in_path, 'r') as f:
        data = json.load(f)

    # Preprocess the data
    preprocess_function(data)

    # Save the modified list to a new JSON file
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=4)

def delete_id(data: list) -> None:

    # Remove "id" from each dictionary
    for joke in data:
        if 'id' in joke:
            del joke['id']

def delete_Content_int(data: list) -> None:

    # Remove "Content_int" from each dictionary
    for joke in data:
        if 'Content_int' in joke:
            del joke['Content_int']

if __name__ == "__main__":

    # preprocess_json(reddit_jokes_2_path, reddit_jokes_2_processed_path, delete_id)
    # preprocess_json(stupidstuff_path, stupidstuff_path_processed, delete_id)
    # preprocess_json(wocka_path, wocka_path_processed, delete_id)
    # csv_to_json(reddit_jokes_1_path, reddit_jokes_1_path_processed)
    # csv_to_json(hate_speech_path, hate_speech_path_processed, delete_Content_int)


    pass


