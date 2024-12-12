import json
import tiktoken
import csv
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_DIR = os.path.join(ROOT_DIR, 'input_data')
OUTPUT_DATA_DIR = os.path.join(ROOT_DIR, 'single_token_words')
OUTPUT_INFO_DIR = os.path.join(ROOT_DIR, 'single_token_words_info')

words = {
    'english': {
        'path': os.path.join(INPUT_DATA_DIR, 'english_valid_words_sorted_by_frequency.csv'),
        'word_index': 1,
    },
}

tokenizer_encoding_map = {
    'openai_tiktoken': [
        'o200k_base', # o1, gpt4o, gpt-4o-mini
        'cl100k_base', # gpt-4, gpt-3.5-turbo
    ]
}

def load_words_map(language):
    with open(words[language]['path'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            yield row[words[language]['word_index']]

def generate_for_tokenizer(tokenizer_name, language):
    try:
        encoders = tokenizer_encoding_map[tokenizer_name]
    except ValueError:
        print(f"Tokenizer {tokenizer_name} not found")
        return

    output = {}
    for encoding_name in encoders:
        out_name = f"{tokenizer_name}-{encoding_name}"
        encoding = tiktoken.get_encoding(encoding_name)
        single_token_words = []
        for word in load_words_map(language):
            tokens = encoding.encode(word)
            if len(tokens) == 1:
                single_token_words.append(word)
        output[out_name] = {
            'tokenizer': tokenizer_name,
            'encoding': encoding_name,
            'count': len(single_token_words),
            'words': list(dict.fromkeys(single_token_words)),
        }
    return output


def generate_for_language(language):
    for tokenizer_name in tokenizer_encoding_map:
        output = generate_for_tokenizer(tokenizer_name, language)
        for encoding_name, data in output.items():
            words = data.pop('words')
            # words to json
            with open(os.path.join(OUTPUT_DATA_DIR, f"{language}-{encoding_name}.json"), 'w') as f:
                json.dump(words, f)
            
            # words to csv
            with open(os.path.join(OUTPUT_DATA_DIR, f"{language}-{encoding_name}.csv"), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(words)

            # info to json
            with open(os.path.join(OUTPUT_INFO_DIR, f"{language}-{encoding_name}.json"), 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)
            

if __name__ == "__main__":
    generate_for_language('english')
