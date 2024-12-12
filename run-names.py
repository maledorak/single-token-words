import csv
import json
import os

import tiktoken
from names_dataset import NameDataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DATA_DIR = os.path.join(ROOT_DIR, 'single_token_names')
OUTPUT_INFO_DIR = os.path.join(ROOT_DIR, 'single_token_names_info')


tokenizer_encoding_map = {
    'openai_tiktoken': [
        'o200k_base', # o1, gpt4o, gpt-4o-mini
        'cl100k_base', # gpt-4, gpt-3.5-turbo
    ]
}

countries_map = {
    'english': ['GB', 'IE', 'US', 'CA', 'JM'],
}

def load_names(language):
    nd = NameDataset()
    names = []
    for country in countries_map[language]:
        names_raw = nd.get_top_names(1000000, country_alpha2=country)
        names.extend(names_raw[country]['M'])
        names.extend(names_raw[country]['F'])
    return names

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
        for word in load_names(language):
            word = word.lower()
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
                json.dump(words, f, ensure_ascii=False)
            
            # words to csv
            with open(os.path.join(OUTPUT_DATA_DIR, f"{language}-{encoding_name}.csv"), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(words)

            # info to json
            with open(os.path.join(OUTPUT_INFO_DIR, f"{language}-{encoding_name}.json"), 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)
            

if __name__ == "__main__":
    generate_for_language('english')
