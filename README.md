# Single token words

This project is to find all the words that can be encoded by a single token in different LLM tokenizers.

Useful if you want to map some large text chunks before sending to LLM.

## How to use

Just copy the output files from [single_token_words](single_token_words) folder to your project and use them.

There are json and csv versions of the files.

In [single_token_words_info](single_token_words_info) folder you can find some info about the words.

## Supported languages

- English - based on [English-Valid-Words](https://github.com/Maximax67/English-Valid-Words) repository

## Supported tokenizers

- openai_tiktoken
    - cl100k_base (gpt-4, gpt-3.5-turbo)
    - o200k_base (gpt-4o, gpt-4o-mini, o1)

