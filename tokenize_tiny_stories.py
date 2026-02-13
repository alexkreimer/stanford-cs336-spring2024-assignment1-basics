import os
import itertools
from typing import Iterable
from collections.abc import Generator
from tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def chunk_generator(text: str, chunk_size: int) -> Generator[str, None, None]:
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def tokenize_text(text: str, chunk_size: int, num_workers: int) -> Iterable:
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        return itertools.chain.from_iterable(tqdm(executor.map(tokenizer.encode, chunk_generator(text, chunk_size))))

if __name__  == '__main__':
    tokenizer = Tokenizer.from_files('tiny_stories_vocab.json', 'tiny_stories_merges.json')

    num_workers = os.cpu_count() or 8
    chunk_size = 100000
    with open('data/TinyStoriesV2-GPT4-valid.txt', 'r') as fd:
        text = fd.read()
        text_len = len(text)
        divisor, remainder = divmod(text_len, chunk_size)
        num_chunks = divisor
        if remainder > 0:
            num_chunks += 1
        print(f'text len is {text_len}, {num_chunks} chunks of size {chunk_size}')

        results = tokenize_text(text, chunk_size, num_workers)

        arr = np.fromiter(results, dtype='int32')
        np.save('tokenized_tiny_stories_valid.npy', arr)

    with open('data/TinyStoriesV2-GPT4-train.txt', 'r') as fd:
        text = fd.read()
        text_len = len(text)
        divisor, remainder = divmod(text_len, chunk_size)
        num_chunks = divisor
        if remainder > 0:
            num_chunks += 1
        print(f'text len is {text_len}, {num_chunks} chunks of size {chunk_size}')

        results = tokenize_text(text, chunk_size, num_workers)

        arr = np.fromiter(results, dtype='int32')
        np.save('tokenized_tiny_stories_train.npy', arr)
