import os
import mmap
import itertools
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from tokenizer import Tokenizer

# Global variable for worker processes to avoid re-loading the tokenizer
TOKENIZER = None

def init_worker(vocab_path, merges_path):
    """Initializes the tokenizer once per worker process."""
    global TOKENIZER
    TOKENIZER = Tokenizer.from_files(vocab_path, merges_path)

def encode_chunk(chunk_bytes):
    """Encodes a byte chunk into tokens."""
    # Decode bytes to string; 'ignore' handles any stray bytes at boundaries
    text_chunk = chunk_bytes.decode('utf-8', errors='ignore')
    return TOKENIZER.encode(text_chunk)

def find_chunk_offsets(file_path, chunk_size):
    """Calculates offsets to avoid splitting UTF-8 characters or words middle-sentence."""
    offsets = []
    file_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        start = 0
        while start < file_size:
            end = min(start + chunk_size, file_size)
            if end < file_size:
                f.seek(end)
                # Nudge 'end' to the next whitespace to avoid splitting tokens
                line_rest = f.readline()
                end += len(line_rest)
            offsets.append((start, end))
            start = end
    return offsets

def process_file(input_path, output_path, chunk_size, num_workers, overall_bar):
    if not os.path.exists(input_path):
        print(f"\n[!] File not found: {input_path}")
        overall_bar.update(1)
        return

    # 1. Calculate smart offsets
    offsets = find_chunk_offsets(input_path, chunk_size)

    with open(input_path, "r+b") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:

            # 2. Setup the worker pool
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=init_worker,
                initargs=('tiny_stories_vocab.json', 'tiny_stories_merges.json')
            ) as executor:

                # Create a generator for the chunks based on our offsets
                chunk_gen = (mm[start:end] for start, end in offsets)

                # 3. Inner Progress Bar (Specific File)
                results = list(tqdm(
                    executor.map(encode_chunk, chunk_gen),
                    total=len(offsets),
                    desc=f" ↳ {os.path.basename(input_path)}",
                    unit="chunk",
                    leave=False
                ))

    # 4. Save results
    flat_results = list(itertools.chain.from_iterable(results))
    arr = np.array(flat_results, dtype='int32')
    np.save(output_path, arr)

    # Update main progress bar
    overall_bar.set_postfix({"tokens": f"{len(arr)/1e6:.1f}M"})
    overall_bar.update(1)

if __name__ == '__main__':
    # Configuration
    NUM_WORKERS = os.cpu_count() or 8
    CHUNK_SIZE = 1_000_000 # ~1MB per chunk

    TASKS = [
        ('data/TinyStoriesV2-GPT4-train.txt', 'tokenized_tiny_stories_train.npy'),
        ('data/TinyStoriesV2-GPT4-valid.txt', 'tokenized_tiny_stories_valid.npy'),
    ]

    print(f"Starting Tokenization with {NUM_WORKERS} workers...")

    with tqdm(total=len(TASKS), desc="Overall Progress", unit="file") as main_bar:
        for inp, outp in TASKS:
            process_file(inp, outp, CHUNK_SIZE, NUM_WORKERS, main_bar)

    print("\nProcessing Complete.")
