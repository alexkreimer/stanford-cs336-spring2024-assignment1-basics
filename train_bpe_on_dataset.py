import os
import time
import resource
import json
import psutil, mmap
import argparse
import cProfile, pstats
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from bpe import train_bpe, SPECIALS_BYTES, PAT_BYTES

def benchmark(func, *args, name="Task"):
    print(f"--- Benchmarking {name} ---")
    
    # Start tracking
    start_time = time.perf_counter()
    start_cpu = time.process_time() # Total CPU time (sum of all cores)
    
    # Execute
    result = func(*args)
    
    # End tracking
    end_time = time.perf_counter()
    end_cpu = time.process_time()
    
    # Get peak RAM (in KB on Linux, convert to MB)
    peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if os.name != 'nt': # On Linux, ru_maxrss is in KB
        pass 
    
    print(f"Wall time: {end_time - start_time:.2f} seconds")
    print(f"CPU time:  {end_cpu - start_cpu:.2f} seconds")
    print(f"Peak RAM:  {peak_ram:.2f} MB")
    print("-" * 30)
    return result


def check_memory(msg: str) -> None:
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    print(f"{msg}: {mem:.2f} MB")


def process_chunk(file_path: str, start: int, end: int) -> Counter:
    local_counts = Counter()
    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), length=end-start, offset=start, access=mmap.ACCESS_READ) as mm:
            last_pos = 0
            for match in SPECIALS_BYTES.finditer(mm):
                s, e = match.span()
                for token_match in PAT_BYTES.finditer(mm, last_pos, s):
                    local_counts[token_match.group(0)] += 1
                last_pos = e

            for token_match in PAT_BYTES.finditer(mm, last_pos):
                local_counts[token_match.group(0)] += 1

    return local_counts


def _process_chunk_wrapper(args):
    return process_chunk(*args)

def parallel_bpe_counts(file_path: str | os.PathLike, chunk_size=512*1024*1024) -> Counter:
    file_size = os.path.getsize(file_path)
    granularity = mmap.ALLOCATIONGRANULARITY
    chunk_size = (chunk_size // granularity) * granularity

    tasks = []
    for start in range(0, file_size, chunk_size):
        end = min(start + chunk_size, file_size)
        tasks.append((file_path, start, end))

    total_counts = Counter()

    pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, *task) for task in tasks]
        for future in as_completed(futures):
            res = future.result()
            total_counts.update(res)
            pbar.update(chunk_size)
        pbar.close()
    return total_counts


def train_bpe_on_file(file_path: str | os.PathLike, vocab_size: int, special_tokens: list) -> tuple:
    counts = parallel_bpe_counts(file_path)
    vocab, merges = train_bpe(counts, vocab_size, special_tokens)
    return vocab, merges


def save_vocab_and_merges(vocab: dict, merges: list, dataset_name: str) -> None:
    with open(f'{dataset_name}_vocab.json', 'w') as fd:
        json.dump(vocab, fd, indent=2)

    merges = [repr(v) for v in merges]
    with open(f'{dataset_name}_merges.json', 'w') as fd:
        json.dump(merges, fd, indent=2)


datasets = {
    'owt': 'data/owt_train.txt',
    'tiny_stories': 'data/TinyStoriesV2-GPT4-train.txt',
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on a dataset")
    parser.add_argument('--dataset', choices=datasets.keys(), default='owt', help="Dataset to use")
    args = parser.parse_args()

    vocab, merges = benchmark(train_bpe_on_file, datasets[args.dataset], 32000, ['<|endoftext|>'], name="BPE Training")
    save_vocab_and_merges(vocab, merges, args.dataset)
