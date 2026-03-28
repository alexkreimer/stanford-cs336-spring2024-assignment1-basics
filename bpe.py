from typing import TypeAlias
import regex as re
from collections import Counter
from collections import defaultdict
Token: TypeAlias = int

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_BYTES = re.compile(br"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
SPECIALS_BYTES = re.compile(br"<\|.*?\|>")

def train_bpe(
        pre_tokens_count: Counter[bytes],
        vocab_size:int,
        special_tokens: list[str]
        ) -> tuple[dict[Token, bytes], list[tuple[bytes, bytes]]]:
    _vocab_size = 2 ** 8
    vocab: dict[Token, bytes] = {}
    for i in range(_vocab_size):
        bytes_of_i = i.to_bytes(1, 'big')
        vocab[i] = bytes_of_i
    _vocab_size -= 1
    for special_token in special_tokens:
        _vocab_size += 1
        vocab[_vocab_size] = bytes(special_token, 'utf-8')

    merges = []
    word_list: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(lambda: set())
    pair_count = defaultdict(lambda: 0)

    for kk in pre_tokens_count.keys():
        k: tuple[bytes, ...] = tuple(kk[i:i+1] for i in range(len(kk)))
        key_pairs = [(c1, c2) for c1, c2 in zip(k[:-1], k[1:])]
        for pair in key_pairs:
            pair_count[pair] = pair_count[pair] + pre_tokens_count[kk]
            word_list[pair].add(k)

    merged_to_original: dict[tuple[bytes, ...], bytes] = {}
    for _ in range(vocab_size - len(vocab)):
        max_count = max(pair_count.values())
        most_common_items = [k for k, v in pair_count.items() if v == max_count]
        most_common_pair = max(most_common_items)

        merges.append(most_common_pair)
        most_common_bytes = most_common_pair[0] + most_common_pair[1]
        _vocab_size += 1
        vocab[_vocab_size] = most_common_bytes

        words = word_list[most_common_pair].copy()
        for word in words:
            # reduce counts
            for i in range(len(word) - 1):
                c1, c2 = word[i], word[i+1]
                pair_count[(c1, c2)] -= pre_tokens_count[merged_to_original.get(word, b''.join(word))]
                if word in word_list[(c1, c2)]:
                    word_list[(c1, c2)].remove(word)

            it = iter(range(len(word)))
            merged = []
            for i in it:
                if i < len(word) - 1:
                    if word[i] == most_common_pair[0] and word[i+1] == most_common_pair[1]:
                        merged.append(most_common_bytes)
                        next(it, None)
                    else:
                        merged.append(word[i])
                else:
                    merged.append(word[i])
            merged = tuple(merged)
            for i in range(len(merged) - 1):
                c1, c2 = merged[i], merged[i+1]
                pair_count[(c1, c2)] += pre_tokens_count[merged_to_original.get(word, b''.join(word))]
                word_list[(c1, c2)].add(merged)
    return vocab, merges
