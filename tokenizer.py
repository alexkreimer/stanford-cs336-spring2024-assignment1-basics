from collections.abc import Iterable, Iterator
import json
import re

from bpe_example import pre_tokenize

def apply_merges(s: str, merges: list[tuple[bytes, bytes]]):
    s_bytes = s.encode('utf-8')
    s_bytes = [s_bytes[i:i+1] for i in range(len(s_bytes))]
    
    # Create merge priority map: lower index = higher priority (learned earlier)
    merge_priority = {merge: i for i, merge in enumerate(merges)}

    while True:
        # Find the highest priority merge in current sequence
        best_pair = None
        best_priority = len(merges)
        best_pos = -1
        
        for i in range(len(s_bytes) - 1):
            pair = (s_bytes[i], s_bytes[i+1])
            if pair in merge_priority and merge_priority[pair] < best_priority:
                best_priority = merge_priority[pair]
                best_pair = pair
                best_pos = i
        
        if best_pair is None:
            break
        
        # Apply the best merge
        new_bytes = []
        i = 0
        while i < len(s_bytes):
            if i == best_pos:
                new_bytes.append(s_bytes[i] + s_bytes[i+1])
                i += 2
            else:
                new_bytes.append(s_bytes[i])
                i += 1
        s_bytes = new_bytes
    return s_bytes

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None): 
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Build regex pattern for special tokens
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped_specials = [re.escape(token) for token in sorted_specials]
            self.special_pattern = '(' + '|'.join(escaped_specials) + ')'
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath) as fd:
            vocab = json.load(fd)

        with open(merges_filepath) as fd:
            merges = json.load(fd)

        vocab = {int(token): eval(v) for token, v in vocab.items()}
        merges = [eval(v) for v in merges]
        return cls(vocab, merges, special_tokens)

    def _encode_chunk(self, text: str) -> Iterator[int]:
        """Encode a text chunk, handling special tokens."""
        if self.special_pattern:
            parts = re.split(self.special_pattern, text)
            for part in parts:
                if part in self.special_tokens:
                    token_bytes = part.encode('utf-8')
                    if token_bytes in self.reverse_vocab:
                        yield self.reverse_vocab[token_bytes]
                elif part:
                    for pre_token in pre_tokenize(part):
                        merged = apply_merges(pre_token, self.merges)
                        for k in merged:
                            yield self.reverse_vocab[k]
        else:
            for pre_token in pre_tokenize(text):
                merged = apply_merges(pre_token, self.merges)
                for k in merged:
                    yield self.reverse_vocab[k]

    def encode(self, text: str) -> list[int]:
        return list(self._encode_chunk(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self._encode_chunk(chunk)

    def decode(self, ids: list[int]) -> str:
        return b''.join([self.vocab[_id] for _id in ids]).decode('utf-8', errors='replace')

