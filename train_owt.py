import json
import cProfile, pstats
from bpe_example import train_bpe

with open('data/owt_train.txt', 'r') as fd:
        data = fd.read()

with cProfile.Profile() as pr:
    vocab, merges = train_bpe(data, 32000, ['<|endoftext|>'])
    pr.print_stats(sort=pstats.SortKey.CUMULATIVE)

vocab = {token: repr(token_bytes) for token, token_bytes in vocab.items()}
with open('owt_vocab.json', 'w') as fd:
    json.dump(vocab, fd, indent=2)

merges = [repr(v) for v in merges]
with open('owt_merges.json', 'w') as fd:
    json.dump(merges, fd, indent=2)