import argparse
import yaml

import torch
from rms_norm import TransformerLM, load_checkpoint, decode
from tokenizer import Tokenizer


def main(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
        device: str) -> str:

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop).to(device)

    load_checkpoint('checkpoint.ckpt', model, None)
    prompt = input("Enter a prompt: ")
    tokenizer = Tokenizer.from_files('tiny_stories_vocab.json', 'tiny_stories_merges.json')
    prompt_tokens = tokenizer.encode(prompt)
    for _ in range(context_length - len(prompt_tokens)):
        output = decode(prompt_tokens, model, context_length)
        prompt_tokens.append(output)
    return tokenizer.decode(prompt_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_file')
    args = parser.parse_args()

    with open(args.experiment_file, 'r') as f:
        config = yaml.safe_load(f)

    vocab_size = config['vocab_size']
    context_length = config['context_length']
    d_model = config['d_model']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
    d_ff = config['d_ff']
    output_dir = config['output_dir']
    checkpoint_frequency = config['checkpoint_frequency']
    log_frequency = config['log_frequency']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    epoch_length = config['epoch_length']
    val_steps = config['val_steps']
    batch_size = config['batch_size']
    logger_type = config['logger']

    output = main(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=.1,
        residual_pdrop=.1,
        device='cuda' if torch.cuda.is_available() else 'cpu')
    print(output)
