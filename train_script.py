import os
import argparse
import yaml
from tqdm import trange
import numpy as np
from rms_norm import TransformerLM, AdamW, CrossEntropy, get_batch, save_checkpoint

def main(
        train_data,
        validation_data,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        residual_pdrop: float,
        checkpoint_frequency: int,
        output_dir: str):

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop)

    optimizer = AdamW(model.parameters())
    loss_fn = CrossEntropy()

    max_steps = 1000
    batch_size = 2
    device = 'cpu'
    x, y = get_batch(train_data, batch_size, context_length, device)
    with trange(max_steps) as t:
        for step in range(max_steps):
            optimizer.zero_grad()
            predictions = model(x)
            loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.item())

            if divmod(step, checkpoint_frequency)[1] == 0:
                save_checkpoint(model, optimizer, step, os.path.join(output_dir, f'checkpoint_%d.ckpt'))


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

    attn_pdrop = .1
    residual_pdrop = .1

    validation_data = np.load("tokenized_tiny_stories_valid.npy", mmap_mode="r")
    train_data = validation_data

    main(
        train_data,
        validation_data,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        attn_pdrop,
        residual_pdrop,
        checkpoint_frequency,
        output_dir)
