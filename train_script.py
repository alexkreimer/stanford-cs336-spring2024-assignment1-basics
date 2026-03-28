import os
import argparse
import yaml
import numpy as np

from abc import ABC, abstractmethod

import torch
import comet_ml
from rms_norm import TransformerLM, AdamW, CrossEntropy, get_batch, save_checkpoint


class Logger(ABC):
    @abstractmethod
    def log(self, name, value, step=None):
        pass


class CometLogger(Logger):
    def __init__(self, experiment):
        self.experiment = experiment

    def log(self, name, value, step=None):
        self.experiment.log_metric(name, value, step=step)

class ConsoleLogger(Logger):
    def log(self, name, value, step=None):
        print(f"{name}: {value} (step: {step})")

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
        log_frequency: int,
        learning_rate: float,
        epochs: int,
        epoch_length: int,
        val_steps: int,
        batch_size: int,
        logger: Logger,
        device: str,
        output_dir: str):

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    loss_fn = CrossEntropy()

    global_step = 0
    for epoch in range(epochs):
        for step in range(epoch_length):
            x, y = get_batch(train_data, batch_size, context_length, device)
            optimizer.zero_grad()
            predictions = model(x)
            loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()
            global_step += 1

            if divmod(global_step, log_frequency)[1] == 0:
                logger.log("train_loss", loss.item(), step=global_step)

            if divmod(global_step, checkpoint_frequency)[1] == 0:
                save_checkpoint(model, optimizer, step, os.path.join(output_dir, f'checkpoint_%d.ckpt'))

        for step in range(val_steps):
            x, y = get_batch(validation_data, batch_size, context_length, device)
            predictions = model(x)
            loss = loss_fn(predictions, y)
            logger.log("val_loss", loss.item(), step=global_step + step)


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

    if logger_type == 'comet':
        experiment = comet_ml.Experiment(project_name="transformer-lm")
        logger = CometLogger(experiment)
    else:
        logger = ConsoleLogger()

    attn_pdrop = .1
    residual_pdrop = .1

    validation_data = np.load("tokenized_tiny_stories_valid.npy", mmap_mode="r")
    train_data = np.load("tokenized_tiny_stories_train.npy", mmap_mode="r")

    main(
        train_data=train_data,
        validation_data=validation_data,
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        checkpoint_frequency=checkpoint_frequency,
        log_frequency=log_frequency,
        learning_rate=learning_rate,
        epochs=epochs,
        epoch_length=epoch_length,
        val_steps=val_steps,
        batch_size=batch_size,
        logger=logger,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=output_dir)
