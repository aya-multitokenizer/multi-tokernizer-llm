"""Utility functions for the project."""

import argparse

import torch
import torch.utils
from torch.utils.data import DataLoader, random_split

from utils.preprocess import get_stories

import yaml


def load_config(config_path: str) -> argparse.Namespace:
    """Load a configuration file."""
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config = {
        k: argparse.Namespace(**v) if isinstance(v, dict) else v
        for k, v in config.items()
    }
    return argparse.Namespace(**config)


def get_dataloaders(config: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Get the data loaders."""
    stories = get_stories(config.data_args.data_path, True)
    stories_dataset = torch.utils.data.TensorDataset(*stories)
    train_split_len = int(len(stories_dataset) * config.data_args.train_split)
    train_dataset, val_dataset = random_split(
        stories_dataset, [train_split_len, len(stories_dataset) - train_split_len]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training_args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_dataloader, val_dataloader
