"""Dataset utilities."""

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

from datasets import load_dataset

import tokenizers

import torch

import tqdm


def preprocess_data(args: argparse.Namespace) -> None:
    """Preprocess the data."""
    tinystories_ds = load_dataset("roneneldan/TinyStories")
    tinystories_sp_ds = load_dataset(
        "robrenaud/multilingual_tinystories",
        data_files=[f"stories_{i:02d}.json" for i in range(1, 50)],
    )

    with open("train.txt", "w") as full_stories_output:
        for story in tqdm.tqdm(tinystories_sp_ds["train"]["story"]):
            full_stories_output.write(story)
        for story in tqdm.tqdm(tinystories_ds["train"]["text"]):
            full_stories_output.write(story)

    tokenizer = (
        create_tokenizer("train.txt")
        if not args.load_tokenizer
        else tokenizers.ByteLevelBPETokenizer(
            "./tiny-stories-bpe-vocab.json", "./tiny-stories-bpe-merges.txt"
        )
    )

    stories = chain(
        tinystories_ds["train"]["text"], tinystories_sp_ds["train"]["story"]
    )
    if not os.path.isdir("tokenized"):
        os.mkdir("tokenized")
    output_buf = []
    num_outputs = 0
    for story in tqdm.tqdm(stories):
        encoded = torch.tensor(tokenizer.encode(story).ids, dtype=torch.short)
        output_buf.append(encoded)
        if len(output_buf) > 500_000:
            torch.save(output_buf, f"tokenized/tokenized-{num_outputs}.pt")
            num_outputs += 1
            output_buf = []
    if output_buf:
        torch.save(output_buf, f"tokenized/tokenized-{num_outputs}.pt")
        num_outputs += 1
        output_buf = []


def load_sharded_story(shard_no: int) -> list[torch.Tensor]:
    """Load a sharded story."""
    return torch.load(f"tokenized/tokenized-{shard_no}.pt")


def get_stories(dir: str, shuffle: bool = False) -> list[torch.Tensor]:
    """Get all stories from a directory of sharded stories."""
    stories = []
    num_shards = len(os.listdir(dir))
    with ThreadPoolExecutor() as pool:
        for story in tqdm.tqdm(
            pool.map(load_sharded_story, range(num_shards)), total=num_shards
        ):
            stories.extend(story)
    random.shuffle(stories) if shuffle else None
    return stories


def create_tokenizer(file: str) -> tokenizers.Tokenizer:
    """Create a tokenizer."""
    tokenizer = tokenizers.ByteLevelBPETokenizer()
    tokenizer.train(files=[file], vocab_size=2**13, min_frequency=2)
    tokenizer.save_model(".", "tiny-stories-bpe")
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-tokenizer", action="store_true")
    args = parser.parse_args()
    preprocess_data(args)
