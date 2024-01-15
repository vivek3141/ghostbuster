from dataclasses import dataclass

import tqdm
import numpy as np
import os


DIR_IGNORE = {"logprobs", "prompts", "headlines"}


@dataclass
class Dataset:
    type: str
    path: str


def get_generate_dataset_normal(path: str, verbose=False):
    files = []
    to_iter = tqdm.tqdm(os.listdir(path)) if verbose else os.listdir(path)

    for file in to_iter:
        if file in DIR_IGNORE:
            continue
        files.append(f"{path}/{file}")

    return files


def get_generate_dataset_author(path: str, author: str, verbose=False):
    files = []

    if author is None:
        authors = sorted(os.listdir(path))
    else:
        authors = [author]

    to_iter = tqdm.tqdm(authors) if verbose else authors

    for author in to_iter:
        for file in sorted(os.listdir(f"{path}/{author}")):
            if file in DIR_IGNORE:
                continue
            files.append(f"{path}/{author}/{file}")

    return files


def get_generate_dataset(*datasets: Dataset):
    def generate_dataset(featurize, split=None, verbose=False, author=None):
        files = []
        for dataset in datasets:
            if dataset.type == "normal":
                files += get_generate_dataset_normal(dataset.path)
            elif dataset.type == "author":
                files += get_generate_dataset_author(dataset.path, author=author)

        if split is not None:
            files = np.array(files)[split]

        data = []
        files = tqdm.tqdm(files) if verbose else files

        for file in files:
            if "logprobs" in file:
                continue
            data.append(featurize(file))
        return np.array(data)

    return generate_dataset
