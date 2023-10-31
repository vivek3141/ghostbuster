import argparse
import math
import numpy as np
import dill as pickle
import torch
import tqdm
import itertools

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

from utils.load import get_generate_dataset, Dataset

from torch.utils.data import Dataset as TorchDataset, DataLoader

if torch.cuda.is_available():
    print("Using CUDA...")
    device = torch.device("cuda")
else:
    print("Using CPU...")
    device = torch.device("cpu")

models = ["gpt", "claude"]
domains = ["wp", "reuter", "essay"]

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

wp_dataset = [
    Dataset("normal", "../data/wp/human"),
    Dataset("normal", "../data/wp/gpt"),
]

reuter_dataset = [
    Dataset("author", "../data/reuter/human"),
    Dataset("author", "../data/reuter/gpt"),
]

essay_dataset = [
    Dataset("normal", "../data/essay/human"),
    Dataset("normal", "../data/essay/gpt"),
]


class RobertaDataset(TorchDataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = roberta_tokenizer(
            self.texts[idx],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze().to(device),
            "attention_mask": encoding["attention_mask"].squeeze().to(device),
            "labels": self.labels[idx],
        }


def train_roberta_model(train_text, train_labels, output_dir):
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-large", num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(roberta_model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Fine-tuning RoBERTa...")

    indices = np.arange(len(train_text))
    np.random.shuffle(indices)

    train, valid = (
        indices[: math.floor(0.8 * len(indices))],
        indices[math.floor(0.8 * len(indices)) :],
    )

    train_labels = np.array(train_labels)
    train_dataset = RobertaDataset([train_text[i] for i in train], train_labels[train])
    val_dataset = RobertaDataset([train_text[i] for i in valid], train_labels[valid])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    prev_val_loss = float("inf")

    for epoch in range(1):
        roberta_model.train()

        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = roberta_model(**batch)
            loss = loss_fn(outputs.logits.to(device), batch["labels"].to(device))
            loss.backward()
            optimizer.step()

            del outputs, loss, batch

        roberta_model.eval()

        val_loss = 0
        for batch in tqdm.tqdm(val_loader):
            outputs = roberta_model(**batch)
            loss = loss_fn(outputs.logits.to(device), batch["labels"].to(device))
            val_loss += loss.item()

            del outputs, loss, batch

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1} Validation Loss: {val_loss}")

        if val_loss > prev_val_loss:
            break

        prev_val_loss = val_loss
        roberta_model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_all", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    datasets = [
        *wp_dataset,
        *reuter_dataset,
        *essay_dataset,
    ]
    generate_dataset_fn = get_generate_dataset(*datasets)

    files = generate_dataset_fn(lambda x: x)
    labels = generate_dataset_fn(
        lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
    )
    indices = np.arange(len(labels))

    # [4320 2006 5689 ... 4256 5807 4875] [5378 5980 5395 ... 1653 2607 2732]
    np.random.shuffle(indices)
    train, test = (
        indices[: math.floor(0.8 * len(indices))],
        indices[math.floor(0.8 * len(indices)) :],
    )
    print("Train/Test Split:", train, test)

    # Construct all indices
    def get_indices(filter_fn):
        where = np.where(generate_dataset_fn(filter_fn))[0]

        curr_train = [i for i in train if i in where]
        curr_test = [i for i in test if i in where]

        return curr_train, curr_test

    def get_texts(indices):
        texts = []
        for file in files[indices]:
            with open(file) as f:
                texts.append(f.read())
        return texts

    indices_dict = {}

    for model in models + ["human"]:
        train_indices, test_indices = get_indices(
            lambda file: 1 if model in file else 0
        )

        indices_dict[f"{model}_train"] = train_indices
        indices_dict[f"{model}_test"] = test_indices

    for model in models + ["human"]:
        for domain in domains:
            train_key = f"{model}_{domain}_train"
            test_key = f"{model}_{domain}_test"

            train_indices, test_indices = get_indices(
                lambda file: 1 if domain in file and model in file else 0
            )

            indices_dict[train_key] = train_indices
            indices_dict[test_key] = test_indices

    if args.train_all:
        train_indices = []
        for domain in domains:
            train_indices += (
                indices_dict[f"gpt_{domain}_train"]
                + indices_dict[f"human_{domain}_train"]
            )

        print("Training on GPT Data")
        print("# of Training Examples:", len(train_indices))
        train_roberta_model(
            get_texts(train_indices),
            labels[train_indices],
            f"models/roberta_gpt",
        )

        for test_domain in domains:
            train_indices = []
            for train_domain in domains:
                if train_domain == test_domain:
                    continue

                train_indices += (
                    indices_dict[f"gpt_{train_domain}_train"]
                    + indices_dict[f"human_{train_domain}_train"]
                )

            print(f"Training on OOD {test_domain} Data")
            print("# of Training Examples:", len(train_indices))
            train_roberta_model(
                get_texts(train_indices),
                labels[train_indices],
                f"models/roberta_{test_domain}",
            )
