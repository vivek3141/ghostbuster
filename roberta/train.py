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
    Dataset("normal", "../writing_prompts/data/human"),
    Dataset("normal", "../writing_prompts/data/gpt"),
    Dataset("normal", "../writing_prompts/data/claude"),
]

reuter_dataset_train = [
    Dataset("author", "../reuter/data/human/train"),
    Dataset("author", "../reuter/data/gpt/train"),
    Dataset("author", "../reuter/data/claude/train"),
]
reuter_dataset_test = [
    Dataset("author", "../reuter/data/human/test"),
    Dataset("author", "../reuter/data/gpt/test"),
    Dataset("author", "../reuter/data/claude/test"),
]

essay_dataset = [
    Dataset("normal", "../essay/data/human"),
    Dataset("normal", "../essay/data/gpt"),
    Dataset("normal", "../essay/data/claude"),
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

    for epoch in range(5):
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
    parser.add_argument("--train_wp", action="store_true")
    parser.add_argument("--train_reuter", action="store_true")
    parser.add_argument("--train_essay", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    datasets = [
        *wp_dataset,
        *reuter_dataset_train,
        *reuter_dataset_test,
        *essay_dataset,
    ]
    generate_dataset_fn = get_generate_dataset(*datasets)

    files = generate_dataset_fn(lambda x: x)
    labels = generate_dataset_fn(
        lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
    )
    indices = np.arange(len(labels))

    np.random.shuffle(indices)
    train, test = (
        indices[: math.floor(0.8 * len(indices))],
        indices[math.floor(0.8 * len(indices)) :],
    )

    # Construct all indices
    def get_indices(filter_fn):
        where = np.where(generate_dataset_fn(filter_fn))[0]

        curr_train = [i for i in train if i in where]
        curr_test = [i for i in test if i in where]

        return curr_train, curr_test

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

            if domain == "wp":
                domain = "writing_prompts"

            train_indices, test_indices = get_indices(
                lambda file: 1 if domain in file and model in file else 0
            )

            indices_dict[train_key] = train_indices
            indices_dict[test_key] = test_indices

    if args.train_all:
        for train_model, train_domain, test_model, test_domain in tqdm.tqdm(
            list(itertools.product(models, domains, models, domains))
        ):
            print(f"Training on {train_model}_{train_domain}...")

            train_indices = (
                indices_dict[f"{train_model}_{train_domain}_train"]
                + indices_dict[f"human_{train_domain}_train"]
            )

            train_texts = []
            for file in files[train_indices]:
                with open(file) as f:
                    train_texts.append(f.read())

            train_roberta_model(
                train_texts,
                labels[train_indices],
                f"models/roberta_{train_model}_{train_domain}",
            )
