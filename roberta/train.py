import argparse
import math
import numpy as np
import dill as pickle
import tiktoken
import torch
import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from tabulate import tabulate

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

from utils.featurize import normalize, t_featurize, select_features
from utils.symbolic import get_all_logprobs, get_exp_featurize
from utils.load import get_generate_dataset, Dataset

from torch.utils.data import Dataset as TorchDataset, DataLoader

if torch.cuda.is_available():
    print("Using CUDA...")
    device = torch.device("cuda")
else:
    print("Using CPU...")
    device = torch.device("cpu")

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
        "roberta-base", num_labels=2
    ).to(device)

    train_dataset = RobertaDataset(train_text, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(roberta_model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Fine-tuning roberta...")
    roberta_model.train()

    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        outputs = roberta_model(**batch)
        loss = loss_fn(outputs.logits.to(device), batch["labels"].to(device))
        loss.backward()
        optimizer.step()

    # Save Model
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

    if args.train_all:
        train_texts = []
        for file in files[train]:
            with open(file) as f:
                train_texts.append(f.read())

        test_texts = []
        for file in files[test]:
            with open(file) as f:
                test_texts.append(f.read())

        roberta_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        ).to(device)

        train_dataset = RobertaDataset(train_texts, labels[train])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        optimizer = torch.optim.AdamW(roberta_model.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()

        print("Fine-tuning roberta...")
        roberta_model.train()

        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = roberta_model(**batch)
            loss = loss_fn(outputs.logits.to(device), batch["labels"].to(device))
            loss.backward()
            optimizer.step()

        # Save Model
        roberta_model.save_pretrained("model/ghostbuster_roberta_all")

    if args.train_wp:
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        ).to(device)

        where = np.where(
            generate_dataset_fn(lambda file: 1 if "writing_prompts" in file else 0)
        )[0]
        indices = indices[where]

        train = [i for i in train if i in indices]
        test = [i for i in test if i in indices]

        train_texts = []
        for file in files[train]:
            with open(file) as f:
                train_texts.append(f.read())

        test_texts = []
        for file in files[test]:
            with open(file) as f:
                test_texts.append(f.read())

        train_dataset = RobertaDataset(train_texts, labels[train])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        optimizer = torch.optim.AdamW(roberta_model.parameters(), lr=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()

        print("Fine-tuning WP roberta...")
        roberta_model.train()

        for batch in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = roberta_model(**batch)
            loss = loss_fn(outputs.logits.to(device), batch["labels"].to(device))
            loss.backward()
            optimizer.step()

        # Save Model
        roberta_model.save_pretrained("models/ghostbuster_roberta_wp")
