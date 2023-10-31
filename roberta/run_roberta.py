import argparse
import math
import numpy as np
import torch
import tqdm
import csv

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from utils.load import get_generate_dataset, Dataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.nn import functional as F

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


def get_scores(labels, probabilities, calibrated=False, precision=6):
    if calibrated:
        threshold = sorted(probabilities)[len(labels) - sum(labels) - 1]
    else:
        threshold = 0.5

    assert len(labels) == len(probabilities)

    if sum(labels) == 0:
        return (
            round(accuracy_score(labels, probabilities > threshold), precision),
            round(f1_score(labels, probabilities > threshold), precision),
            -1,
        )

    return (
        round(accuracy_score(labels, probabilities > threshold), precision),
        round(f1_score(labels, probabilities > threshold), precision),
        round(roc_auc_score(labels, probabilities), precision),
    )


def train_roberta_model(train_text, train_labels, output_dir, max_epochs=1):
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    ).to(device)

    optimizer = torch.optim.SGD(roberta_model.parameters(), lr=0.001)
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

    for epoch in range(max_epochs):
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


def run_roberta_model(model_name, texts, labels):
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        f"models/roberta_{model_name}", num_labels=2
    ).to(device)

    roberta_model.eval()

    probs = []
    for text in tqdm.tqdm(texts):
        encoding = roberta_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = roberta_model(**encoding)

        probs.append(float(F.softmax(outputs.logits, dim=1)[0][1].item()))

        del encoding, outputs

    return get_scores(np.array(labels), np.array(probs), calibrated=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_file", default="roberta_results.csv")
    args = parser.parse_args()

    np.random.seed(args.seed)
    # Construct the test/train split. Seed of 0 ensures seriality across
    # all files performing the same split.
    indices = np.arange(6000)
    np.random.shuffle(indices)

    train, test = (
        indices[: math.floor(0.8 * len(indices))],
        indices[math.floor(0.8 * len(indices)) :],
    )

    # [4320 2006 5689 ... 4256 5807 4875] [5378 5980 5395 ... 1653 2607 2732]
    print("Train/Test Split:", train, test)

    generate_dataset = get_generate_dataset(
        *wp_dataset, *reuter_dataset, *essay_dataset
    )
    labels = generate_dataset(lambda f: "gpt" in f)
    assert len(labels) // 2 == sum(labels)
    if args.train:

        def train_roberta_gen(
            gen_fn, out_dir, indices=None, max_epochs=2, filter_fn=lambda x: True
        ):
            train_text, train_labels = [], []
            if indices is not None:
                files = gen_fn(lambda f: f)[indices]
            else:
                files = gen_fn(lambda f: f)

            for file in files:
                if not filter_fn(file):
                    continue

                with open(file) as f:
                    text = f.read()

                train_text.append(text)
                train_labels.append(int("gpt" in file))

            train_roberta_model(
                train_text, train_labels, out_dir, max_epochs=max_epochs
            )

        gen_fn_all = get_generate_dataset(*wp_dataset, *reuter_dataset, *essay_dataset)
        gen_fn_wp = get_generate_dataset(*reuter_dataset, *essay_dataset)
        gen_fn_reuter = get_generate_dataset(*wp_dataset, *essay_dataset)
        gen_fn_essay = get_generate_dataset(*wp_dataset, *reuter_dataset)

        train_roberta_gen(gen_fn_all, "models/roberta_gpt", indices=train)
        train_roberta_gen(gen_fn_wp, "models/roberta_wp")
        train_roberta_gen(gen_fn_reuter, "models/roberta_reuter")
        train_roberta_gen(gen_fn_essay, "models/roberta_essay")

    if args.run:
        results_table = [
            ["Model Type", "Experiment", "Accuracy", "F1", "AUC"],
        ]

        def get_data(gen_fn, indices=None, filter_fn=lambda f: True):
            if indices is not None:
                files = gen_fn(lambda f: f)[indices]
            else:
                files = gen_fn(lambda f: f)

            texts, labels = [], []
            for file in files:
                if not filter_fn(file):
                    continue

                with open(file) as f:
                    text = f.read()

                texts.append(text)
                labels.append(int("gpt" in file))

            return texts, labels

        gen_fn_all = get_generate_dataset(*wp_dataset, *reuter_dataset, *essay_dataset)
        gen_fn_wp = get_generate_dataset(*wp_dataset)
        gen_fn_reuter = get_generate_dataset(*reuter_dataset)
        gen_fn_essay = get_generate_dataset(*essay_dataset)

        results_table.append(
            [
                "RoBERTa",
                "In-Domain",
                *run_roberta_model("gpt", *get_data(gen_fn_all, test)),
            ]
        )

        for domain in ["wp", "reuter", "essay"]:
            results_table.append(
                [
                    "RoBERTa",
                    f"In-Domain ({domain})",
                    *run_roberta_model(
                        "gpt" if domain != "reuter" else "only_reuter",
                        *get_data(gen_fn_all, test, lambda x: domain in x),
                    ),
                ]
            )

            results_table.append(
                [
                    "RoBERTa",
                    f"Out-Domain ({domain})",
                    *run_roberta_model(
                        domain, *get_data(gen_fn_all, test, lambda x: domain in x)
                    ),
                ]
            )

        if len(results_table) > 1:
            with open(args.output_file, "w") as f:
                writer = csv.writer(f)
                writer.writerows(results_table)

            print(f"Saved results to {args.output_file}")
