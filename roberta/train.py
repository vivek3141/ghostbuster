import argparse
import math
import numpy as np
import dill as pickle
import torch
import tqdm


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

    if args.train_all:
        train_texts = []
        for file in files[train]:
            with open(file) as f:
                train_texts.append(f.read())

        test_texts = []
        for file in files[test]:
            with open(file) as f:
                test_texts.append(f.read())

        train_roberta_model(
            train_texts, labels[train], "models/ghostbuster_roberta_all"
        )

    if args.train_wp:
        where_wp = np.where(
            generate_dataset_fn(lambda file: 1 if "writing_prompts" in file else 0)
        )
        train_wp = [i for i in train if i in where_wp[0]]

        train_texts_wp = []
        for file in files[train_wp]:
            with open(file) as f:
                train_texts_wp.append(f.read())
            assert "writing_prompts" in file

        train_roberta_model(
            train_texts, labels[train_wp], "models/ghostbuster_roberta_wp"
        )
        print("Saved model to models/ghostbuster_roberta_wp")

    if args.train_reuter:
        where_reuter = np.where(
            generate_dataset_fn(lambda file: 1 if "reuter" in file else 0)
        )
        train_reuter = [i for i in train if i in where_reuter[0]]

        train_texts_reuter = []
        for file in files[train_reuter]:
            with open(file) as f:
                train_texts_reuter.append(f.read())
            assert "reuter" in file

        train_roberta_model(
            train_texts_reuter,
            labels[train_reuter],
            "models/ghostbuster_roberta_reuter",
        )
        print("Saved model to models/ghostbuster_roberta_reuter")

    if args.train_essay:
        where_essay = np.where(
            generate_dataset_fn(lambda file: 1 if "essay" in file else 0)
        )
        train_essay = [i for i in train if i in where_essay[0]]

        train_texts_essay = []
        for file in files[train_essay]:
            with open(file) as f:
                train_texts_essay.append(f.read())
            assert "essay" in file

        train_roberta_model(
            train_texts_essay, labels[train_essay], "models/ghostbuster_roberta_essay"
        )
        print("Saved model to models/ghostbuster_roberta_essay")
