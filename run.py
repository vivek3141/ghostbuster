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

with open("best_features.txt") as f:
    best_features = f.read().strip().split("\n")

print("Loading trigram model...")
trigram_model = pickle.load(open("trigram_model.pkl", "rb"), pickle.HIGHEST_PROTOCOL)
tokenizer = tiktoken.encoding_for_model("davinci").encode

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

wp_dataset = [
    Dataset("normal", "writing_prompts/data/human"),
    Dataset("normal", "writing_prompts/data/gpt"),
    Dataset("normal", "writing_prompts/data/claude"),
]

reuter_dataset_train = [
    Dataset("author", "reuter/data/human/train"),
    Dataset("author", "reuter/data/gpt/train"),
    Dataset("author", "reuter/data/claude/train"),
]
reuter_dataset_test = [
    Dataset("author", "reuter/data/human/test"),
    Dataset("author", "reuter/data/gpt/test"),
    Dataset("author", "reuter/data/claude/test"),
]

essay_dataset = [
    Dataset("normal", "essay/data/human"),
    Dataset("normal", "essay/data/gpt"),
    Dataset("normal", "essay/data/claude"),
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


def get_featurized_data(generate_dataset_fn, best_features):
    t_data = generate_dataset_fn(t_featurize)

    davinci, ada, trigram, unigram = get_all_logprobs(
        generate_dataset_fn, trigram=trigram_model, tokenizer=tokenizer
    )

    vector_map = {
        "davinci-logprobs": lambda file: davinci[file],
        "ada-logprobs": lambda file: ada[file],
        "trigram-logprobs": lambda file: trigram[file],
        "unigram-logprobs": lambda file: unigram[file],
    }
    exp_featurize = get_exp_featurize(best_features, vector_map)
    exp_data = generate_dataset_fn(exp_featurize)

    return np.concatenate([t_data, exp_data], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roberta", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
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

    where_gpt = np.where(
        generate_dataset_fn(lambda file: 0 if "claude" in file else 1)
    )[0]
    indices = indices[where_gpt]

    train = [i for i in train if i in indices]
    test = [i for i in test if i in indices]

    if args.roberta:
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            "roberta/models/ghostbuster_roberta_all", num_labels=2
        )
        roberta_model.to(device)

        test_labels = labels[test]
        test_predictions = []

        print("Computing Roberta predictions...")
        roberta_model.eval()
        with torch.no_grad():
            for file in tqdm.tqdm(files[test]):
                with open(file) as f:
                    text = f.read()
                inputs = roberta_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = roberta_model(**inputs)
                test_predictions.append(outputs.logits.argmax(dim=1).item())

        print("Roberta Test Accuracy:", accuracy_score(test_labels, test_predictions))
        print("Roberta Test F1:", f1_score(test_labels, test_predictions))
        print("Roberta Test AUC:", roc_auc_score(test_labels, test_predictions))
