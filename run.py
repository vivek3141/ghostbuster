import argparse
import math
import numpy as np
import dill as pickle
import tiktoken
import torch
import tqdm
import itertools
import csv
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

from utils.featurize import normalize, t_featurize
from utils.symbolic import get_all_logprobs, get_exp_featurize
from utils.load import get_generate_dataset, Dataset

from torch.utils.data import Dataset as TorchDataset

models = ["gpt", "claude"]
domains = ["wp", "reuter", "essay"]

if torch.cuda.is_available():
    print("Using CUDA...")
    device = torch.device("cuda")
else:
    print("Using CPU...")
    device = torch.device("cpu")

with open("results/best_features_one.txt") as f:
    best_features_one = f.read().strip().split("\n")

with open("results/best_features_two.txt") as f:
    best_features_two = f.read().strip().split("\n")

with open("results/best_features_three.txt") as f:
    best_features_three = f.read().strip().split("\n")

with open("results/best_features_no_gpt.txt") as f:
    best_features_no_gpt = f.read().strip().split("\n")

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

    parser.add_argument("--ghostbuster_depth_one", action="store_true")
    parser.add_argument("--ghostbuster", action="store_true")
    parser.add_argument("--ghostbuster_depth_three", action="store_true")
    parser.add_argument("--ghostbuster_no_gpt", action="store_true")

    parser.add_argument("--ghostbuster_no_handcrafted", action="store_true")
    parser.add_argument("--ghostbuster_no_symbolic", action="store_true")
    parser.add_argument("--ghostbuster_vary_training_data", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="results.csv")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Results table, outputted to args.output_file.
    # Example Row: ["Ghostbuster (No GPT)", "WP", "gpt_wp", 0.5, 0.5, 0.5]
    results_table = [
        ["Model Type", "Training Data", "Test Data", "Accuracy", "F1", "AUC"],
    ]

    # Construct the generate_dataset_fn. This function takes in a featurize function,
    # which is a mapping from a file location (str) to a desired feature vector.
    datasets = [
        *wp_dataset,
        *reuter_dataset_train,
        *reuter_dataset_test,
        *essay_dataset,
    ]
    generate_dataset_fn = get_generate_dataset(*datasets)

    # Get a list of all files and the corresponding labels
    files = generate_dataset_fn(lambda x: x)
    labels = generate_dataset_fn(
        lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
    )

    # Construct the test/train split. Seed of 0 ensures seriality across
    # all files performing the same split.
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

    def get_roberta_predictions(model, indices):
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            model, num_labels=2
        )
        roberta_model.to(device)

        test_labels = labels[indices]
        test_predictions = []

        roberta_model.eval()
        with torch.no_grad():
            for file in files[indices]:
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

        return (
            accuracy_score(test_labels, test_predictions),
            f1_score(test_labels, test_predictions),
            roc_auc_score(test_labels, test_predictions),
        )

    if args.roberta:
        print("Running Roberta Predictions...")

        for m_domain, model, domain in tqdm.tqdm(
            list(itertools.product(domains, models, domains))
        ):
            results_table.append(
                [
                    "Roberta",
                    m_domain,
                    f"{model}_{domain}",
                    *get_roberta_predictions(
                        f"roberta/models/ghostbuster_roberta_{m_domain}",
                        indices_dict[f"{model}_{domain}_test"]
                        + indices_dict[f"human_{domain}_test"],
                    ),
                ]
            )

    def train_ghostbuster(data, train, test):
        model = LogisticRegression(C=10, penalty="l2", max_iter=10000)
        model.fit(data[train], labels[train])

        predictions = model.predict(data[test])
        probs = model.predict_proba(data[test])[:, 1]

        return (
            accuracy_score(labels[test], predictions),
            f1_score(labels[test], predictions),
            roc_auc_score(labels[test], probs),
        )

    def run_experiment(best_features, model_name, train_fn):
        data = normalize(get_featurized_data(generate_dataset_fn, best_features))

        print(f"Running {model_name} Predictions...")
        for train_model, train_domain, test_model, test_domain in tqdm.tqdm(
            list(itertools.product(models, domains, models, domains))
        ):
            train_indices = (
                indices_dict[f"{train_model}_{train_domain}_train"]
                + indices_dict[f"human_{train_domain}_train"]
            )
            test_indices = (
                indices_dict[f"{test_model}_{test_domain}_test"]
                + indices_dict[f"human_{test_domain}_test"]
            )

            results_table.append(
                [
                    model_name,
                    f"{train_model}_{train_domain}",
                    f"{test_model}_{test_domain}",
                    *train_fn(data, train_indices, test_indices),
                ]
            )

    if args.ghostbuster_depth_one:
        run_experiment(best_features_one, "Ghostbuster (Depth One)", train_ghostbuster)

    if args.ghostbuster:
        run_experiment(best_features_two, "Ghostbuster (Depth Two)", train_ghostbuster)

    if args.ghostbuster_depth_three:
        run_experiment(
            best_features_three, "Ghostbuster (Depth Three)", train_ghostbuster
        )

    if args.ghostbuster_no_gpt:
        run_experiment(
            best_features_no_gpt, "Ghostbuster (N-Gram Only)", train_ghostbuster
        )

    if args.ghostbuster_no_handcrafted:

        def train_ghostbuster_no_handcrafted(data, train, test):
            model = LogisticRegression(C=10, penalty="l2", max_iter=10000)
            model.fit(data[train, 7:], labels[train])

            predictions = model.predict(data[test, 7:])
            probs = model.predict_proba(data[test, 7:])[:, 1]

            return (
                accuracy_score(labels[test], predictions),
                f1_score(labels[test], predictions),
                roc_auc_score(labels[test], probs),
            )

        run_experiment(
            best_features_three,
            "Ghostbuster (Depth Three, No Handcrafted)",
            train_ghostbuster_no_handcrafted,
        )

    if args.ghostbuster_no_symbolic:

        def train_ghostbuster_no_symbolic(data, train, test):
            model = LogisticRegression(C=10, penalty="l2", max_iter=10000)
            model.fit(data[train, :7], labels[train])

            predictions = model.predict(data[test, :7])
            probs = model.predict_proba(data[test, :7])[:, 1]

            return (
                accuracy_score(labels[test], predictions),
                f1_score(labels[test], predictions),
                roc_auc_score(labels[test], probs),
            )

        run_experiment(
            best_features_three,
            "Ghostbuster (Depth Three, No Symbolic)",
            train_ghostbuster_no_symbolic,
        )

    if args.ghostbuster_vary_training_data:
        data = normalize(get_featurized_data(generate_dataset_fn, best_features_three))
        training_sizes, scores = [], []

        train_indices = indices_dict["gpt_train"] + indices_dict["human_train"]
        test_indices = indices_dict["gpt_test"] + indices_dict["human_test"]

        np.random.shuffle(train_indices)

        for i in tqdm.tqdm(range(5, len(train_indices))):
            training_sizes.append(i + 1)
            scores.append(train_ghostbuster(data, train_indices[: i + 1], test_indices))

        scores = np.array(scores)

        plt.plot(training_sizes, scores[:, 0], label="Accuracy")
        plt.plot(training_sizes, scores[:, 1], label="F1")
        plt.plot(training_sizes, scores[:, 2], label="AUC")

        plt.xlabel("Training Size (# of Documents)")
        plt.ylabel("Score")

        plt.legend()
        plt.savefig("results/training_size.png")

    if len(results_table) > 1:
        # Write data to output csv file
        with open(args.output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerows(results_table)

        print(f"Saved results to {args.output_file}")
