# Built-In Imports
import csv
import itertools
import math
import os
from collections import defaultdict

# External Imports
import argparse
import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import tqdm

# Torch imports
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset as TorchDataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

# Local Imports
from utils.featurize import normalize, t_featurize, select_features
from utils.symbolic import get_all_logprobs, get_exp_featurize, backtrack_functions
from utils.load import Dataset, get_generate_dataset

from generate import perturb_char_names, perturb_char_sizes
from generate import perturb_sent_names, perturb_sent_sizes

models = ["gpt"]
domains = ["wp", "reuter", "essay"]
eval_domains = ["claude", "gpt_prompt1", "gpt_prompt2", "gpt_writing", "gpt_semantic"]

if torch.cuda.is_available():
    print("Using CUDA...")
    device = torch.device("cuda")
else:
    print("Using CPU...")
    device = torch.device("cpu")

best_features_map = {}

for file in os.listdir("results"):
    if file.startswith("best_features"):
        with open(f"results/{file}") as f:
            best_features_map[file[:-4]] = f.read().strip().split("\n")

print("Loading trigram model...")
trigram_model = pickle.load(
    open("model/trigram_model.pkl", "rb"), pickle.HIGHEST_PROTOCOL
)
tokenizer = tiktoken.encoding_for_model("davinci").encode

print("Loading features...")
exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
t_data = pickle.load(open("t_data", "rb"))

print("Loading eval data...")
# exp_to_data_eval = pickle.load(open("symbolic_data_eval", "rb"))
# t_data_eval = pickle.load(open("t_data_eval", "rb"))

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

datasets = [
    Dataset("normal", "data/wp/human"),
    Dataset("normal", "data/wp/gpt"),
    Dataset("author", "data/reuter/human"),
    Dataset("author", "data/reuter/gpt"),
    Dataset("normal", "data/essay/human"),
    Dataset("normal", "data/essay/gpt"),
]

eval_dataset = [
    Dataset("normal", "data/wp/claude"),
    Dataset("author", "data/reuter/claude"),
    Dataset("normal", "data/essay/claude"),
    Dataset("normal", "data/wp/gpt_prompt1"),
    Dataset("author", "data/reuter/gpt_prompt1"),
    Dataset("normal", "data/essay/gpt_prompt1"),
    Dataset("normal", "data/wp/gpt_prompt2"),
    Dataset("author", "data/reuter/gpt_prompt2"),
    Dataset("normal", "data/essay/gpt_prompt2"),
    Dataset("normal", "data/wp/gpt_writing"),
    Dataset("author", "data/reuter/gpt_writing"),
    Dataset("normal", "data/essay/gpt_writing"),
    Dataset("normal", "data/wp/gpt_semantic"),
    Dataset("author", "data/reuter/gpt_semantic"),
    Dataset("normal", "data/essay/gpt_semantic"),
]

other_dataset = []


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

    if sum(labels) == 0 or sum(labels) == len(labels):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--claude", action="store_true")

    parser.add_argument("--roberta", action="store_true")
    parser.add_argument("--perplexity_only", action="store_true")

    parser.add_argument("--ghostbuster", action="store_true")

    parser.add_argument("--ghostbuster_depth_one", action="store_true")
    parser.add_argument("--ghostbuster_depth_two", action="store_true")
    parser.add_argument("--ghostbuster_depth_three", action="store_true")
    parser.add_argument("--ghostbuster_depth_four", action="store_true")

    parser.add_argument("--ghostbuster_random", action="store_true")
    parser.add_argument("--ghostbuster_no_gpt", action="store_true")
    parser.add_argument("--ghostbuster_no_handcrafted", action="store_true")
    parser.add_argument("--ghostbuster_no_symbolic", action="store_true")
    parser.add_argument("--ghostbuster_only_ada", action="store_true")
    parser.add_argument("--ghostbuster_custom", action="store_true")
    parser.add_argument("--ghostbuster_other_eval", action="store_true")

    parser.add_argument("--ghostbuster_vary_training_data", action="store_true")
    parser.add_argument("--ghostbuster_vary_document_size", action="store_true")

    parser.add_argument("--hyperparameter_search", action="store_true")

    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--calibration", action="store_true")

    parser.add_argument("--toefl", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="results.csv")
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

    # Results table, outputted to args.output_file.
    # Example Row: ["Ghostbuster (No GPT)", "WP", "gpt_wp", 0.5, 0.5, 0.5]
    results_table = [
        ["Model Type", "Experiment", "Accuracy", "F1", "AUC"],
    ]

    # Construct the generate_dataset_fn. This function takes in a featurize function,
    # which is a mapping from a file location (str) to a desired feature vector.

    generate_dataset_fn_gpt = get_generate_dataset(*datasets)
    generate_dataset_fn_eval = get_generate_dataset(*eval_dataset)

    # t_data_eval = generate_dataset_fn_eval(t_featurize, verbose=True)
    # pickle.dump(t_data_eval, open("t_data_eval", "wb"), pickle.HIGHEST_PROTOCOL)

    generate_dataset_fn = get_generate_dataset(*datasets, *eval_dataset)

    # t_data = generate_dataset_fn(t_featurize, verbose=True)
    # pickle.dump(t_data, open("t_data", "wb"), pickle.HIGHEST_PROTOCOL)

    def get_featurized_data(best_features, gpt_only=False):
        gpt_data = np.concatenate(
            [t_data] + [exp_to_data[i] for i in best_features], axis=1
        )
        if gpt_only:
            return gpt_data

        eval_data = np.concatenate(
            [t_data_eval] + [exp_to_data_eval[i] for i in best_features], axis=1
        )
        return np.concatenate([gpt_data, eval_data], axis=0)

    # Construct all indices
    def get_indices(filter_fn):
        where = np.where(generate_dataset_fn_gpt(filter_fn))[0]

        curr_train = [i for i in train if i in where]
        curr_test = [i for i in test if i in where]

        return curr_train, curr_test

    indices_dict = {}

    for model in models + ["human"]:
        train_indices, test_indices = get_indices(
            lambda file: 1 if model in file else 0,
        )

        indices_dict[f"{model}_train"] = train_indices
        indices_dict[f"{model}_test"] = test_indices

    for model in models + ["human"]:
        for domain in domains:
            train_key = f"{model}_{domain}_train"
            test_key = f"{model}_{domain}_test"

            train_indices, test_indices = get_indices(
                lambda file: 1 if domain in file and model in file else 0,
            )

            indices_dict[train_key] = train_indices
            indices_dict[test_key] = test_indices

    for key in eval_domains:
        where = np.where(generate_dataset_fn(lambda file: 1 if key in file else 0))[0]
        assert len(where) == 3000

        indices_dict[f"{key}_test"] = list(where)

    files = generate_dataset_fn(lambda x: x)
    labels = generate_dataset_fn(
        lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
    )

    def get_roberta_predictions(data, train, test, domain):
        print(f"Loading model roberta/models/roberta_{domain}...")

        roberta_model = RobertaForSequenceClassification.from_pretrained(
            f"roberta/models/roberta_{domain}", num_labels=2
        )
        roberta_model.to(device)

        test_labels = labels[test]
        test_predictions = []

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
                test_predictions.append(
                    float(F.softmax(outputs.logits, dim=1)[0][1].item())
                )

        return get_scores(np.array(test_labels), np.array(test_predictions))

    def train_ghostbuster(data, train, test, domain):
        model = LogisticRegression()
        model.fit(data[train], labels[train])
        probs = model.predict_proba(data[test])[:, 1]

        return get_scores(labels[test], probs)

    def train_perplexity(data, train, test, domain):
        features = data[train][:, -1].reshape(-1, 1)
        threshold = sorted(features)[len(features) - sum(labels[train]) - 1]
        probs = (data[test][:, -1] > threshold).astype(float)
        return get_scores(labels[test], probs)

    def run_experiment(best_features, model_name, train_fn, gpt_only=True):
        gpt_data = get_featurized_data(best_features, gpt_only=True)
        _, mu, sigma = normalize(gpt_data, ret_mu_sigma=True)

        data = normalize(
            get_featurized_data(best_features, gpt_only=gpt_only), mu=mu, sigma=sigma
        )

        print(f"Running {model_name} Predictions...")

        train_indices, test_indices = [], []
        for domain in domains:
            train_indices += (
                indices_dict[f"gpt_{domain}_train"]
                + indices_dict[f"human_{domain}_train"]
            )
            test_indices += (
                indices_dict[f"gpt_{domain}_test"]
                + indices_dict[f"human_{domain}_test"]
            )

            results_table.append(
                [
                    model_name,
                    f"In-Domain ({domain})",
                    *train_fn(
                        data,
                        indices_dict[f"gpt_{domain}_train"]
                        + indices_dict[f"human_{domain}_train"],
                        indices_dict[f"gpt_{domain}_test"]
                        + indices_dict[f"human_{domain}_test"],
                        "gpt",
                    ),
                ]
            )

        results_table.append(
            [
                model_name,
                "In-Domain",
                *train_fn(data, train_indices, test_indices, "gpt"),
            ]
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

            results_table.append(
                [
                    model_name,
                    f"Out-Domain ({test_domain})",
                    *train_fn(
                        data,
                        train_indices,
                        indices_dict[f"gpt_{test_domain}_test"]
                        + indices_dict[f"human_{test_domain}_test"],
                        test_domain,
                    ),
                ]
            )

        if gpt_only:
            return

        train_indices, test_indices = [], []
        for domain in domains:
            train_indices += (
                indices_dict[f"gpt_{domain}_train"]
                + indices_dict[f"human_{domain}_train"]
            )
            test_indices += indices_dict[f"human_{domain}_test"]

        for domain in eval_domains:
            curr_test_indices = list(indices_dict[f"{domain}_test"]) + test_indices

            results_table.append(
                [
                    model_name,
                    f"Out-Domain ({domain})",
                    *train_fn(data, train_indices, curr_test_indices, "gpt"),
                ]
            )

    if args.perplexity_only:
        run_experiment(
            ["davinci-logprobs s-avg"],
            "Perplexity-Only",
            train_perplexity,
        )

    if args.roberta:
        run_experiment([], "RoBERTa", get_roberta_predictions, gpt_only=True)

    if args.ghostbuster_depth_one or args.ghostbuster:
        run_experiment(
            best_features_map["best_features_one"],
            "Ghostbuster (Depth One)",
            train_ghostbuster,
        )

    if args.ghostbuster_depth_two or args.ghostbuster:
        run_experiment(
            best_features_map["best_features_two"],
            "Ghostbuster (Depth Two)",
            train_ghostbuster,
        )

    if args.ghostbuster_depth_three or args.ghostbuster:
        run_experiment(
            best_features_map["best_features_three"],
            "Ghostbuster (Depth Three)",
            train_ghostbuster,
        )

    if args.ghostbuster_depth_four or args.ghostbuster:
        run_experiment(
            best_features_map["best_features_four"],
            "Ghostbuster (Depth Four)",
            train_ghostbuster,
            gpt_only=True,
        )

    if args.ghostbuster_no_gpt or args.ghostbuster:
        run_experiment(
            best_features_map["best_features_no_gpt"],
            "Ghostbuster (N-Gram Only)",
            train_ghostbuster,
        )

    if args.ghostbuster_only_ada or args.ghostbuster:
        run_experiment(
            best_features_map["best_features_only_ada"],
            "Ghostbuster (N-Gram and Ada)",
            train_ghostbuster,
        )

    if args.ghostbuster_random or args.ghostbuster:
        all_features = backtrack_functions(max_depth=3)
        random_features = np.random.choice(all_features, 10, replace=False)

        run_experiment(
            random_features,
            "Ghostbuster (Random)",
            train_ghostbuster,
        )

    if args.ghostbuster_custom:
        run_experiment(
            best_features_map["best_features_custom"],
            "Ghostbuster (Custom)",
            train_ghostbuster,
        )

    if args.ghostbuster_no_handcrafted or args.ghostbuster:

        def train_ghostbuster_no_handcrafted(data, train, test, domain):
            data = data[:, 7:]
            return train_ghostbuster(data, train, test, domain)

        run_experiment(
            best_features_map["best_features_three"],
            "Ghostbuster (Depth Three, No Handcrafted)",
            train_ghostbuster_no_handcrafted,
        )

    if args.ghostbuster_no_symbolic or args.ghostbuster:

        def train_ghostbuster_no_symbolic(data, train, test, domain):
            data = data[:, :7]
            return train_ghostbuster(data, train, test, domain)

        run_experiment(
            best_features_map["best_features_three"],
            "Ghostbuster (No Symbolic)",
            train_ghostbuster_no_symbolic,
        )

    if args.ghostbuster_other_eval:
        data, mu, sigma = normalize(
            get_featurized_data(
                best_features_map["best_features_three"], gpt_only=True
            ),
            ret_mu_sigma=True,
        )

        model = LogisticRegression()
        model.fit(
            data[indices_dict["gpt_train"] + indices_dict["human_train"]],
            labels[indices_dict["gpt_train"] + indices_dict["human_train"]],
        )

        # Get roberta results on ets
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            f"roberta/models/roberta_gpt", num_labels=2
        )
        roberta_model.to(device)

        print(
            get_scores(
                labels[indices_dict["gpt_test"] + indices_dict["human_test"]],
                model.predict_proba(
                    data[indices_dict["gpt_test"] + indices_dict["human_test"]]
                )[:, 1],
            )
        )

        other_datasets = [
            Dataset("normal", "data/other/lang8"),
            Dataset("normal", "data/other/pelic"),
            Dataset("normal", "data/other/gptzero/gpt"),
            Dataset("normal", "data/other/gptzero/human"),
        ]

        def get_data(generate_dataset_fn, best_features):
            davinci, ada, trigram, unigram = get_all_logprobs(
                generate_dataset_fn,
                trigram=trigram_model,
                tokenizer=tokenizer,
            )
            vector_map = {
                "davinci-logprobs": lambda file: davinci[file],
                "ada-logprobs": lambda file: ada[file],
                "trigram-logprobs": lambda file: trigram[file],
                "unigram-logprobs": lambda file: unigram[file],
            }
            exp_featurize = get_exp_featurize(best_features, vector_map)
            exp_data = generate_dataset_fn(exp_featurize)
            return exp_data

        def evaluate_on_dataset(
            model,
            best_features,
            curr_labels,
            generate_dataset_fn,
            dataset_name,
            train=False,
            to_normalize=True,
        ):
            data, mu, sigma = normalize(
                get_featurized_data(best_features, gpt_only=True),
                ret_mu_sigma=True,
            )

            t_data = generate_dataset_fn(t_featurize, verbose=True)
            exp_data = get_data(generate_dataset_fn, best_features)

            if to_normalize:
                curr_data = normalize(
                    np.concatenate([t_data, exp_data], axis=1), mu=mu, sigma=sigma
                )
            else:
                curr_data = np.concatenate([t_data, exp_data], axis=1)

            if train:
                indices = np.arange(len(curr_data))
                np.random.shuffle(indices)

                train_indices = indices[: math.floor(0.8 * len(indices))]
                test_indices = indices[math.floor(0.8 * len(indices)) :]

                curr_train_data = np.concatenate(
                    [
                        curr_data[train_indices],
                        data[indices_dict["gpt_train"] + indices_dict["human_train"]],
                    ],
                    axis=0,
                )
                curr_train_labels = np.concatenate(
                    [
                        curr_labels[train_indices],
                        labels[indices_dict["gpt_train"] + indices_dict["human_train"]],
                    ],
                    axis=0,
                )

                model = LogisticRegression()
                model.fit(curr_train_data, curr_train_labels)

                probs = model.predict_proba(curr_data[test_indices])[:, 1]
                results_table.append(
                    [
                        "Ghostbuster",
                        f"In-Domain ({dataset_name})",
                        *get_scores(curr_labels[test_indices], probs),
                    ]
                )
            else:
                probs = model.predict_proba(curr_data)[:, 1]
                results_table.append(
                    [
                        "Ghostbuster",
                        f"Out-Domain ({dataset_name})",
                        *get_scores(curr_labels, probs),
                    ]
                )

        for dataset in ["lang8"]:
            gen_fn = get_generate_dataset(Dataset("normal", f"data/other/{dataset}"))
            curr_labels = gen_fn(lambda _: 0)

            evaluate_on_dataset(
                model,
                best_features_map["best_features_three"],
                curr_labels,
                gen_fn,
                dataset,
            )

            exp_data = get_data(gen_fn, ["davinci-logprobs s-avg"])

            model_p = LogisticRegression()
            model_p.fit(
                exp_to_data["davinci-logprobs s-avg"][
                    indices_dict["gpt_train"] + indices_dict["human_train"]
                ],
                labels[indices_dict["gpt_train"] + indices_dict["human_train"]],
            )

            probs = model_p.predict_proba(exp_data)[:, 1]

            results_table.append(
                [
                    "Perplexity Only",
                    f"Out-Domain (lang8)",
                    *get_scores(curr_labels, probs),
                ]
            )

            # Evaluate roberta
            roberta_test = RobertaDataset(
                gen_fn(lambda file: open(file).read()),
                gen_fn(lambda _: 0),
            )

            roberta_test_loader = torch.utils.data.DataLoader(
                roberta_test, batch_size=1, shuffle=False
            )

            roberta_model.eval()

            roberta_probs = []
            with torch.no_grad():
                for batch in tqdm.tqdm(roberta_test_loader):
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    outputs = roberta_model(**inputs)
                    roberta_probs.append(
                        float(F.softmax(outputs.logits, dim=1)[0][1].item())
                    )

            results_table.append(
                [
                    "RoBERTa",
                    f"Out-Domain ({dataset})",
                    *get_scores(gen_fn(lambda _: 0), np.array(roberta_probs)),
                ]
            )

        gpt_zero = get_generate_dataset(
            Dataset("normal", f"data/other/gptzero/human"),
            Dataset("normal", f"data/other/gptzero/gpt"),
        )
        curr_labels = np.array([0] * 50 + [1] * 50)

        evaluate_on_dataset(
            model,
            best_features_map["best_features_three"],
            curr_labels,
            gpt_zero,
            "gptzero",
        )

        evaluate_on_dataset(
            model,
            best_features_map["best_features_three"],
            [1] * 100,
            get_generate_dataset(Dataset("normal", "data/other/undetectable")),
            "undetectable",
        )

        gen_ets = get_generate_dataset(Dataset("normal", f"data/other/ets"))
        curr_labels = gen_ets(lambda _: 0)

        evaluate_on_dataset(
            model,
            best_features_map["best_features_three"],
            curr_labels,
            gen_ets,
            "ets",
        )

        evaluate_on_dataset(
            model,
            best_features_map["best_features_three"],
            curr_labels,
            gen_ets,
            "ets",
            train=True,
        )

        exp_data = get_data(gen_ets, ["davinci-logprobs s-avg"])

        model = LogisticRegression()
        model.fit(
            exp_to_data["davinci-logprobs s-avg"][
                indices_dict["gpt_train"] + indices_dict["human_train"]
            ],
            labels[indices_dict["gpt_train"] + indices_dict["human_train"]],
        )

        probs = model.predict_proba(exp_data)[:, 1]

        results_table.append(
            [
                "Perplexity Only",
                f"Out-Domain (ets)",
                *get_scores(curr_labels, probs),
            ]
        )

        roberta_test = RobertaDataset(
            gen_ets(lambda file: open(file).read()),
            gen_ets(lambda _: 0),
        )
        roberta_test_loader = torch.utils.data.DataLoader(
            roberta_test, batch_size=1, shuffle=False
        )

        roberta_model.eval()
        roberta_probs = []
        with torch.no_grad():
            for batch in tqdm.tqdm(roberta_test_loader):
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = roberta_model(**inputs)
                roberta_probs.append(
                    float(F.softmax(outputs.logits, dim=1)[0][1].item())
                )

        results_table.append(
            [
                "RoBERTa",
                f"Out-Domain (ets)",
                *get_scores(gen_ets(lambda _: 0), np.array(roberta_probs)),
            ]
        )

    if args.toefl:

        def get_data(generate_dataset_fn, best_features):
            davinci, ada, trigram, unigram = get_all_logprobs(
                generate_dataset_fn,
                trigram=trigram_model,
                tokenizer=tokenizer,
            )
            vector_map = {
                "davinci-logprobs": lambda file: davinci[file],
                "ada-logprobs": lambda file: ada[file],
                "trigram-logprobs": lambda file: trigram[file],
                "unigram-logprobs": lambda file: unigram[file],
            }
            exp_featurize = get_exp_featurize(best_features, vector_map)
            exp_data = generate_dataset_fn(exp_featurize)

            t_data = generate_dataset_fn(t_featurize, verbose=True)

            return np.concatenate([t_data, exp_data], axis=1)

        # Evaluate on data contained in data/other/toefl91
        data = get_featurized_data(best_features_map["best_features_three"])
        data, mu, sigma = normalize(data, ret_mu_sigma=True)

        model = LogisticRegression()
        model.fit(
            data[indices_dict["gpt_train"] + indices_dict["human_train"]],
            labels[indices_dict["gpt_train"] + indices_dict["human_train"]],
        )
        print(
            f"Model F1: {f1_score(labels[indices_dict['gpt_test'] + indices_dict['human_test']], model.predict(data[indices_dict['gpt_test'] + indices_dict['human_test']]))}"
        )

        toefl = get_generate_dataset(Dataset("normal", "data/other/toefl91"))
        toefl_labels = toefl(lambda _: 0)
        toefl_data = get_data(toefl, best_features_map["best_features_three"])
        toefl_data = normalize(toefl_data, mu=mu, sigma=sigma)

        results_table.append(
            [
                "Ghostbuster",
                f"Out-Domain (toefl)",
                accuracy_score(toefl_labels, model.predict(toefl_data)),
            ]
        )

        # Do with perplexity only
        perplxity_data = get_featurized_data(["davinci-logprobs s-avg"], gpt_only=True)

        perplexity_model = LogisticRegression()
        perplexity_model.fit(
            perplxity_data[indices_dict["gpt_train"] + indices_dict["human_train"]],
            labels[indices_dict["gpt_train"] + indices_dict["human_train"]],
        )

        toefl_data = get_data(toefl, ["davinci-logprobs s-avg"])

        results_table.append(
            [
                "Perplexity Only",
                f"Out-Domain (toefl)",
                accuracy_score(toefl_labels, perplexity_model.predict(toefl_data)),
            ]
        )

        # Do RoBERTa
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            f"roberta/models/roberta_gpt", num_labels=2
        )
        roberta_model.to(device)

        roberta_predictions = []
        with torch.no_grad():
            for file in tqdm.tqdm(toefl(lambda file: file)):
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
                roberta_predictions.append(
                    float(F.softmax(outputs.logits, dim=1)[0][1].item())
                )

        results_table.append(
            [
                "RoBERTa",
                f"Out-Domain (toefl)",
                accuracy_score(toefl_labels, np.array(roberta_predictions) > 0.5),
            ]
        )

        results_table.append(["GPT Zero", f"Out-Domain (toefl)", 0.923077])

        results_table.append(["DetectGPT", f"Out-Domain (toefl)", 0.6373626373626373])

    if args.ghostbuster_vary_training_data:
        exp_to_data_three = pickle.load(open("symbolic_data_gpt", "rb"))

        train_indices = indices_dict["gpt_train"] + indices_dict["human_train"]
        test_indices = indices_dict["gpt_test"] + indices_dict["human_test"]
        np.random.shuffle(train_indices)

        claude_test_indices = (
            list(indices_dict["claude_test"]) + indices_dict["human_test"]
        )

        scores = []
        train_sizes = [int(125 * (2**i)) for i in range(6)] + [len(train_indices)]
        print(train_sizes)

        for size in tqdm.tqdm(train_sizes):
            print(f"Now running size: {size}")

            curr_train_indices = train_indices[:size]
            curr_best_features = select_features(
                exp_to_data_three,
                labels,
                verbose=True,
                to_normalize=True,
                indices=curr_train_indices,
            )
            data = normalize(get_featurized_data(curr_best_features))

            curr_score_vec = []
            print(data[curr_train_indices].shape)

            model = LogisticRegression()
            model.fit(data[curr_train_indices], labels[curr_train_indices])

            curr_score_vec.append(
                f1_score(labels[test_indices], model.predict(data[test_indices]))
            )

            curr_score_vec.append(
                f1_score(
                    labels[claude_test_indices],
                    model.predict(data[claude_test_indices]),
                )
            )

            for test_domain in domains:
                domain_train_indices = []

                for train_domain in domains:
                    if train_domain == test_domain:
                        continue

                    domain_train_indices += (
                        indices_dict[f"gpt_{train_domain}_train"]
                        + indices_dict[f"human_{train_domain}_train"]
                    )

                domain_train_indices = np.intersect1d(
                    domain_train_indices, curr_train_indices
                )

                domain_test_indices = (
                    indices_dict[f"gpt_{test_domain}_test"]
                    + indices_dict[f"human_{test_domain}_test"]
                )

                model = LogisticRegression()
                model.fit(data[domain_train_indices], labels[domain_train_indices])

                curr_score_vec.append(
                    f1_score(
                        labels[domain_test_indices],
                        model.predict(data[domain_test_indices]),
                    )
                )
            scores.append(curr_score_vec)

        scores = np.array(scores)
        np.save("results/training_size.npy", scores)

        plt.plot(train_sizes, scores[:, 0], label="In-Domain")
        plt.plot(train_sizes, scores[:, 1], label="Out-Domain (Claude)")
        plt.plot(train_sizes, scores[:, 2], label="Out-Domain (WP)")
        plt.plot(train_sizes, scores[:, 3], label="Out-Domain (Reuter)")
        plt.plot(train_sizes, scores[:, 4], label="Out-Domain (Essay)")

        plt.xlabel("Training Size (# of Documents)")
        plt.ylabel("F1 Score")

        plt.legend()
        plt.savefig("results/training_size.png")

    if args.ghostbuster_vary_document_size:
        token_sizes = [10, 25, 50, 100, 250, 500, 1000]
        scores = []

        train_indices = indices_dict["gpt_train"] + indices_dict["human_train"]
        test_indices = indices_dict["gpt_test"] + indices_dict["human_test"]
        claude_test_indices = (
            list(indices_dict["claude_test"]) + indices_dict["human_test"]
        )

        data = get_featurized_data(best_features_map["best_features_three"])
        data, mu, sigma = normalize(data, ret_mu_sigma=True)

        for num_tokens in tqdm.tqdm(token_sizes):
            print(f"Now running size: {num_tokens}")

            curr_t_data = generate_dataset_fn(
                lambda file: t_featurize(file, num_tokens=num_tokens), verbose=True
            )
            davinci, ada, trigram, unigram = get_all_logprobs(
                generate_dataset_fn,
                trigram=trigram_model,
                tokenizer=tokenizer,
                num_tokens=num_tokens,
            )

            vector_map = {
                "davinci-logprobs": lambda file: davinci[file],
                "ada-logprobs": lambda file: ada[file],
                "trigram-logprobs": lambda file: trigram[file],
                "unigram-logprobs": lambda file: unigram[file],
            }
            exp_featurize = get_exp_featurize(
                best_features_map["best_features_three"], vector_map
            )
            curr_exp_data = generate_dataset_fn(exp_featurize)
            curr_data = np.concatenate([curr_t_data, curr_exp_data], axis=1)
            curr_data = normalize(curr_data, mu=mu, sigma=sigma)

            curr_score_vec = []
            print(data.shape)

            model = LogisticRegression()
            model.fit(data[train_indices], labels[train_indices])

            curr_score_vec.append(
                f1_score(labels[test_indices], model.predict(curr_data[test_indices]))
            )

            curr_score_vec.append(
                f1_score(
                    labels[claude_test_indices],
                    model.predict(curr_data[claude_test_indices]),
                )
            )

            for test_domain in domains:
                domain_train_indices = []

                for train_domain in domains:
                    if train_domain == test_domain:
                        continue

                    domain_train_indices += (
                        indices_dict[f"gpt_{train_domain}_train"]
                        + indices_dict[f"human_{train_domain}_train"]
                    )

                domain_train_indices = np.intersect1d(
                    domain_train_indices, train_indices
                )

                domain_test_indices = (
                    indices_dict[f"gpt_{test_domain}_test"]
                    + indices_dict[f"human_{test_domain}_test"]
                )

                model = LogisticRegression()
                model.fit(data[domain_train_indices], labels[domain_train_indices])

                curr_score_vec.append(
                    f1_score(
                        labels[domain_test_indices],
                        model.predict(curr_data[domain_test_indices]),
                    )
                )

            scores.append(curr_score_vec)

            print(curr_score_vec)

        scores = np.array(scores)
        np.save("results/document_size.npy", scores)

        plt.plot(token_sizes, scores[:, 0], label="In-Domain")
        plt.plot(token_sizes, scores[:, 1], label="Out-Domain (Claude)")
        plt.plot(token_sizes, scores[:, 2], label="Out-Domain (WP)")
        plt.plot(token_sizes, scores[:, 3], label="Out-Domain (Reuter)")
        plt.plot(token_sizes, scores[:, 4], label="Out-Domain (Essay)")

        plt.xlabel("Document Size (# of Tokens)")
        plt.ylabel("F1 Score")

        plt.legend()

        plt.savefig("results/document_size.png")

    if args.hyperparameter_search:
        data = normalize(get_featurized_data(best_features_map["best_features_three"]))

        param_grid = {
            "C": [
                0.01,
                0.1,
                0.125,
                0.25,
                0.375,
                0.5,
                0.675,
                0.75,
                0.875,
                1,
                2,
                4,
                8,
                10,
            ],
            "penalty": ["l1", "l2", "elasticnet", None],
        }

        model = LogisticRegression()
        grid_search = GridSearchCV(
            model, param_grid=param_grid, scoring="roc_auc", cv=5, verbose=1
        )

        grid_search.fit(data[train], labels[train])
        print(grid_search.best_params_)

        model = LogisticRegression(
            C=grid_search.best_params_["C"],
            penalty=grid_search.best_params_["penalty"],
        )
        model.fit(data[train], labels[train])

        probs = model.predict_proba(data[test])[:, 1]
        print(get_scores(labels[test], probs))

    if args.perturb:
        data = get_featurized_data(best_features_map["best_features_three"])
        data, mu, sigma = normalize(data, ret_mu_sigma=True)

        model = LogisticRegression()
        model.fit(
            data[indices_dict["gpt_train"] + indices_dict["human_train"]],
            labels[indices_dict["gpt_train"] + indices_dict["human_train"]],
        )

        with open("data/perturb/labels.txt") as f:
            perturb_labels = np.array([int(i) for i in f.read().split("\n")])

        def get_data(generate_dataset_fn, best_features):
            t_data = generate_dataset_fn(t_featurize, verbose=False)

            davinci, ada, trigram, unigram = get_all_logprobs(
                generate_dataset_fn,
                trigram=trigram_model,
                tokenizer=tokenizer,
                verbose=False,
            )
            vector_map = {
                "davinci-logprobs": lambda file: davinci[file],
                "ada-logprobs": lambda file: ada[file],
                "trigram-logprobs": lambda file: trigram[file],
                "unigram-logprobs": lambda file: unigram[file],
            }
            exp_featurize = get_exp_featurize(best_features, vector_map)
            exp_data = generate_dataset_fn(exp_featurize, verbose=False)

            return np.concatenate([t_data, exp_data], axis=1)

        def get_perturb_data(perturb_names, perturb_sizes, save_file):
            data = defaultdict(list)

            for perturb_type in tqdm.tqdm(perturb_names):
                for n in perturb_sizes:
                    gen_fn = get_generate_dataset(
                        Dataset("normal", f"data/perturb/{perturb_type}/{n}")
                    )
                    curr_labels = gen_fn(
                        lambda file: perturb_labels[
                            int(os.path.basename(file).split(".")[0])
                        ]
                    )

                    curr_data = get_data(
                        gen_fn, best_features_map["best_features_three"]
                    )
                    curr_data = (curr_data - mu) / sigma
                    probs = model.predict_proba(curr_data)[:, 1]

                    results_table.append(
                        [
                            "Ghostbuster",
                            f"Out-Domain ({perturb_type}, {n})",
                            *get_scores(curr_labels, probs),
                        ]
                    )

                    _, f1, _ = get_scores(curr_labels, probs)
                    data[perturb_type].append(f1)

            np.save(save_file, data)
            return data

        perturb_char_data = get_perturb_data(
            perturb_char_names, perturb_char_sizes, "results/perturb_char.npy"
        )

        for perturb_type in perturb_char_names:
            plt.plot(
                perturb_char_sizes,
                perturb_char_data[perturb_type],
                label=perturb_type,
            )

        plt.xlabel("Number of Perturbations")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig("results/perturb_char.png")

        plt.clf()

        perturb_sent_data = get_perturb_data(
            perturb_sent_names, perturb_sent_sizes, "results/perturb_sent.npy"
        )

        for perturb_type in perturb_sent_names:
            plt.plot(
                perturb_sent_sizes,
                perturb_sent_data[perturb_type],
                label=perturb_type,
            )

        plt.xlabel("Number of Perturbations")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig("results/perturb_sent.png")

    if args.calibration:

        def calculate_ece(y_true, y_probs, n_bins=10):
            """Compute ECE"""
            bin_lowers = np.linspace(0.0, 1.0 - 1.0 / n_bins, n_bins)
            bin_uppers = np.linspace(1.0 / n_bins, 1.0, n_bins)

            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = np.logical_and(bin_lower <= y_probs, y_probs < bin_upper)
                prop_in_bin = np.mean(in_bin)
                if prop_in_bin > 0:
                    y_true_bin = y_true[in_bin]
                    avg_confidence_in_bin = np.mean(y_probs[in_bin])
                    avg_accuracy_in_bin = np.mean(y_true_bin)
                    ece += (
                        np.abs(avg_accuracy_in_bin - avg_confidence_in_bin)
                        * prop_in_bin
                    )

            return ece

        def train_ghostbuster_ece(data, train, test, domain):
            model = LogisticRegression()
            model.fit(data[train], labels[train])
            probs = model.predict_proba(data[test])[:, 1]

            return [calculate_ece(labels[test], probs)]

        def train_ghostbuster_calibrated_ece(data, train, test, domain):
            clf = LogisticRegression()
            model = CalibratedClassifierCV(clf, method="isotonic", cv=5)
            model.fit(data[train], labels[train])
            probs = model.predict_proba(data[test])[:, 1]

            return [calculate_ece(labels[test], probs)]

        run_experiment(
            best_features_map["best_features_three"],
            "Ghostbuster (Uncalibrated)",
            train_ghostbuster_ece,
        )

        run_experiment(
            best_features_map["best_features_three"],
            "Ghostbuster (Calibrated)",
            train_ghostbuster_calibrated_ece,
        )

    if len(results_table) > 1:
        # Write data to output csv file
        with open(args.output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerows(results_table)

        print(f"Saved results to {args.output_file}")
