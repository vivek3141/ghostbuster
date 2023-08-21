import argparse
import os
import tqdm
import openai
import math
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from tabulate import tabulate

from utils.featurize import normalize, t_featurize
from utils.symbolic import get_all_logprobs, get_words, train_trigram
from utils.symbolic import vec_functions, scalar_functions, get_exp_featurize
from utils.load import get_generate_dataset, Dataset


NUM_STORIES = 1000
NUM_ESSAYS = 2685

with open("model/best_features.txt") as f:
    best_features_all = f.read().strip().split("\n")

trigram_model, tokenizer = train_trigram(return_tokenizer=True)

wp_dataset = [
    Dataset("normal", "writing_prompts/data/gpt"),
    Dataset("normal", "writing_prompts/data/human"),
]

reuter_dataset_train = [
    Dataset("author", "reuter/data/human/train"),
    Dataset("author", "reuter/data/gpt/train"),
]
reuter_dataset_test = [
    Dataset("author", "reuter/data/human/test"),
    Dataset("author", "reuter/data/gpt/test"),
]

essay_dataset = [
    Dataset("normal", "essay/data/human"),
    Dataset("normal", "essay/data/gpt"),
]


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


def run_experiment(
    generate_dataset,
    best_features,
    model_names=["gpt"],
    train_split=None,
    test_split=None,
    train_split_size=0.9,
):
    data = get_featurized_data(generate_dataset, best_features)
    labels = generate_dataset(
        lambda file: 1 if any([name in file for name in model_names]) else 0
    )

    if train_split is None or test_split is None:
        indices = np.arange(len(data))
        np.random.shuffle(indices)

        train_split = indices[: math.floor(train_split_size * len(data))]
        test_split = indices[math.floor(train_split_size * len(data)) :]

    train_data = data[train_split]
    train_labels = labels[train_split]

    test_data = data[test_split]
    test_labels = labels[test_split]

    print("Training data shape:", train_data.shape)
    print("Testing data shape:", test_data.shape)

    # Normalize train, then apply same normalization to test
    train_data, mu, sigma = normalize(train_data, ret_mu_sigma=True)
    test_data = normalize(test_data, mu=mu, sigma=sigma)

    clf = LogisticRegression(C=10, penalty="l2", max_iter=10000)
    clf.fit(train_data, train_labels)

    return (
        f1_score(test_labels, clf.predict(test_data)),
        clf.score(test_data, test_labels),
        roc_auc_score(test_labels, clf.predict_proba(test_data)[:, 1]),
    )


# def gpt_generate_stories(args):
#     print("Generating stories...")

#     prompts = []
#     for i in range(NUM_STORIES):
#         with open(f"writing_prompts/data/human/{i}.txt") as f:
#             words = len(f.read().split(" "))

#         with open(f"writing_prompts/data/gpt/prompts/{i}.txt") as f:
#             prompt = f.read().strip()

#         prompts.append(
#             f"Write a story in {round_up(words, 50)} words to the prompt: {prompt}")

#     generate_documents(
#         "writing_prompts/data/gpt",
#         prompts,
#         verbose=True,
#         force_regenerate=args.force_regenerate
#     )


# def gpt_generate_essay(args):
#     print("Generating essays...")

#     prompts = []
#     for i in range(NUM_ESSAYS):
#         with open(f"essay/data/human/{i}.txt") as f:
#             words = len(f.read().split(" "))

#         if not args.force_regenerate:
#             if not os.path.exists(f"essay/data/gpt/prompts/{i}.txt"):
#                 raise Exception("Prompt directory not found!")

#             with open(f"essay/data/gpt/prompts/{i}.txt") as f:
#                 prompt = f.read().strip()

#         else:


#         prompts.append(
#             f"Write a story in {round_up(words, 50)} words to the prompt: {prompt}")

#     generate_documents(
#         "writing_prompts/data/gpt",
#         prompts,
#         verbose=True,
#         force_regenerate=args.force_regenerate
#     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_key", type=str, default="")
    parser.add_argument("--run_all", action="store_true")
    # parser.add_argument("--force_regenerate", action="store_true")
    # parser.add_argument("--generate_story", action="store_true")
    # parser.add_argument("--generate_reuter", action="store_true")
    # parser.add_argument("--generate_essay", action="store_true")
    parser.add_argument("--include_wp", action="store_true")
    parser.add_argument("--include_reuter", action="store_true")
    parser.add_argument("--include_essay", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--run_claude_model", action="store_true")
    args = parser.parse_args()

    if args.openai_key != "":
        openai.api_key = args.openai_key

    # # Generate GPT Text
    # if args.run_all or args.generate_story:
    #     generate_stories(args)

    # if args.run_all or args.generate_reuter:
    #     generate_reuter(args)

    # if args.run_all or args.generate_essay:
    #     generate_essay(args)

    result_table = [["Experiment Name", "F1", "Accuracy", "AUC"]]

    datasets = []
    if args.include_wp:
        datasets += wp_dataset

    generate_dataset_fn = get_generate_dataset(*datasets)

    results = run_experiment(
        generate_dataset=generate_dataset_fn, best_features=best_features_all
    )
    result_table.append(["GPT (in-domain)", *results])

    # if args.run_wp_model:
    #     print("Running WP Model...")

    #     with open("writing_prompts/data/train.txt") as f:
    #         train_split = list(map(int, f.read().strip().split("\n")))

    #     with open("writing_prompts/data/test.txt") as f:
    #         test_split = list(map(int, f.read().strip().split("\n")))

    #     with open("writing_prompts/best_features.txt") as f:
    #         best_features_wp = f.read().strip().split("\n")

    #     results = run_experiment(
    #         lambda featurize: generate_gpt_wp(
    #             featurize, split=train_split, base_dir="writing_prompts/"
    #         ),
    #         lambda featurize: generate_gpt_wp(
    #             featurize, split=test_split, base_dir="writing_prompts/"
    #         ),
    #         best_features_wp,
    #     )
    #     result_table.append(["GPT WP (in-domain)", *results])

    # if args.run_essay_model:
    #     print("Running Essay Model...")

    #     with open("essay/data/train.txt") as f:
    #         train_split = list(map(int, f.read().strip().split("\n")))

    #     with open("essay/data/test.txt") as f:
    #         test_split = list(map(int, f.read().strip().split("\n")))

    #     with open("essay/best_features.txt") as f:
    #         best_features_essay = f.read().strip().split("\n")

    #     results = run_experiment(
    #         lambda featurize: generate_gpt_essay(
    #             featurize, split=train_split, base_dir="essay/"
    #         ),
    #         lambda featurize: generate_gpt_essay(
    #             featurize, split=test_split, base_dir="essay/"
    #         ),
    #         best_features_essay,
    #     )
    #     result_table.append(["GPT Essay (in-domain)", *results])

    print(tabulate(result_table, headers="firstrow", tablefmt="grid"))
