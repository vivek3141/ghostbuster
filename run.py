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

from writing_prompts.data.load import generate_dataset as generate_gpt_wp
from writing_prompts.data.load import generate_dataset_claude as generate_claude_wp

from essay.data.load import generate_dataset as generate_gpt_essay
from essay.data.load import generate_dataset_claude as generate_claude_essay

from reuter.data.load import generate_dataset as generate_gpt_reuter
from reuter.data.load import generate_dataset_claude as generate_claude_reuter

from utils.generate import generate_documents, round_up, openai_backoff


NUM_STORIES = 1000
NUM_ESSAYS = 2685

with open("model/best_features.txt") as f:
    best_features_all = f.read().strip().split("\n")

trigram_model, tokenizer = train_trigram(return_tokenizer=True)


def get_featurized_data(generate_dataset_fn, best_features):
    t_data = generate_dataset_fn(t_featurize)

    davinci, ada, trigram, unigram = get_all_logprobs(
        generate_dataset_fn, trigram=trigram_model, tokenizer=tokenizer)

    vector_map = {
        "davinci-logprobs": lambda file: davinci[file],
        "ada-logprobs": lambda file: ada[file],
        "trigram-logprobs": lambda file: trigram[file],
        "unigram-logprobs": lambda file: unigram[file]
    }
    exp_featurize = get_exp_featurize(best_features, vector_map)
    exp_data = generate_dataset_fn(exp_featurize)

    return np.concatenate([t_data, exp_data], axis=1)


def run_experiment(generate_train_dataset, generate_test_dataset, best_features, model_name="gpt"):
    train_data = get_featurized_data(generate_train_dataset, best_features)
    train_labels = generate_train_dataset(lambda file: model_name in file)

    test_data = get_featurized_data(generate_test_dataset, best_features)
    test_labels = generate_test_dataset(lambda file: model_name in file)

    print(train_data.shape)
    print(test_data.shape)

    # Normalize train, then apply same normalization to test
    train_data, mu, sigma = normalize(train_data, ret_mu_sigma=True)
    test_data = normalize(test_data, mu=mu, sigma=sigma)

    clf = LogisticRegression(C=10, penalty='l2', max_iter=10000)
    clf.fit(train_data, train_labels)

    return f1_score(test_labels, clf.predict(test_data)), \
        clf.score(test_data, test_labels), \
        roc_auc_score(test_labels, clf.predict_proba(test_data)[:, 1])


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
    parser.add_argument("--run_wp_model", action="store_true")
    parser.add_argument("--run_essay_model", action="store_true")
    parser.add_argument("--run_reuter_model", action="store_true")
    parser.add_argument("--run_claude_model", action="store_true")
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

    if args.run_wp_model:
        print("Running WP Model...")

        with open("writing_prompts/data/train.txt") as f:
            train_split = list(map(int, f.read().strip().split("\n")))

        with open("writing_prompts/data/test.txt") as f:
            test_split = list(map(int, f.read().strip().split("\n")))

        with open("writing_prompts/best_features.txt") as f:
            best_features_wp = f.read().strip().split("\n")

        results = run_experiment(
            lambda featurize: generate_gpt_wp(
                featurize, split=train_split, base_dir="writing_prompts/"),
            lambda featurize: generate_gpt_wp(
                featurize, split=test_split, base_dir="writing_prompts/"),
            best_features_wp,
        )
        result_table.append(["GPT WP (in-domain)", *results])

    if args.run_essay_model:
        print("Running Essay Model...")

        with open("essay/data/train.txt") as f:
            train_split = list(map(int, f.read().strip().split("\n")))

        with open("essay/data/test.txt") as f:
            test_split = list(map(int, f.read().strip().split("\n")))

        with open("essay/best_features.txt") as f:
            best_features_essay = f.read().strip().split("\n")

        results = run_experiment(
            lambda featurize: generate_gpt_essay(
                featurize, split=train_split, base_dir="essay/"),
            lambda featurize: generate_gpt_essay(
                featurize, split=test_split, base_dir="essay/"),
            best_features_essay,
        )
        result_table.append(["GPT Essay (in-domain)", *results])

    print(tabulate(result_table, headers="firstrow", tablefmt="grid"))
