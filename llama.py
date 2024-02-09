import os
import argparse

from utils.featurize import convert_file_to_logprob_file, get_logprobs
from utils.load import Dataset, get_generate_dataset
from utils.n_gram import TrigramBackoff
from utils.featurize import select_features, normalize
from utils.symbolic import vec_functions, scalar_functions

from transformers import AutoTokenizer
from collections import defaultdict

import math
import tqdm
import numpy as np
import dill as pickle

from nltk.util import ngrams
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

datasets = [
    Dataset("normal", "data/wp/human"),
    Dataset("normal", "data/wp/gpt"),
    Dataset("author", "data/reuter/human"),
    Dataset("author", "data/reuter/gpt"),
    Dataset("normal", "data/essay/human"),
    Dataset("normal", "data/essay/gpt"),
]

best_features = [
    "trigram-logprobs v-add unigram-logprobs v-> llama-logprobs s-var",
    "trigram-logprobs v-div unigram-logprobs v-div trigram-logprobs s-avg-top-25",
    "unigram-logprobs v-mul llama-logprobs s-avg",
    "trigram-logprobs v-mul unigram-logprobs v-div trigram-logprobs s-avg",
    "trigram-logprobs v-< unigram-logprobs v-mul llama-logprobs s-avg-top-25",
    "trigram-logprobs v-mul unigram-logprobs v-sub llama-logprobs s-min",
    "trigram-logprobs v-mul unigram-logprobs s-avg",
    "trigram-logprobs v-< unigram-logprobs v-sub llama-logprobs s-avg",
    "trigram-logprobs v-> unigram-logprobs v-add llama-logprobs s-avg",
    "trigram-logprobs v-div llama-logprobs v-div trigram-logprobs s-min",
]

models = ["gpt"]
domains = ["wp", "reuter", "essay"]
eval_domains = ["claude", "gpt_prompt1", "gpt_prompt2", "gpt_writing", "gpt_semantic"]


vectors = ["llama-logprobs", "unigram-logprobs", "trigram-logprobs"]

parser = argparse.ArgumentParser()
parser.add_argument("--feature_select", action="store_true")
parser.add_argument("--classify", action="store_true")

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

sentences = brown.sents()

tokenized_corpus = []
for sentence in tqdm.tqdm(sentences):
    tokens = tokenizer(" ".join(sentence))["input_ids"]
    tokenized_corpus += tokens

trigram = TrigramBackoff(tokenized_corpus)


vec_combinations = defaultdict(list)
for vec1 in range(len(vectors)):
    for vec2 in range(vec1):
        for func in vec_functions:
            if func != "v-div":
                vec_combinations[vectors[vec1]].append(f"{func} {vectors[vec2]}")

for vec1 in vectors:
    for vec2 in vectors:
        if vec1 != vec2:
            vec_combinations[vec1].append(f"v-div {vec2}")


def get_words(exp):
    """
    Splits up expression into words, to be individually processed
    """
    return exp.split(" ")


def backtrack_functions(
    max_depth=2,
):
    """
    Backtrack all possible features.
    """

    def helper(prev, depth):
        if depth >= max_depth:
            return []

        all_funcs = []
        prev_word = get_words(prev)[-1]

        for func in scalar_functions:
            all_funcs.append(f"{prev} {func}")

        for comb in vec_combinations[prev_word]:
            all_funcs += helper(f"{prev} {comb}", depth + 1)

        return all_funcs

    ret = []
    for vec in vectors:
        ret += helper(vec, 0)
    return ret


def score_ngram(doc, model, tokenizer, n=3):
    """
    Returns vector of ngram probabilities given document, model and tokenizer
    """
    scores = []
    tokens = (
        tokenizer(doc.strip())[1:] if n == 1 else (n - 2) * [2] + tokenizer(doc.strip())
    )

    for i in ngrams(tokens, n):
        scores.append(model.n_gram_probability(i))

    return np.array(scores)


def get_all_logprobs(
    generate_dataset,
    preprocess=lambda x: x.strip(),
    verbose=True,
    trigram=None,
    tokenizer=None,
    num_tokens=2047,
):
    llama_logprobs = {}
    trigram_logprobs, unigram_logprobs = {}, {}

    if verbose:
        print("Loading logprobs into memory")

    file_names = generate_dataset(lambda file: file, verbose=False)
    to_iter = tqdm.tqdm(file_names) if verbose else file_names

    for file in to_iter:
        if "logprobs" in file:
            continue

        with open(file, "r") as f:
            doc = preprocess(f.read())
        llama_logprobs[file] = get_logprobs(
            convert_file_to_logprob_file(file, "llama-7b")
        )[:num_tokens]
        trigram_logprobs[file] = score_ngram(doc, trigram, tokenizer, n=3)[:num_tokens]
        unigram_logprobs[file] = score_ngram(doc, trigram.base, tokenizer, n=1)[
            :num_tokens
        ]

    return llama_logprobs, trigram_logprobs, unigram_logprobs


all_funcs = backtrack_functions(max_depth=3)
np.random.seed(0)

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

generate_dataset_fn = get_generate_dataset(*datasets)
labels = generate_dataset_fn(
    lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
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

if args.feature_select:
    (
        llama_logprobs,
        trigram_logprobs,
        unigram_logprobs,
    ) = get_all_logprobs(
        generate_dataset_fn,
        verbose=True,
        tokenizer=lambda x: tokenizer(x)["input_ids"],
        trigram=trigram,
    )

    vector_map = {
        "llama-logprobs": lambda file: llama_logprobs[file],
        "trigram-logprobs": lambda file: trigram_logprobs[file],
        "unigram-logprobs": lambda file: unigram_logprobs[file],
    }

    def calc_features(file, exp):
        exp_tokens = get_words(exp)
        curr = vector_map[exp_tokens[0]](file)

        for i in range(1, len(exp_tokens)):
            if exp_tokens[i] in vec_functions:
                next_vec = vector_map[exp_tokens[i + 1]](file)
                curr = vec_functions[exp_tokens[i]](curr, next_vec)
            elif exp_tokens[i] in scalar_functions:
                return scalar_functions[exp_tokens[i]](curr)

    print("Preparing exp_to_data")
    exp_to_data = {}
    for exp in tqdm.tqdm(all_funcs):
        exp_to_data[exp] = generate_dataset_fn(
            lambda file: calc_features(file, exp)
        ).reshape(-1, 1)

    select_features(exp_to_data, labels, verbose=True, to_normalize=True, indices=train)

if args.classify:
    (
        llama_logprobs,
        trigram_logprobs,
        unigram_logprobs,
    ) = get_all_logprobs(
        generate_dataset_fn,
        verbose=True,
        tokenizer=lambda x: tokenizer(x)["input_ids"],
        trigram=trigram,
    )

    vector_map = {
        "llama-logprobs": lambda file: llama_logprobs[file],
        "trigram-logprobs": lambda file: trigram_logprobs[file],
        "unigram-logprobs": lambda file: unigram_logprobs[file],
    }

    def get_exp_featurize(best_features, vector_map):
        def calc_features(file, exp):
            exp_tokens = get_words(exp)
            curr = vector_map[exp_tokens[0]](file)

            for i in range(1, len(exp_tokens)):
                if exp_tokens[i] in vec_functions:
                    next_vec = vector_map[exp_tokens[i + 1]](file)
                    curr = vec_functions[exp_tokens[i]](curr, next_vec)
                elif exp_tokens[i] in scalar_functions:
                    return scalar_functions[exp_tokens[i]](curr)

        def exp_featurize(file):
            return np.array([calc_features(file, exp) for exp in best_features])

        return exp_featurize

    data = generate_dataset_fn(get_exp_featurize(best_features, vector_map))
    data = normalize(data)

    def train_llama(data, train, test):
        model = LogisticRegression()
        model.fit(data[train], labels[train])
        return f1_score(labels[test], model.predict(data[test]))

    print(
        f"In-Domain: {train_llama(data, indices_dict['gpt_train'] + indices_dict['human_train'], indices_dict['gpt_test'] + indices_dict['human_test'])}"
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

        print(
            f"Out-Domain ({test_domain}): {train_llama(data, train_indices, indices_dict[f'gpt_{test_domain}_test'] + indices_dict[f'human_{test_domain}_test'])}"
        )
