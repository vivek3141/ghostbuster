from nltk.util import ngrams
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

import tqdm
import numpy as np
import tiktoken
import dill as pickle

from utils.featurize import *
from utils.n_gram import *

from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression


vec_functions = {
    "v-add": lambda a, b: a + b,
    "v-sub": lambda a, b: a - b,
    "v-mul": lambda a, b: a * b,
    "v-div": lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=(b != 0), casting='unsafe'),
    "v->": lambda a, b: a > b,
    "v-<": lambda a, b: a < b
}

scalar_functions = {
    "s-max": max,
    "s-min": min,
    "s-avg": lambda x: sum(x) / len(x),
    "s-avg-top-25": lambda x: sum(sorted(x, reverse=True)[:25]) / len(sorted(x, reverse=True)[:25]),
    "s-len": len,
    "s-var": np.var,
    "s-l2": np.linalg.norm
}

vectors = ["davinci-logprobs", "ada-logprobs",
           "trigram-logprobs", "unigram-logprobs"]

# Get vec_combinations
vec_combinations = []
vector_names = list(vectors)

for vec1 in range(len(vectors)):
    for vec2 in range(vec1):
        for func in vec_functions:
            if func != "v-div":
                vec_combinations.append(
                    f"{vector_names[vec1]} {func} {vector_names[vec2]}")

for vec1 in vectors:
    for vec2 in vectors:
        if vec1 != vec2:
            vec_combinations.append(f"{vec1} v-div {vec2}")


def get_words(exp):
    """
    Splits up expression into words, to be individually processed
    """
    return exp.split(" ")


def backtrack_functions(prev="", depth=0, max_depth=2):
    """
    Backtrack all possible features.
    """
    if depth >= max_depth:
        return []

    all_funcs = []
    prev_word = get_words(prev)[-1]

    if prev != "":
        for func in scalar_functions:
            all_funcs.append(f"{prev} {func}")

        for vec in vectors:
            for func in vec_functions:
                all_funcs += backtrack_functions(
                    prev + f" {func} {vec}",
                    depth + 1,
                    max_depth
                )
    else:
        for func in scalar_functions:
            for vec in vectors:
                all_funcs.append(f"{vec} {func}")

        for comb in vec_combinations:
            if get_words(comb)[0] != prev_word:
                all_funcs += backtrack_functions(
                    comb,
                    depth + 1,
                    max_depth
                )

    return all_funcs


def train_trigram(verbose=True, return_tokenizer=False):
    """
    Trains and returns a trigram model on the brown corpus
    """

    enc = tiktoken.encoding_for_model("davinci")
    tokenizer = enc.encode
    vocab_size = enc.n_vocab

    # We use the brown corpus to train the n-gram model
    sentences = brown.sents()

    if verbose:
        print("Tokenizing corpus...")
    tokenized_corpus = []
    for sentence in tqdm.tqdm(sentences):
        tokens = tokenizer(' '.join(sentence))
        tokenized_corpus += tokens

    if verbose:
        print("\nTraining n-gram model...")

    if return_tokenizer:
        return TrigramBackoff(tokenized_corpus), tokenizer
    else:
        return TrigramBackoff(tokenized_corpus)


def get_all_logprobs(generate_dataset, preprocess=lambda x: x, verbose=True,
                     trigram=None, tokenizer=None):
    if trigram is None:
        trigram, tokenizer = train_trigram(
            verbose=verbose, return_tokenizer=True)

    davinci_logprobs, ada_logprobs = {}, {}
    trigram_logprobs, unigram_logprobs = {}, {}

    if verbose:
        print("Loading logprobs into memory")

    file_names = generate_dataset(lambda file: file)
    to_iter = tqdm.tqdm(file_names) if verbose else file_names

    for file in to_iter:
        with open(file, "r") as f:
            doc = preprocess(f.read())
        davinci_logprobs[file] = get_logprobs(
            convert_file_to_logprob_file(file, "davinci")
        )
        ada_logprobs[file] = get_logprobs(
            convert_file_to_logprob_file(file, "ada")
        )
        trigram_logprobs[file] = score_ngram(doc, trigram, tokenizer, n=3)
        unigram_logprobs[file] = score_ngram(doc, trigram.base, tokenizer, n=1)

    return davinci_logprobs, ada_logprobs, trigram_logprobs, unigram_logprobs


def generate_symbolic_data(generate_dataset, preprocess=lambda x: x,
                           max_depth=2, output_file="symbolic_data", verbose=True):
    """
    Brute forces and generates symbolic data from a dataset of text files.
    """
    davinci_logprobs, ada_logprobs, trigram_logprobs, unigram_logprobs = get_all_logprobs(
        generate_dataset, preprocess=preprocess, verbose=verbose)

    vector_map = {
        "davinci-logprobs": lambda file: davinci_logprobs[file],
        "ada-logprobs": lambda file: ada_logprobs[file],
        "trigram-logprobs": lambda file: trigram_logprobs[file],
        "unigram-logprobs": lambda file: unigram_logprobs[file]
    }

    all_funcs = backtrack_functions(max_depth=max_depth)

    if verbose:
        print(f"\nTotal # of Features: {len(all_funcs)}.")
        print("Sampling 5 features:")
        for i in range(5):
            print(all_funcs[np.random.randint(0, len(all_funcs))])
        print("\nGenerating datasets...")

    def calc_features(file, exp):
        exp_tokens = get_words(exp)
        curr = vector_map[exp_tokens[0]](file)

        for i in range(1, len(exp_tokens)):
            if exp_tokens[i] in vec_functions:
                next_vec = vector_map[exp_tokens[i+1]](file)
                curr = vec_functions[exp_tokens[i]](curr, next_vec)
            elif exp_tokens[i] in scalar_functions:
                return scalar_functions[exp_tokens[i]](curr)

    exp_to_data = {}
    for exp in tqdm.tqdm(all_funcs):
        exp_to_data[exp] = generate_dataset(
            lambda file: calc_features(file, exp)
        ).reshape(-1, 1)

    pickle.dump(exp_to_data, open(output_file, "wb"))
