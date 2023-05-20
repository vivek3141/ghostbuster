import numpy as np
import os
from nltk import ngrams


def get_logprobs(file):
    """
    Returns a vector containing all the logprobs from a given logprobs file
    """
    logprobs = []

    with open(file) as f:
        for line in f.read().strip().split("\n"):
            line = line.split(" ")
            logprobs.append(np.exp(-float(line[1])))

    return np.array(logprobs)


def get_tokens(file):
    """
    Returns a list of all tokens from a given logprobs file
    """
    with open(file) as f:
        tokens = list(map(lambda x: x.split(
            " ")[0], f.read().strip().split("\n")))
    return tokens


def get_token_len(file):
    """
    Returns a vector of word lengths, in tokens
    """
    tokens = get_tokens(file)
    tokens_len = []
    curr = 0

    for token in tokens:
        if token[0] == "Ġ":
            tokens_len.append(curr)
            curr = 1
        else:
            curr += 1

    return np.array(tokens_len)


def get_diff(file1, file2):
    """
    Returns difference in logprobs bewteen file1 and file2
    """
    return get_logprobs(file1) - get_logprobs(file2)


def convolve(X, window=100):
    """
    Returns a vector of running average with window size
    """
    ret = []
    for i in range(len(X) - window):
        ret.append(np.mean(X[i:i+window]))
    return np.array(ret)


def score_ngram(file, model, tokenizer, strip_first=False):
    """
    Returns vector of ngram probabilities given document, model and tokenizer
    """
    scores = []

    with open(file) as f:
        doc = f.read().strip()
        if strip_first:
            doc = doc[doc.index("\n") + 1:]
        doc = " ".join(doc.split()[:1000])
        for i in ngrams([50256, 50256] + tokenizer(doc), 3):
            scores.append(model.n_gram_probability(i))

    return np.array(scores)


def normalize(data, mu=None, sigma=None, ret_mu_sigma=False):
    """
    Normalizes data, where data is a matrix where the first dimension is the number of examples
    """
    if mu is None:
        mu = np.mean(data.T, axis=1)
    if sigma is None:
        raw_std = np.std(data.T, axis=1)
        sigma = np.ones_like(raw_std)
        sigma[raw_std != 0] = raw_std[raw_std != 0]

    if ret_mu_sigma:
        return (data - mu) / sigma, mu, sigma
    else:
        return (data - mu) / sigma


def convert_file_to_logprob_file(file_name, model):
    """
    Removes the extension of file_name, then goes to the logprobs folder of the current directory,
    and appends a -{model}.txt to it.
    Example: convert_file_to_logprob_file("data/test.txt", "davinci") = "data/logprobs/test-davinci.txt"
    """
    directory = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)

    file_name_without_ext = os.path.splitext(base_name)[0]
    logprob_directory = os.path.join(directory, "logprobs")

    logprob_file_name = f"{file_name_without_ext}-{model}.txt"
    logprob_file_path = os.path.join(logprob_directory, logprob_file_name)

    return logprob_file_path


def t_featurize(file):
    davinci_file = convert_file_to_logprob_file(file, "davinci")
    ada_file = convert_file_to_logprob_file(file, "ada")

    X = []

    davinci_logprobs = get_logprobs(davinci_file)
    outliers = []
    for logprob in davinci_logprobs:
        if logprob > 10 and len(outliers) < 50:
            outliers.append(logprob)

    X.append(len(outliers))
    outliers += [0] * (50 - len(outliers))
    X.append(np.mean(outliers[:25]))
    X.append(np.mean(outliers[25:]))

    diffs = sorted(get_diff(davinci_file, ada_file), reverse=True)
    diffs += [0] * (50 - min(50, len(diffs)))
    X.append(np.mean(diffs[:25]))
    X.append(np.mean(diffs[25:]))

    token_len = sorted(get_token_len(davinci_file), reverse=True)
    token_len += [0] * (50 - min(50, len(token_len)))
    X.append(np.mean(token_len[:25]))
    X.append(np.mean(token_len[25:]))

    return X
