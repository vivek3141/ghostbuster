import numpy as np
import os
import tqdm
from nltk import ngrams
from utils.score import k_fold_score


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


def score_ngram(doc, model, tokenizer, n=3, strip_first=False):
    """
    Returns vector of ngram probabilities given document, model and tokenizer
    """
    scores = []
    doc = " ".join(doc.split()[:1000])
    for i in ngrams((n-1) * [50256] + tokenizer(doc), n):
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
    """
    Manually handcrafted features for classification.
    """
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


def select_features(exp_to_data, labels, verbose=True, to_normalize=True):
    if to_normalize:
        normalized_exp_to_data = {}
        for key in exp_to_data:
            normalized_exp_to_data[key] = normalize(exp_to_data[key])
    else:
        normalized_exp_to_data = exp_to_data

    def get_data(*exp):
        return np.concatenate(
            [normalized_exp_to_data[e] for e in exp],
            axis=1
        )

    assert sum(labels) == len(labels) // 2

    val_exp = list(exp_to_data.keys())
    curr = 0
    best_features = []
    i = 0

    while val_exp:
        best_score, best_exp = -1, ""

        for exp in (tqdm.tqdm(val_exp) if verbose else val_exp):
            score = k_fold_score(get_data(*best_features, exp), labels, k=5)

            if score > best_score:
                best_score = score
                best_exp = exp

        if verbose:
            print(f"Iteration {i}, Current Score: {curr}, \
                Best Feature: {best_exp}, New Score: {best_score}")

        if best_score <= curr:
            break
        else:
            best_features.append(best_exp)
            val_exp.remove(best_exp)
            curr = best_score

        i += 1

    return best_features
