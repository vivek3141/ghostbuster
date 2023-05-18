import numpy as np


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


def normalize(data, mu=None, sigma=None, ret_mu_sigma=False):
    """
    Normalizes data, where data is a matrix where the first dimension is the number of examples
    """
    if mu is None:
        mu = np.mean(data.T, axis=1)
    if std is None:
        raw_std = np.std(data.T, axis=1)
        std = np.ones_like(raw_std)
        std[raw_std != 0] = raw_std[raw_std != 0]

    if ret_mu_sigma:
        return (data - mu) / std, mu, std
    else:
        return (data - mu) / std
