import numpy as np
import dill as pickle
import tqdm
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from utils.featurize import normalize, t_featurize
from writing_prompts.data.load import generate_dataset_author as generate_dataset

from utils.symbolic import get_words, vec_functions, scalar_functions, get_all_logprobs, train_trigram

accuracies, f1_scores, aucs = {}, {}, {}

authors = sorted(os.listdir("data/human_author"))
best_features = open("best_features.txt").read().strip().split("\n")


def get_exp_featurize(vector_map):
    def calc_features(file, exp):
        exp_tokens = get_words(exp)
        curr = vector_map[exp_tokens[0]](file)

        for i in range(1, len(exp_tokens)):
            if exp_tokens[i] in vec_functions:
                next_vec = vector_map[exp_tokens[i+1]](file)
                curr = vec_functions[exp_tokens[i]](curr, next_vec)
            elif exp_tokens[i] in scalar_functions:
                return scalar_functions[exp_tokens[i]](curr)

    def exp_featurize(file):
        return np.array([calc_features(file, exp) for exp in best_features])

    return exp_featurize


def get_split(author):
    davinci_logprobs, ada_logprobs, trigram_logprobs, unigram_logprobs = get_all_logprobs(
        lambda featurize: generate_dataset(featurize, author=author),
        tokenizer=tokenizer, trigram=trigram,
        verbose=False
    )
    vector_map = {
        "davinci-logprobs": lambda file: davinci_logprobs[file],
        "ada-logprobs": lambda file: ada_logprobs[file],
        "trigram-logprobs": lambda file: trigram_logprobs[file],
        "unigram-logprobs": lambda file: unigram_logprobs[file]
    }
    data = normalize(
        np.concatenate(
            [generate_dataset(t_featurize, author=author),
             generate_dataset(get_exp_featurize(vector_map), author=author)],
            axis=1
        )
    )
    labels = generate_dataset(
        lambda file: 1 if "gpt" in file else 0, author=author)

    assert len(data) == len(labels) and sum(labels) == len(labels) // 2

    return data, np.array(labels)


trigram, tokenizer = train_trigram(
    verbose=True, return_tokenizer=True)

for author in tqdm.tqdm(authors):
    X, y = get_split(author)

    indices = np.arange(200)
    np.random.shuffle(indices)
    train, test = np.split(indices, [100])

    train_data, train_labels = X[train], y[train]
    test_data, test_labels = X[test], y[test]

    model = LogisticRegression(random_state=0, max_iter=1000).fit(
        train_data, train_labels)

    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data)[:, 1]

    accuracies[author] = accuracy_score(test_labels, predictions)
    f1_scores[author] = f1_score(test_labels, predictions)
    aucs[author] = roc_auc_score(test_labels, probabilities)

# Print average accuracy, f1, and auroc
print(f"Average Accuracy: {np.mean(list(accuracies.values()))}")
print(f"Average F1 Score: {np.mean(list(f1_scores.values()))}")
print(f"Average AUROC: {np.mean(list(aucs.values()))}")
