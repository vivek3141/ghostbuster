import numpy as np
import tiktoken
import dill as pickle

from writing_prompts.data.load import generate_dataset as wp_generate_dataset
from reuter.data.load import generate_dataset as reuter_generate_dataset
from essay.data.load import generate_dataset as essay_generate_dataset

from utils.featurize import t_featurize, normalize
from utils.symbolic import get_words, vec_functions, scalar_functions, get_all_logprobs

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

np.random.seed(0)

WP_PATH = "writing_prompts"
REUTER_PATH = "reuter/data"
ESSAY_PATH = "essay"

best_features = open("model/best_features.txt").read().strip().split("\n")


def get_exp_featurize(vector_map, best_features):
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


# Generate WP Data
wp_symbolic_data = pickle.load(open(f"writing_prompts/symbolic_data", "rb"))
wp_t = wp_generate_dataset(t_featurize, base_dir=f"{WP_PATH}/", verbose=True)
wp_data = np.concatenate(
    [wp_symbolic_data[e]
     for e in best_features] + [wp_t], axis=1
)
wp_labels = wp_generate_dataset(
    lambda file: 1 if "gpt" in file else 0, base_dir=f"{WP_PATH}/")

# Generate Reuter Train Data
reuter_train_symbolic_data = pickle.load(
    open(f"reuter/symbolic_data", "rb"))
reuter_train_t = reuter_generate_dataset(
    t_featurize, "train", base_dir=f"{REUTER_PATH}/", verbose=True)
reuter_train_data = np.concatenate(
    [reuter_train_symbolic_data[e]
        for e in best_features] + [reuter_train_t], axis=1
)
reuter_train_labels = reuter_generate_dataset(
    lambda file: 1 if "gpt" in file else 0, "train", base_dir=f"{REUTER_PATH}/")

# Generate Reuter Test Data
reuter_test_data = reuter_generate_dataset(
    t_featurize, "test", base_dir=f"{REUTER_PATH}/")

davinci_logprobs, ada_logprobs, trigram_logprobs, unigram_logprobs = get_all_logprobs(
    lambda featurize: reuter_generate_dataset(
        featurize, "test", base_dir=f"{REUTER_PATH}/")
)
vector_map = {
    "davinci-logprobs": lambda file: davinci_logprobs[file],
    "ada-logprobs": lambda file: ada_logprobs[file],
    "trigram-logprobs": lambda file: trigram_logprobs[file],
    "unigram-logprobs": lambda file: unigram_logprobs[file]
}

reuter_t_test_data = reuter_generate_dataset(
    t_featurize, "test", base_dir=f"{REUTER_PATH}/")
reuter_exp_test_data = reuter_generate_dataset(
    get_exp_featurize(vector_map, best_features), "test",
    base_dir=f"{REUTER_PATH}/", verbose=True
)
reuter_test_data = np.concatenate(
    (reuter_t_test_data, reuter_exp_test_data),
    axis=1
)
reuter_test_labels = reuter_generate_dataset(
    lambda file: 1 if "gpt" in file else 0, "test", base_dir=f"{REUTER_PATH}/")

# Generate Essay Data
essay_symbolic_data = pickle.load(open(f"essay/symbolic_data", "rb"))
essay_t = essay_generate_dataset(
    t_featurize, base_dir=f"{ESSAY_PATH}/", verbose=True)
essay_data = np.concatenate(
    [essay_symbolic_data[e]
        for e in best_features] + [essay_t], axis=1
)
essay_labels = essay_generate_dataset(
    lambda file: 1 if "gpt" in file else 0, base_dir=f"{ESSAY_PATH}/")


# Combine data together
data, mu, sigma = normalize(np.concatenate(
    (wp_data, reuter_train_data, reuter_test_data, essay_data)), ret_mu_sigma=True)
labels = np.concatenate((wp_labels, reuter_train_labels,
                        reuter_test_labels, essay_labels))

# Generate split
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
train, test = indices[:int(0.8 * len(indices))
                      ], indices[int(0.8 * len(indices)):]

# Write train, test split to file
with open("model/train.txt", "w") as f:
    for i in train:
        f.write(str(i) + "\n")

with open("model/test.txt", "w") as f:
    for i in test:
        f.write(str(i) + "\n")

# Train model
model = LogisticRegression(C=10, penalty="l2", max_iter=10000)
model.fit(data[train], labels[train])

# Evaluate model
preds = model.predict(data[test])
print("F1 Score:", f1_score(labels[test], preds))
print("Accuracy:", accuracy_score(labels[test], preds))
print("ROC AUC Score:", roc_auc_score(labels[test], preds))

# Save model
pickle.dump(model, open("model/model", "wb"))
pickle.dump(mu, open("model/mu", "wb"))
pickle.dump(sigma, open("model/sigma", "wb"))
