import tqdm
import numpy as np
import os

DIR_IGNORE = {"logprobs", "prompts", "headlines"}

def get_generate_dataset(dataset_type="normal", 
        base_dir=".", dir_splits=["human", "gpt"]):
    assert len(dir_splits) == 2

    def generate_dataset(featurize, split=None, verbose=False):
        data = []
        if split: split = set(split)
        
        if dataset_type == "normal":
            left_iter = [(dir_splits[0], i) for i in os.listdir(f"{base_dir}/{dir_splits[0]}") if i not in DIR_IGNORE]
            right_iter = [(dir_splits[1], i) for i in os.listdir(f"{base_dir}/{dir_splits[1]}") if i not in DIR_IGNORE]
            to_iter = tqdm.tqdm(left_iter + right_iter) if verbose else left_iter + right_iter

            for idx, (base, file) in enumerate(to_iter):
                if split and idx not in split:
                    continue
                data.append(featurize(f"{base_dir}/{base}/{file}"))

        return np.array(data)

    return generate_dataset

