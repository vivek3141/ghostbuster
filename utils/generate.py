import math
import os
import tqdm
import openai

from utils import write_logprobs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


def round_up(x, base=50):
    return int(math.ceil(x / 50)) * 50


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def generate_documents(output_dir, prompts, verbose=True, force_regenerate=False):
    if not os.path.exists(f"{output_dir}/logprobs"):
        os.mkdir(f"{output_dir}/logprobs")

    if verbose:
        print("Generating Articles...")

    for idx, prompt in (enumerate(tqdm.tqdm(prompts)) if verbose else enumerate(prompts)):
        if os.path.exists(f"{output_dir}/{idx}.txt") and not force_regenerate:
            continue

        response = openai_backoff(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                            "content": prompt,
                }
            ],
        )
        reply = response["choices"][0]["message"]["content"].strip()

        with open(f"{output_dir}/{idx}.txt", "w") as f:
            f.write(f"{reply}")

    if verbose:
        print("Writing logprobs...")

    for idx, prompt in (enumerate(tqdm.tqdm(prompts)) if verbose else enumerate(prompts)):

        with open(f"{output_dir}/{idx}.txt") as f:
            doc = f.read().strip()

        if not os.path.exists(f"{output_dir}/logprobs/{idx}-davinci.txt") and not force_regenerate:
            write_logprobs(
                doc, f"{output_dir}/logprobs/{idx}-davinci.txt", "davinci"
            )
        if not os.path.exists(f"{output_dir}/logprobs/{idx}-ada.txt") and not force_regenerate:
            write_logprobs(
                doc, f"{output_dir}/logprobs/{idx}-curie.txt", "ada"
            )
