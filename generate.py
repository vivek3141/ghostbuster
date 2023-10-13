import argparse
import openai
import re
import tqdm
import os
import math
import nltk
import numpy as np

from datasets import load_dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from utils.generate import generate_documents
from utils.write_logprobs import write_logprobs
from utils.symbolic import convert_file_to_logprob_file
from utils.load import Dataset, get_generate_dataset

datasets = [
    Dataset("normal", "data/wp/human"),
    Dataset("normal", "data/wp/gpt"),
    Dataset("author", "data/reuter/human"),
    Dataset("author", "data/reuter/gpt"),
    Dataset("normal", "data/essay/human"),
    Dataset("normal", "data/essay/gpt"),
]
generate_dataset_fn = get_generate_dataset(*datasets)

prompt_types = ["gpt", "gpt_prompt1", "gpt_prompt2", "gpt_writing", "gpt_semantic"]
html_replacements = [
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&apos;", "'"),
]


def html_replace(text):
    for replacement in html_replacements:
        text = text.replace(replacement[0], replacement[1])
    return text


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def round_to_100(n):
    return int(round(n / 100.0)) * 100


def get_wp_prompts(words, prompt):
    return [
        f'Write a story in {words} words to the prompt "{prompt}."',
        f'You are an author, who is writing a story in response to the prompt "{prompt}." What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word story on the following prompt: "{prompt}." Could you please draft something for me?',
        f'Please help me write a short story in response to the prompt "{prompt}."',
        f'Write a {words}-word story in the style of a beginner writer in response to the prompt "{prompt}."',
        f'Write a story with very short sentences in {words} words to the prompt "{prompt}."',
    ]


def get_reuter_prompts(words, headline):
    return [
        f'Write a news article in {words} words based on the headline "{headline}."',
        f'You are a news reporter, who is writing an article with the headline "{headline}." What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word news article based on the following headline: "{headline}." Could you please draft something for me?',
        f'Please help me write a New York Times article for the headline "{headline}."',
        f'Write a {words}-word news article in the style of a New York Times article based on the headline "{headline}."',
        f'Write a news article with very short sentences in {words} words based on the headline "{headline}."',
    ]


def get_essay_prompts(words, prompts):
    return [
        f'Write an essay in {words} words to the prompt "{prompt}."',
        f'You are a student, who is writing an essay in response to the prompt "{prompt}." What would you write in {words} words?',
        f'Hi! I\'m trying to write a {words}-word essay based on the following prompt: "{prompt}." Could you please draft something for me?',
        f'Please help me write an essay in response to the prompt "{prompt}."',
        f"Write a {words}-word essay in the style of a high-school student  in response to the following prompt: {prompt}.",
        f'Write an essay with very short sentences in {words} words to the prompt "{prompt}."',
    ]


def perturb_letter(doc, n=1):
    """
    Randomly swap n pairs of adjacent letters in the document
    """
    if len(doc) < 2:
        return doc

    for _ in range(n):
        idx = np.random.randint(len(doc) - 1)
        doc = doc[:idx] + doc[idx + 1] + doc[idx] + doc[idx + 2 :]
    return doc


def perturb_word(doc, n=1):
    """
    Randomly swap n pairs of adjacent words in the document
    """
    doc = doc.split(" ")
    if len(doc) < 2:
        return " ".join(doc)

    for _ in range(n):
        idx = np.random.randint(len(doc) - 1)
        doc[idx], doc[idx + 1] = doc[idx + 1], doc[idx]
    return " ".join(doc)


def petrub_sent(doc, n=1):
    """
    Randomly swap n pairs of adjacent sentences in the document
    """
    doc = nltk.sent_tokenize(doc)
    if len(doc) < 2:
        return (" ".join(doc)).strip()

    for _ in range(n):
        # Account for the fact that some sentences have new lines in them, so keep them where they were
        idx = np.random.randint(len(doc) - 1)
        doc[idx], doc[idx + 1] = doc[idx + 1], doc[idx]

    return (" ".join(doc)).strip()


def perturb_para(doc, n=1):
    """
    Randomly swap n pairs of adjacent paragraphs in the document
    """
    doc = doc.split("\n")
    if len(doc) < 2:
        return "\n".join(doc)

    for _ in range(n):
        idx = np.random.randint(len(doc) - 1)
        doc[idx], doc[idx + 1] = doc[idx + 1], doc[idx]
    return "\n".join(doc)


def generate_logprobs(generate_dataset_fn):
    files = generate_dataset_fn(lambda f: f)

    for file in tqdm.tqdm(files):
        base_path = os.path.dirname(file) + "/logprobs"
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        with open(file, "r") as f:
            doc = f.read().strip()

        davinci_file = convert_file_to_logprob_file(file, "davinci")
        if not os.path.exists(davinci_file):
            write_logprobs(doc, davinci_file, "davinci")

        ada_file = convert_file_to_logprob_file(file, "ada")
        if not os.path.exists(ada_file):
            write_logprobs(doc, ada_file, "ada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--wp_prompts", action="store_true")
    parser.add_argument("--wp_human", action="store_true")
    parser.add_argument("--wp_gpt", action="store_true")

    parser.add_argument("--reuter_human", action="store_true")
    parser.add_argument("--reuter_gpt", action="store_true")

    parser.add_argument("--essay_prompts", action="store_true")
    parser.add_argument("--essay_human", action="store_true")
    parser.add_argument("--essay_gpt", action="store_true")

    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--logprob_other", action="store_true")

    parser.add_argument("--gen_perturb", action="store_true")
    parser.add_argument("--logprob_perturb", action="store_true")

    args = parser.parse_args()

    if args.wp_prompts:

        def format_prompt(p):
            p = re.sub(r"\[.*\]", "", p)
            p = re.sub(r"\\n", " ", p)
            p = re.sub(r"\\t", " ", p)
            p = re.sub(r"\s+", " ", p)
            return p.strip()

        with open("data/wp/raw/train.wp_source", "r") as f:
            num_lines_read = 0

            print("Generating and writing WP prompts...")

            pbar = tqdm.tqdm(total=1000)
            for prompt in f:
                if num_lines_read >= 1000:
                    break

                input_prompt = format_prompt(prompt)

                response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Remove all the formatting in this prompt:\n\n{input_prompt}",
                        }
                    ],
                )
                reply = response["choices"][0]["message"]["content"].strip()

                with open(f"data/wp/prompts/{num_lines_read + 1}.txt", "w") as f:
                    f.write(reply)

                num_lines_read += 1
                pbar.update(1)

            pbar.close()

    if args.wp_human:
        print("Formatting Human WP documents...")

        with open("data/wp/raw/train.wp_target", "r") as f:
            num_lines_read = 0

            pbar = tqdm.tqdm(total=1000)
            for doc in f:
                if num_lines_read >= 1000:
                    break

                doc = doc.strip()
                tokens = doc.split(" ")

                replace = [
                    ["<newline>", "\n"],
                ]
                for r in replace:
                    tokens = [t.replace(r[0], r[1]) for t in tokens]

                detokenizer = TreebankWordDetokenizer()
                formatted_doc = detokenizer.detokenize(tokens)

                formatted_doc = "\n".join(
                    [i.strip() for i in formatted_doc.split("\n")]
                )
                formatted_doc = formatted_doc.replace("\n\n", "\n")
                formatted_doc = formatted_doc.replace("\n\n", "\n")

                formatted_doc = formatted_doc.replace(" .", ".")
                formatted_doc = formatted_doc.replace(" ’ ", "'")

                formatted_doc = formatted_doc.replace(" ”", '"')
                formatted_doc = formatted_doc.replace("“ ", '"')

                formatted_doc = html_replace(formatted_doc)

                with open(f"data/wp/human/{num_lines_read + 1}.txt", "w") as f:
                    f.write(formatted_doc)

                num_lines_read += 1
                pbar.update(1)

            pbar.close()

    if args.wp_gpt:
        print("Generating GPT WP documents...")

        for idx in tqdm.tqdm(range(1, 1001)):
            with open(f"data/wp/prompts/{idx}.txt", "r") as f:
                prompt = f.read().strip()

            with open(f"data/wp/human/{idx}.txt", "r") as f:
                words = round_to_100(len(f.read().split(" ")))

            prompts = get_wp_prompts(words, prompt)

            for type, prompt in zip(prompt_types, prompts):
                if os.path.exists(f"data/wp/{type}/{idx}.txt"):
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
                reply = reply.replace("\n\n", "\n")

                with open(f"data/wp/{type}/{idx}.txt", "w") as f:
                    f.write(reply)

    if args.reuter_human:
        reuter_replace = ["--", "202-898-8312", "((", "($1=", "(A$", "Reuters Chicago"]

        authors = os.listdir("data/reuter/raw/C50train")
        print("Formatting Human Reuters documents...")

        for author in tqdm.tqdm(authors):
            if not os.path.exists(f"data/reuter/human/{author}"):
                os.makedirs(f"data/reuter/human/{author}")

            files = [
                f"data/reuter/raw/C50train/{author}/{i}"
                for i in os.listdir(f"data/reuter/raw/C50train/{author}")
            ] + [
                f"data/reuter/raw/C50test/{author}/{i}"
                for i in os.listdir(f"data/reuter/raw/C50test/{author}")
            ]

            for n, file in enumerate(files[:20]):
                with open(file, "r") as f:
                    doc = f.read().strip()
                    doc = doc.replace("\n\n", "\n")

                    lines = doc.split("\n")
                    if any([i in lines[-1] for i in reuter_replace]):
                        lines = lines[:-1]
                    doc = "\n".join(lines)
                    doc = html_replace(doc)

                    with open(f"data/reuter/human/{author}/{n+1}.txt", "w") as f:
                        f.write(doc.strip())

    if args.reuter_gpt:
        print("Generating GPT Reuters documents...")

        authors = os.listdir("data/reuter/human")
        for author in tqdm.tqdm(authors):
            for idx in range(1, 21):
                with open(f"data/reuter/human/{author}/{idx}.txt", "r") as f:
                    words = round_to_100(len(f.read().split(" ")))

                with open(f"data/reuter/gpt/{author}/headlines/{idx}.txt", "r") as f:
                    headline = f.read().strip()

                prompts = get_reuter_prompts(words, headline)

                for type, prompt in zip(prompt_types, prompts):
                    if not os.path.exists(f"data/reuter/{type}/{author}"):
                        os.makedirs(f"data/reuter/{type}/{author}")

                    if os.path.exists(f"data/reuter/{type}/{author}/{idx}.txt"):
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
                    reply = reply.replace("\n\n", "\n")

                    lines = reply.split("\n")
                    if any([i in lines[0].lower() for i in ["sure", "certainly"]]):
                        reply = "\n".join(lines[1:])

                    lines = reply.split("\n")
                    if any([i in lines[0].lower() for i in ["title"]]):
                        reply = "\n".join(lines[1:])

                    with open(f"data/reuter/{type}/{author}/{idx}.txt", "w") as f:
                        f.write(reply)

    if args.essay_human or args.essay_gpt:
        essay_dataset = load_dataset("qwedsacf/ivypanda-essays")

    if args.essay_human:
        print("Formatting Human Essay documents...")

        num_documents, idx = 0, 0
        pbar = tqdm.tqdm(total=1000)

        while num_documents < 1000:
            essay = essay_dataset["train"][idx]
            essay = essay["TEXT"].strip()
            essay = essay[essay.index("\n") + 1 :]

            idx += 1

            if "table of contents" in essay.lower() or "[" in essay.lower():
                continue

            essay = essay.replace("\n\n", "\n")
            lines = essay.split("\n")

            doc = []
            for line in lines:
                if any(
                    [
                        i in line.lower()
                        for i in [
                            "references",
                            "reference",
                            "work cited",
                            "works cited",
                            "bibliography",
                        ]
                    ]
                ):
                    break
                doc.append(line)
            doc = "\n".join(doc)

            if len(doc.split(" ")) < 200:
                continue

            with open(f"data/essay/human/{num_documents + 1}.txt", "w") as f:
                f.write(doc.strip())

            num_documents += 1
            pbar.update(1)

    if args.essay_prompts:
        print("Generating Essay prompts...")

        for idx in tqdm.tqdm(range(1, 1001)):
            with open(f"data/essay/human/{idx}.txt", "r") as f:
                doc = f.read().strip()

            response = openai_backoff(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"Given the following essay, write a prompt for it:\n\n{' '.join(doc.split(' ')[:500])}",
                    }
                ],
            )
            reply = response["choices"][0]["message"]["content"].strip()
            reply = reply.replace("Prompt: ", "").strip()

            with open(f"data/essay/prompts/{idx}.txt", "w") as f:
                f.write(reply)

    if args.essay_gpt:
        print("Generating GPT Essay documents...")

        for type in prompt_types:
            if not os.path.exists(f"data/essay/{type}"):
                os.makedirs(f"data/essay/{type}")

        for idx in tqdm.tqdm(range(1, 1001)):
            with open(f"data/essay/prompts/{idx}.txt", "r") as f:
                prompt = f.read().strip()

            with open(f"data/essay/human/{idx}.txt", "r") as f:
                words = round_to_100(len(f.read().split(" ")))

            prompts = get_essay_prompts(words, prompt)

            for type, prompt in zip(prompt_types, prompts):
                if os.path.exists(f"data/essay/{type}/{idx}.txt"):
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
                reply = reply.replace("\n\n", "\n")

                lines = reply.split("\n")
                if any([i in lines[0].lower() for i in ["sure", "certainly"]]):
                    reply = "\n".join(lines[1:])

                lines = reply.split("\n")
                if any([i in lines[0].lower() for i in ["title"]]):
                    reply = "\n".join(lines[1:])

                with open(f"data/essay/{type}/{idx}.txt", "w") as f:
                    f.write(reply)

    if args.logprobs:
        print("Generating WP logprobs...")

        for idx in tqdm.tqdm(range(1, 1001)):
            if not os.path.exists(f"data/wp/human/logprobs"):
                os.makedirs(f"data/wp/human/logprobs")

            with open(f"data/wp/human/{idx}.txt", "r") as f:
                doc = f.read().strip()

            if not os.path.exists(f"data/wp/human/logprobs/{idx}-davinci.txt"):
                write_logprobs(
                    doc, f"data/wp/human/logprobs/{idx}-davinci.txt", "davinci"
                )

            if not os.path.exists(f"data/wp/human/logprobs/{idx}-ada.txt"):
                write_logprobs(doc, f"data/wp/human/logprobs/{idx}-ada.txt", "ada")

            for type in prompt_types + ["claude"]:
                if not os.path.exists(f"data/wp/{type}/logprobs"):
                    os.makedirs(f"data/wp/{type}/logprobs")

                with open(f"data/wp/{type}/{idx}.txt", "r") as f:
                    doc = f.read().strip()

                if not os.path.exists(f"data/wp/{type}/logprobs/{idx}-davinci.txt"):
                    write_logprobs(
                        doc, f"data/wp/{type}/logprobs/{idx}-davinci.txt", "davinci"
                    )

                if not os.path.exists(f"data/wp/{type}/logprobs/{idx}-ada.txt"):
                    write_logprobs(doc, f"data/wp/{type}/logprobs/{idx}-ada.txt", "ada")

        print("Generating Reuters logprobs...")

        authors = os.listdir("data/reuter/human")
        for type in ["human", "claude"] + prompt_types:
            print(f"Generating {type} logprobs...")
            for author in tqdm.tqdm(authors):
                if not os.path.exists(f"data/reuter/{type}/{author}/logprobs"):
                    os.makedirs(f"data/reuter/{type}/{author}/logprobs")

                for i in range(1, 21):
                    with open(f"data/reuter/{type}/{author}/{i}.txt", "r") as f:
                        doc = f.read().strip()

                    if not os.path.exists(
                        f"data/reuter/{type}/{author}/logprobs/{i}-davinci.txt"
                    ):
                        write_logprobs(
                            doc,
                            f"data/reuter/{type}/{author}/logprobs/{i}-davinci.txt",
                            "davinci",
                        )
                    if not os.path.exists(
                        f"data/reuter/{type}/{author}/logprobs/{i}-ada.txt"
                    ):
                        write_logprobs(
                            doc,
                            f"data/reuter/{type}/{author}/logprobs/{i}-ada.txt",
                            "ada",
                        )

        print("Generating Essay logprobs...")

        for idx in tqdm.tqdm(range(1, 1001)):
            if not os.path.exists(f"data/essay/human/logprobs"):
                os.makedirs(f"data/essay/human/logprobs")

            with open(f"data/essay/human/{idx}.txt", "r") as f:
                doc = f.read().strip()

            if not os.path.exists(f"data/essay/human/logprobs/{idx}-davinci.txt"):
                write_logprobs(
                    doc, f"data/essay/human/logprobs/{idx}-davinci.txt", "davinci"
                )

            if not os.path.exists(f"data/essay/human/logprobs/{idx}-ada.txt"):
                write_logprobs(doc, f"data/essay/human/logprobs/{idx}-ada.txt", "ada")

            for type in prompt_types + ["claude"]:
                if not os.path.exists(f"data/essay/{type}/logprobs"):
                    os.makedirs(f"data/essay/{type}/logprobs")

                with open(f"data/essay/{type}/{idx}.txt", "r") as f:
                    doc = f.read().strip()

                if not os.path.exists(f"data/essay/{type}/logprobs/{idx}-davinci.txt"):
                    write_logprobs(
                        doc, f"data/essay/{type}/logprobs/{idx}-davinci.txt", "davinci"
                    )

                if not os.path.exists(f"data/essay/{type}/logprobs/{idx}-ada.txt"):
                    write_logprobs(
                        doc, f"data/essay/{type}/logprobs/{idx}-ada.txt", "ada"
                    )

    if args.logprob_other:
        other_datasets = [
            Dataset("normal", "data/other/ets"),
            Dataset("normal", "data/other/lang8"),
            Dataset("normal", "data/other/pelic"),
            Dataset("normal", "data/other/gptzero/gpt"),
            Dataset("normal", "data/other/gptzero/human"),
        ]

        generate_logprobs(get_generate_dataset(*other_datasets))

    if args.gen_perturb:
        perturb_fns = {
            "letter": perturb_letter,
            "word": perturb_word,
            "sentences": petrub_sent,
            "paragraphs": perturb_para,
        }

        if not os.path.exists("data/perturb"):
            os.makedirs("data/perturb")

        np.random.seed(args.seed)
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
        files = generate_dataset_fn(lambda f: f, verbose=False)

        indices = np.arange(len(test))
        np.random.shuffle(indices)
        indices = indices[:200]

        labels = []
        for file in files[test][indices]:
            if "human" in file and "gpt" not in file:
                labels.append(0)
            elif "gpt" in file and "human" not in file:
                labels.append(1)
            else:
                raise ValueError("Invalid file name")

        with open("data/perturb/labels.txt", "w") as f:
            f.write("\n".join([str(i) for i in labels]))

        # Generate the perturbed documents
        num_perturb = [0, 10, 25, 50, 100, 200]
        for n in tqdm.tqdm(num_perturb):
            for perturb_type, func in perturb_fns.items():
                if not os.path.exists(f"data/perturb/{perturb_type}/{n}"):
                    os.makedirs(f"data/perturb/{perturb_type}/{n}")

                for idx, file in enumerate(files[test][indices]):
                    with open(file, "r") as f:
                        doc = f.read().strip()

                    perturb_doc = func(doc, n=n)
                    with open(f"data/perturb/{perturb_type}/{n}/{idx}.txt", "w") as f:
                        f.write(perturb_doc)

    if args.logprob_perturb:
        perturb_datasets = [
            Dataset("normal", f"data/perturb/{perturb_type}/{n}")
            for perturb_type in ["letter", "word", "sentences", "paragraphs"]
            for n in [0, 10, 25, 50, 100, 200]
        ]

        generate_logprobs(get_generate_dataset(*perturb_datasets))
