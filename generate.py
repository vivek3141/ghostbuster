import argparse
import openai
import re
import tqdm
import os
import math
import nltk
import numpy as np
import string
import torch

from nltk.corpus import wordnet
from datasets import load_dataset
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.generate import generate_documents
from utils.write_logprobs import write_logprobs, write_llama_logprobs
from utils.symbolic import convert_file_to_logprob_file
from utils.load import Dataset, get_generate_dataset


nltk.download("wordnet")
nltk.download("omw-1.4")


llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

perturb_char_names = [
    "char_basic",
    "char_space",
    "char_cap",
    "word_adj",
    "word_syn",
]
perturb_char_sizes = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200]

perturb_sent_names = ["sent_adj", "sent_paraph", "para_adj", "para_paraph"]
perturb_sent_sizes = list(range(11))


def closest_synonym(word):
    synonyms = wordnet.synsets(word)
    if not synonyms:
        return None  # Return None if there are no synonyms
    closest_synset = synonyms[0]  # Assume the first synset is the closest
    for synset in synonyms[1:]:
        # Update closest_synset if we find a synset with more lemmas (synonyms)
        if len(synset.lemmas()) > len(closest_synset.lemmas()):
            closest_synset = synset
    # Return the name of the lemma from the closest synset
    # that is not the same as the input word
    for lemma in closest_synset.lemmas():
        if lemma.name() != word:
            return lemma.name()
    return None


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


def generate_logprobs(generate_dataset_fn, llama_7b_model=None, llama_13b_model=None):
    files = generate_dataset_fn(lambda f: f)

    for file in tqdm.tqdm(files):
        if "logprobs" in file:
            continue

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

        llama_7b_file = convert_file_to_logprob_file(file, "llama-7b")
        if llama_7b_model and not os.path.exists(llama_7b_file):
            write_llama_logprobs(doc, llama_7b_file, llama_7b_model)

        llama_13b_file = convert_file_to_logprob_file(file, "llama-13b")
        if llama_13b_model and not os.path.exists(llama_13b_file):
            write_llama_logprobs(doc, llama_13b_file, llama_13b_model)


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
    parser.add_argument("--logprob_llama", action="store_true")

    parser.add_argument("--gen_perturb_char", action="store_true")
    parser.add_argument("--logprob_perturb_char", action="store_true")

    parser.add_argument("--gen_perturb_sent", action="store_true")
    parser.add_argument("--logprob_perturb_sent", action="store_true")

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

            if "table of contents" in essay.lower():
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
        datasets = [
            Dataset("normal", "data/wp/human"),
            Dataset("normal", "data/wp/gpt"),
            Dataset("author", "data/reuter/human"),
            Dataset("author", "data/reuter/gpt"),
            Dataset("normal", "data/essay/human"),
            Dataset("normal", "data/essay/gpt"),
        ]
        generate_logprobs(get_generate_dataset(*datasets))

    if args.logprob_other:
        other_datasets = [
            Dataset("normal", "data/other/ets"),
            Dataset("normal", "data/other/lang8"),
            Dataset("normal", "data/other/pelic"),
            Dataset("normal", "data/other/gptzero/gpt"),
            Dataset("normal", "data/other/gptzero/human"),
            Dataset("normal", "data/other/toefl91"),
            Dataset("normal", "data/other/undetectable"),
        ]

        generate_logprobs(get_generate_dataset(*other_datasets))

    if args.logprob_llama:
        print("Loading LLAMA...")
        # llama_7b = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(
        #     device
        # )
        llama_13b = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-AWQ").to(
            device
        )
        print("LLAMA Loaded")

        datasets = [
            Dataset("normal", "data/wp/human"),
            Dataset("normal", "data/wp/gpt"),
            Dataset("author", "data/reuter/human"),
            Dataset("author", "data/reuter/gpt"),
            Dataset("normal", "data/essay/human"),
            Dataset("normal", "data/essay/gpt"),
        ]
        generate_logprobs(
            get_generate_dataset(*datasets),
            # llama_7b_model=llama_7b,
            llama_13b_model=llama_13b,
        )

    if args.gen_perturb_char:

        def perturb_char_basic(doc, n=1):
            if len(doc) < 2:
                return doc

            for _ in range(n):
                peturb_type = np.random.choice(["swap", "delete", "insert"])
                if peturb_type == "swap":
                    idx = np.random.randint(len(doc) - 1)
                    doc = doc[:idx] + doc[idx + 1] + doc[idx] + doc[idx + 2 :]
                elif peturb_type == "delete" and len(doc) > 1:
                    idx = np.random.randint(len(doc))
                    doc = doc[:idx] + doc[idx + 1 :]
                elif peturb_type == "insert":
                    idx = np.random.randint(len(doc))
                    doc = (
                        doc[:idx]
                        + np.random.choice(list(string.ascii_letters))
                        + doc[idx:]
                    )
            return doc

        def perturb_char_space(doc, n=1):
            if len(doc) < 2:
                return doc

            for _ in range(n):
                perturb_type = np.random.choice(["insert", "delete"])
                if perturb_type == "insert":
                    idx = np.random.randint(len(doc))
                    doc = doc[:idx] + " " + doc[idx:]
                elif perturb_type == "delete":
                    space_indices = [
                        idx for idx, c in enumerate(doc) if c == " " or c == "\n"
                    ]
                    if len(space_indices) > 0:
                        idx = np.random.choice(space_indices)
                        doc = doc[:idx] + doc[idx + 1 :]
            return doc

        def perturb_char_cap(doc, n=1):
            if len(doc) < 2:
                return doc

            for _ in range(n):
                idx = np.random.randint(len(doc))
                if doc[idx].isalpha():
                    if doc[idx].isupper():
                        doc = doc[:idx] + doc[idx].lower() + doc[idx + 1 :]
                    else:
                        doc = doc[:idx] + doc[idx].upper() + doc[idx + 1 :]
            return doc

        def perturb_word_adj(doc, n=1):
            words = doc.split(" ")
            if len(words) < 2:
                return doc

            for _ in range(n):
                idx = np.random.randint(len(words) - 1)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
            doc = " ".join(words)

            return doc

        def perturb_word_syn(doc, n=1):
            words = doc.split(" ")
            if len(words) < 2:
                return doc

            for _ in range(n):
                idx = np.random.randint(len(words))
                word = words[idx]
                synonym = closest_synonym(word)
                if synonym:
                    words[idx] = synonym
            doc = " ".join(words)

            return doc

        perturb_char_word_fns = {
            "char_basic": perturb_char_basic,
            "char_space": perturb_char_space,
            "char_cap": perturb_char_cap,
            "word_adj": perturb_word_adj,
            "word_syn": perturb_word_syn,
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
        num_perturb = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200]
        for n in tqdm.tqdm(num_perturb):
            for perturb_type, func in perturb_char_word_fns.items():
                if not os.path.exists(f"data/perturb/{perturb_type}/{n}"):
                    os.makedirs(f"data/perturb/{perturb_type}/{n}")

                for idx, file in enumerate(files[test][indices]):
                    with open(file, "r") as f:
                        doc = f.read().strip()

                    perturb_doc = func(doc, n=n)
                    with open(f"data/perturb/{perturb_type}/{n}/{idx}.txt", "w") as f:
                        f.write(perturb_doc)

    if args.logprob_perturb_char:
        perturb_datasets = [
            Dataset("normal", f"data/perturb/{perturb_type}/{n}")
            for perturb_type in perturb_char_names
            for n in perturb_char_sizes
        ]

        generate_logprobs(get_generate_dataset(*perturb_datasets))

    if args.gen_perturb_sent:
        if torch.cuda.is_available():
            device = "cuda"
            print("Using GPU")
        else:
            device = "cpu"
            print("Using CPU")

        tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
        model = PegasusForConditionalGeneration.from_pretrained(
            "tuner007/pegasus_paraphrase"
        ).to(device)

        def paraphrase(text):
            batch = tokenizer(
                [text], truncation=True, padding="longest", return_tensors="pt"
            ).to(device)
            translated = model.generate(**batch)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
            return tgt_text[0]

        def perturb_sent_adj(doc, n=1):
            """
            Randomly swap n pairs of adjacent sentences in the document
            """
            doc = nltk.sent_tokenize(doc)
            if len(doc) < 2:
                return (" ".join(doc)).strip()

            for _ in range(n):
                idx = np.random.randint(len(doc) - 1)
                doc[idx], doc[idx + 1] = doc[idx + 1], doc[idx]

            return (" ".join(doc)).strip()

        def perturb_sent_paraph(doc, n=1):
            """
            Randomly paraphrase n sentences in the document
            """
            doc = nltk.sent_tokenize(doc)
            if len(doc) < 1:
                return (" ".join(doc)).strip()

            for _ in range(n):
                idx = np.random.randint(len(doc))
                doc[idx] = paraphrase(doc[idx])

            return (" ".join(doc)).strip()

        def perturb_para_adj(doc, n=1):
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

        def perturb_para_paraph(doc, n=1):
            """
            Randomly paraphrase n paragraphs in the document
            """
            doc = doc.split("\n")
            if len(doc) < 1:
                return "\n".join(doc)

            for _ in range(n):
                idx = np.random.randint(len(doc))
                doc[idx] = paraphrase(doc[idx])

            return "\n".join(doc)

        perturb_sent_fns = {
            "sent_adj": perturb_sent_adj,
            "sent_paraph": perturb_sent_paraph,
            "para_adj": perturb_para_adj,
            "para_paraph": perturb_para_paraph,
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
        num_perturb = list(range(11))
        for n in tqdm.tqdm(num_perturb):
            for perturb_type, func in perturb_sent_fns.items():
                if not os.path.exists(f"data/perturb/{perturb_type}/{n}"):
                    os.makedirs(f"data/perturb/{perturb_type}/{n}")

                for idx, file in enumerate(files[test][indices]):
                    with open(file, "r") as f:
                        doc = f.read().strip()

                    perturb_doc = func(doc, n=n)
                    with open(f"data/perturb/{perturb_type}/{n}/{idx}.txt", "w") as f:
                        f.write(perturb_doc)

    if args.logprob_perturb_sent:
        perturb_datasets = [
            Dataset("normal", f"data/perturb/{perturb_type}/{n}")
            for perturb_type in perturb_sent_names
            for n in perturb_sent_sizes
        ]

        generate_logprobs(get_generate_dataset(*perturb_datasets))
