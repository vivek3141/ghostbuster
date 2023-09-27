import argparse
import openai
import re
import tqdm
import os
from datasets import load_dataset

from nltk.tokenize.treebank import TreebankWordDetokenizer
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from utils.generate import generate_documents
from utils.write_logprobs import write_logprobs

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wp_prompts", action="store_true")
    parser.add_argument("--wp_human", action="store_true")
    parser.add_argument("--wp_gpt", action="store_true")

    parser.add_argument("--reuter_human", action="store_true")
    parser.add_argument("--reuter_gpt", action="store_true")

    parser.add_argument("--essay_prompts", action="store_true")
    parser.add_argument("--essay_human", action="store_true")
    parser.add_argument("--essay_gpt", action="store_true")

    parser.add_argument("--logprobs", action="store_true")

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

            if len(doc.split(" ")) < 100:
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

            for type in prompt_types:
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
        for type in ["human", "gpt"]:
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

            for type in prompt_types:
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
