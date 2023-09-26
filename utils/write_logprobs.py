import openai
import json
import tiktoken

tokenizer = tiktoken.encoding_for_model("davinci")


def write_logprobs(text, file, model):
    """
    Run text under model and write logprobs to file, separated by newline.
    """
    tokens = tokenizer.encode(text)
    doc = tokenizer.decode(tokens[:2047])

    response = openai.Completion.create(
        model=model,
        prompt="<|endoftext|>" + doc,
        max_tokens=0,
        echo=True,
        logprobs=1,
    )

    subwords = response["choices"][0]["logprobs"]["tokens"][1:]
    subprobs = response["choices"][0]["logprobs"]["token_logprobs"][1:]

    gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}

    for i in range(len(subwords)):
        for k, v in gpt2_map.items():
            subwords[i] = subwords[i].replace(k, v)

    to_write = ""
    for _, (w, p) in enumerate(zip(subwords, subprobs)):
        to_write += f"{w} {-p}\n"

    with open(file, "w") as f:
        f.write(to_write)
