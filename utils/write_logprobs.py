import openai
import json
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


tokenizer = tiktoken.encoding_for_model("davinci")

llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_map = {}
vocab = llama_tokenizer.vocab
for token in vocab:
    idx = vocab[token]
    vocab_map[idx] = token


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


def write_llama_logprobs(text, file, model):
    with torch.no_grad():
        encodings = llama_tokenizer(text, return_tensors="pt").to(device)
        logits = F.softmax(model(encodings["input_ids"]).logits, dim=2)

        tokens = encodings["input_ids"]
        indices = torch.tensor([[[i] for i in tokens[0]]])[:, 1:, :].to(device)

        subwords = [vocab_map[int(idx)] for idx in encodings["input_ids"][0][1:]]
        subprobs = (
            torch.gather(logits[:, :-1, :], dim=2, index=indices)
            .flatten()
            .cpu()
            .detach()
            .numpy()
        )

    to_write = ""
    for _, (w, p) in enumerate(zip(subwords, subprobs)):
        to_write += f"{w} {-np.log(p)}\n"

    with open(file, "w") as f:
        f.write(to_write)
