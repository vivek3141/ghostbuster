import openai
from .write_logprobs import *

openai_config = json.loads(
    open("/home/nickatomlin/vivek/detect_ai/openai.config").read()
)
openai.api_key = openai_config["api_key"]
openai.organization = openai_config["organization"]
