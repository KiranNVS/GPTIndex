import openai
import json
from time import sleep
import numpy as np

with open("openai_config.json", encoding="utf-8") as f:
    config = json.load(f)

openai.api_key = config["openai_api_key"]

def request_api(prompt: str, consistency: bool = False, engine: str = "code-davinci-002"):
    got_result = False
    if consistency:
        while not got_result:
            try:
                result = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=64,
                    temperature=0.5,
                    top_p=1,
                    n=30,
                    stop=[']', '.'],
                    logprobs=10,
                )
                got_result = True
            except Exception:
                sleep(3)
    else:
        while not got_result:
            try:
                result = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=64,
                    temperature=0.0,
                    top_p=1,
                    n=1,
                    stop=[']', '.'],
                    logprobs=10,
                )
                got_result = True
            except Exception:
                sleep(3)
    return result


def parse_api_result(result):
    to_return = []
    for idx, g in enumerate(result['choices']):
        text = g['text']
        logprob = sum(g['logprobs']['token_logprobs'])
        to_return.append((text, logprob))
    to_return = sorted(to_return, key=lambda tup: tup[1], reverse=True)
    to_return = [r[0] for r in to_return]
    return to_return

def parse_api_result_rank(result):
    logprob = result['choices'][0]['logprobs']['top_logprobs'][0]
    logprob_list = [(x.strip(), logprob[x]) for x in logprob]
    sorted_logprob_list = sorted(logprob_list, key=lambda tup: tup[1], reverse=True)
    probs = [x[1] for x in sorted_logprob_list]
    softmax_prob = np.exp(probs) / np.sum(np.exp(probs), axis=0)
    to_return = []
    for x, p in zip(sorted_logprob_list, softmax_prob):
        try:
            to_return.append((int(x[0]), p))
        except:
            continue
    return to_return