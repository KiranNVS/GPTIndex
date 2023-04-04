
from fastapi import FastAPI, BackgroundTasks
from fastapi import status

import json_schema as schema
from langchain_sample import *

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/query", status_code=status.HTTP_200_OK)
def get_gptindex_response(PromptPayload: schema.PromptPayload):

    query = PromptPayload.query
    print(query)

    response = qa.query(query)
    print(f'GPT-Index response: {response}')
    return response

if __name__ == "__main__":

    # initialize texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The slow black elephant walks over the lazy dog.",
        "The white cat jumps over the lazy dog.",
    ]

    # initialize class and process documents
    qa = QuestionAnswering(texts)
