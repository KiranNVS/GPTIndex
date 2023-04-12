
from fastapi import FastAPI, BackgroundTasks
from fastapi import status
from fastapi.encoders import jsonable_encoder

import json_schema as schema
from question_answering import *

app = FastAPI()

 # initialize texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The slow black elephant walks over the lazy dog.",
    "The white cat jumps over the lazy dog.",
]

# initialize class and process documents
qa = QuestionAnswering(texts)
print('qa initialized')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query", status_code=status.HTTP_200_OK)
def get_gptindex_response(PromptPayload: schema.PromptPayload):

    query = PromptPayload.query
    print(f'query: {query}')

    response = qa.query(query)
    print(f'GPT-Index response: {response}')
    print(f'GPT-Index response type: {type(response)}')
    print(jsonable_encoder(response))
    return jsonable_encoder(response)

if __name__ == "__main__":

    print('inside main.')
    # initialize texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The slow black elephant walks over the lazy dog.",
        "The white cat jumps over the lazy dog.",
    ]

    # initialize class and process documents
    qa = QuestionAnswering(texts)
    print('qa initialized')
