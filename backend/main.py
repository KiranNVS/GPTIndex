
from fastapi import FastAPI, BackgroundTasks
from fastapi import status
from fastapi.encoders import jsonable_encoder

import json_schema as schema
from question_answering import *


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query", status_code=status.HTTP_200_OK)
def get_gptindex_response(PromptPayload: schema.PromptPayload):
    parameters = PromptPayload.params
    query = PromptPayload.query
    
    model = QuestionAnswering(parameters)
    print(parameters)
    print(f'query: {query}')

    response, _ = model.query(query)
    print(f'GPT-Index response: {response}')
    print(f'GPT-Index response type: {type(response)}')
    print(jsonable_encoder(response))
    return jsonable_encoder(response)
