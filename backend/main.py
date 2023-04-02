
from fastapi import FastAPI, BackgroundTasks
from fastapi import status

import json_schema as schema

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/gptindex", status_code=status.HTTP_200_OK)
def get_gptindex_response(PromptPayload: schema.PromptPayload):
    # call function

    # return response
    pass