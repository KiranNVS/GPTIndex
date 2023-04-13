from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Union, Optional
from typing_extensions import Literal


class PromptPayload(BaseModel):
    params: dict
    query: str

    class Config:
        schema_extra = {
            "example": {
                "query": "Who is the 47th president of United States"
            }
        }
