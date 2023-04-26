import os

script_location = os.path.dirname(os.path.realpath(__file__))
project_root_path = os.path.abspath(os.path.join(script_location, os.pardir))

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_PATH = 'backend/index'
INDEX_ABS_PATH = os.path.join(project_root_path, INDEX_PATH)
DATA_PATH = 'data'
DATA_ABS_PATH = os.path.join(project_root_path, DATA_PATH)
QUESTION_PATH = 'data/question.txt'
QUESTION_ABS_PATH = os.path.join(project_root_path, QUESTION_PATH)
TEST_RESULT_PATH = 'backend/test_result/'
TEST_RESULT_ABS_PATH = os.path.join(project_root_path, TEST_RESULT_PATH)
SIMILARITY_RESULTS_COUNT = 10

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

INSTRUCTION = """Imagine that the sequence of events in the format "subject, predicate, object, time" have occured. As an expert at event forecasting, based on the events that have occured predict the missing entity indicated by "???" for the last event template. Simply write the entity name in the response using underscores instead of spaces. If you are unable to predict the missing entity, write "None" in the response."""

"""What would happen next?"""