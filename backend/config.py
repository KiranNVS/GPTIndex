EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_PATH = 'backend/index'
DATA_PATH = 'data'

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

INSTRUCTION = "The input is a sequence of events in the format 'subject, predicate, object, time'. This is the only information available to you. Answer the question following question as an event in the same format."