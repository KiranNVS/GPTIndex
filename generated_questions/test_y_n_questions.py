import os
import jsonlines
import sys
import datetime
lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)

from frontend.main import parameters
from backend.question_answering import *
from backend.config import QUESTION_ABS_PATH, TEST_RESULT_ABS_PATH
from question_utils import extract_q_a_pairs

# def extract_q_a_pairs(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         i = 0
#         questions = []
#         answers = []
#         for line in f:
#             if not i % 2:
#                 questions.append(line)
#             else:
#                 answers.append(line)
#             i += 1
#     return questions, answers

# TODO: how to calculate accuracy?
# def accuracy_calc(test_result_path):

questions, answers = extract_q_a_pairs(os.path.join(QUESTION_ABS_PATH, 'qa_pairs.json'))
parameters['model'] = 'alpaca'
model = QuestionAnswering(parameters)

now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")  
result_file_path = os.path.join(TEST_RESULT_ABS_PATH, "output_" + timestamp + ".txt")

for i, question in enumerate(questions):
    response, context = model.query(question)
    print("question: " + question)
    print("response: " + response)
    print("answer:" + answers[i])
    result = {
        'question': question,
        'context': context.split('\n'),
        'response': response
    }
    with jsonlines.open(result_file_path, mode="a") as writer:
        writer.write(result)