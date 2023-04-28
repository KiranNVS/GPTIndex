import json

def extract_q_a_pairs(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        questions = [d['question'] for d in data]
        answers = [d['answer'] for d in data]   

    return questions, answers