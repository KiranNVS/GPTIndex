import os, sys
import random
import json

lib_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.append(lib_path)

from backend.config import DATA_ABS_PATH, QUESTION_ABS_PATH
from backend.icews_utils import ICEWSDataset

def completion_question_generator():

    TEMPLATES = [
        "Which entity {predicate} to {object} on {timestamp}?\n{subject}",
        "Which entity was {predicate} by {subject} on {timestamp}?\n{object}"
    ]

    icews_dataset = ICEWSDataset(dir_path=DATA_ABS_PATH, dataset_name='ICEWS14', filename='train', idx=[0, 110])
    q_a_pairs = []
    for quadruple in icews_dataset.data:
        for i in range(4):
            if i == 0 or i == 2:# the datum is an entity
                quadruple[i] = quadruple[i].replace('_', ' ').replace("(", "of ").replace(")", "")
            elif i == 1:# the datum is a predicate
                quadruple[i] = quadruple[i].replace('_', ' ')
        q_a_pairs.append(TEMPLATES[0].format(subject=quadruple[0], predicate=quadruple[1], object=quadruple[2], timestamp=quadruple[3]) + '\n')
    with open(QUESTION_ABS_PATH, 'w', encoding='utf-8') as f:
        for q_a_pair in q_a_pairs:
            f.write(q_a_pair)

def y_n_question_generator():

    # to generate training, testing set separately
    def question_generator_helper(set_type):
        if set_type == 'train':
            length = 74844
        elif set_type == 'test':
            length = 7370
        data = ICEWSDataset(dir_path=DATA_ABS_PATH, dataset_name='ICEWS14', filename=set_type, idx=[0, length]).data

        templates = {
            "Make_statement": [
                "Will {subject} make a statement about {object} on {date}?",
                "Is {subject} expected to make a statement about {object} on {date}?",
                "Do you think {subject} will make a statement about {object} on {date}?"
            ],
            "Make_an_appeal_or_request": [
                "Will {subject} make an appeal or request to {object} on {date}?",
                "Is {subject} expected to make an appeal or request to {object} on {date}?",
                "Do you think {subject} will make an appeal or request to {object} on {date}?"
            ],
            "Consult": [
                "Will {subject} consult {object} on {date}?",
                "Is {subject} expected to consult {object} on {date}?",
                "Do you think {subject} will consult {object} on {date}?"
            ],
            "Arrest,_detain,_or_charge_with_legal_action": [
                "Will {subject} arrest, detain, or charge {object} with legal action on {date}?",
                "Is {subject} expected to take any legal action against {object} on {date}?",
                "Do you think {subject} will arrest, detain, or charge {object} with legal action on {date}?"
            ]
        }
            
        entities = []
        with open(os.path.join(DATA_ABS_PATH, 'ICEWS14/entity2id.txt'), 'r', encoding='utf-8') as f:
            for l in f.readlines():
                entities.append(l.split()[0])

        qa_pairs = []

        for r in data:
            if r[1] in templates:
                rand_idx = random.randint(0, len(templates[r[1]]) - 1)
                yes_qa_pair = {
                    'question': templates[r[1]][rand_idx].format(subject=r[0], object=r[2], date=r[3]),
                    'answer': 'Yes'
                }

                rand_entity = random.choice(entities)
                while rand_entity == r[2]:
                    rand_entity = random.choice(entities)
                
                no_qa_pair = {
                    'question': templates[r[1]][rand_idx].format(subject=r[0], object=rand_entity, date=r[3]),
                    'answer': 'No'
                }
                
                qa_pairs.append(yes_qa_pair)
                qa_pairs.append(no_qa_pair)

        json.dump(qa_pairs, open(f'qa_pairs_{set_type}.json', 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        print(len(qa_pairs))
    
    question_generator_helper("train")
    question_generator_helper("test")

if __name__ == "__main__":
    y_n_question_generator()
