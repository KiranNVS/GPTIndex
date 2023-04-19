from config import DATA_ABS_PATH, QUESTION_ABS_PATH
from icews_utils import ICEWSDataset

TEMPLATES = [
    "Which entity {predicate} to {object} on {timestamp}?\n{subject}",
    "Which entity was {predicate} by {subject} on {timestamp}?\n{object}"
]

def main():
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

if __name__ == "__main__":
    main()
