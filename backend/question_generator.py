from config import DATA_ABS_PATH, QUESTION_ABS_PATH
from icews_utils import ICEWSDataset

TEMPLATES = [
    "Which entity {predicate} to {object} on {timestamp}?",
    "Which entity was {predicate} by {subject} on {timestamp}?"
]

def main():
    icews_dataset = ICEWSDataset(dir_path=DATA_ABS_PATH, dataset_name='ICEWS14', filename='train', idx=[0, 110])
    with open(QUESTION_ABS_PATH, 'w', encoding='utf-8') as f:
        for quadruple in icews_dataset.data:
            for i in range(4):
                if i == 0 or i == 2:# the datum is an entity
                    quadruple[i] = quadruple[i].replace('_', ' ').replace("(", "of ").replace(")", "")
                elif i == 1:# the datum is a predicate
                    quadruple[i] = quadruple[i].replace('_', ' ')
            f.write(TEMPLATES[0].format(subject=quadruple[0], predicate=quadruple[1], object=quadruple[2], timestamp=quadruple[3]) + '\n')

if __name__ == "__main__":
    main()
