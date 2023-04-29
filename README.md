# GPT INDEX

## Setup

```
pip install -r requirements.txt
```
## Run

start frontend:
```
streamlit run frontend/main.py
```

create index:
```
python backend/create_index.py
```

start backend:
```
uvicorn main:app --app-dir=backend/ --reload --port=9000
```

## Testing

First to generate the questions, run:

```
python generated_questions/question_generator.py
```

Then to test the model using generated questions:

```
python generated_questions/test_y_n_questions.py
```

## Extras

If you want to use local alpaca. Follow setup instructions here: https://github.com/antimatter15/alpaca.cpp#get-started-7b

- Put chat executable and the model in `backend/alpaca/`.

## BERT/RoBERTa baseline training and testing

- Create directories `bert` and `roberta` under `bert` for storing trained models.
- Run following commands:
```
cd ./bert
python bert.py -m bert #for training BERT
python bert.py -m roberta #for training RoBERTa
```