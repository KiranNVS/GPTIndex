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

```python
python backend/question_generator.py
```

Then to test the model using generated questions:

```python
python backend/test.py
```

## Extras

If you want to use local alpaca. Follow setup instructions here: https://github.com/antimatter15/alpaca.cpp#get-started-7b

- Put chat executable and the model in `backend/alpaca/`.