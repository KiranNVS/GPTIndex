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

start backend:
```
uvicorn main:app --app-dir=backend/ --reload --port=9000
```

## Extras

If you want to use local alpaca. Follow setup instructions here: https://github.com/antimatter15/alpaca.cpp#get-started-7b

- Put chat executable and the model in `backend/alpaca/`.