import os
from typing import *

from config import EMBEDDING_MODEL, INDEX_PATH, PROMPT_TEMPLATE, INSTRUCTION
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from local_llm import LocalLLM


class QuestionAnswering:
    def __init__(self, parameters: dict):
        self.embeddings = self.get_embeddings()
        self.vector_store = self.get_vector_store(INDEX_PATH)

        if parameters['model'] == 'alpaca':
            self.llm = LocalLLM()
        else:
            self.llm = OpenAI(
                api_key=parameters['api_key'],
                temperature=parameters['temperature'],
                max_tokens=parameters['max_length'],
                top_p=parameters['top_p'],
                n=parameters['best_of'],
            )

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return embeddings

    def get_vector_store(self, path) -> FAISS:
        if not os.path.exists(INDEX_PATH):
            raise Exception(f"Index not found at path {path}!")

        faiss = FAISS.load_local(path, self.embeddings)
        return faiss

    def get_context(self, query) -> List[str]:
        def get_text(x): return x[0].page_content
        context_texts = [get_text(doc)
                         for doc in self.vector_store.similarity_search_with_score(query, 10)]
        context = '\n'.join(context_texts)
        return context

    def query(self, query) -> str:
        prompt = PROMPT_TEMPLATE.format(
            instruction=INSTRUCTION,
            input=self.get_context(query) + '\n\n' + query,
        )
        print(f"Prompt: {prompt}\n{50 * '='}")
        return self.llm(prompt)
    

        