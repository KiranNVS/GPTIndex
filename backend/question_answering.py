import os
from typing import *

from config import EMBEDDING_MODEL, INDEX_ABS_PATH, PROMPT_TEMPLATE, INSTRUCTION, SIMILARITY_RESULTS_COUNT
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from local_llm import LocalLLM


class QuestionAnswering:
    def __init__(self, parameters: dict):
        self.embeddings = self.get_embeddings()
        self.vector_store = self.get_vector_store(INDEX_ABS_PATH)
        self.is_test_mode = parameters['test_mode']

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
        if not os.path.exists(INDEX_ABS_PATH):
            raise Exception(f"Index not found at path {path}!")

        faiss = FAISS.load_local(path, self.embeddings)
        return faiss

    def get_context(self, query, similarity_results_count) -> List[str]:
        def get_text(x): return x[0].page_content
        context_texts = [get_text(doc)
                         for doc in self.vector_store.similarity_search_with_score(query, similarity_results_count)]
        context = '\n'.join(context_texts)
        return context

    def query(self, query) -> str:
        if self.is_test_mode:
            context = self.get_context(query, 1)
            prompt = context + '\n\n' + query
        else:
            context = self.get_context(query, SIMILARITY_RESULTS_COUNT)
            prompt = PROMPT_TEMPLATE.format(
                instruction=INSTRUCTION,
                input=context + '\n\n' + query,
            )
        print(f"Prompt: {prompt}\n{50 * '='}")
        return self.llm(prompt), context
    

        