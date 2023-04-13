from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from typing import *

import openai
import numpy as np


class QuestionAnswering:
    
    # @staticmethod
    def __new__(self, parameters: dict):
        if parameters['model'] == 'GPT-Neo 125M':
            print(parameters['model'])
            return GPTNeoModel()
        else:
            print(parameters['model'])
            return OpenAIModel(parameters)

class GPTNeoModel:
    def __init__(self):
        self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        self.LLM = "EleutherAI/gpt-neo-125M"
        
        self.embeddings = self.get_embeddings()
        # self.vector_store = self.get_vector_store(query=query)
        self.llm = self.get_llm()

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
        return embeddings

    def get_llm(self):
        model = AutoModelForCausalLM.from_pretrained(self.LLM)
        tokenizer = AutoTokenizer.from_pretrained(self.LLM)
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=64
                        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

    def get_vector_store(self, query) -> FAISS:
        faiss = FAISS.from_texts(query, self.embeddings)
        return faiss

    def get_context(self, query) -> List[str]:
        vector_store = self.get_vector_store(query)
        
        def get_text(x): return x[0].page_content
        context_texts = [get_text(doc)
                         for doc in vector_store.similarity_search_with_score(query, 1)]
        context = '\n'.join(context_texts)
        return context

    def query(self, query) -> str:
        prompt = "{context}\n\n{query}".format(
            context=self.get_context(query),
            query=query
        )
        print(f"Prompt: {prompt}\n{50 * '/'}")
        return self.llm(prompt)

class OpenAIModel:
    def __init__(self, parameters: dict):
        self.params = parameters

    def request_api(self, query: str, parameters: dict):
        openai.api_key = parameters['api_key']

        try:
            response = openai.Completion.create(
                prompt=query,
                engine=parameters['model'],
                temperature=parameters['temperature'],
                max_tokens=parameters['max_length'],
                top_p=parameters['top_p'],
                n=parameters['best_of'],
                logprobs=10,
                stop=[']', '.'],
            )

        except openai.error.Timeout as e:
            #Handle timeout error, e.g. retry or log
            print(f"OpenAI API request timed out: {e}")
            pass
        except openai.error.APIError as e:
            #Handle API error, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            #Handle connection error, e.g. check network or log
            print(f"OpenAI API request failed to connect: {e}")
            pass
        except openai.error.InvalidRequestError as e:
            #Handle invalid request error, e.g. validate parameters or log
            print(f"OpenAI API request was invalid: {e}")
            pass
        except openai.error.AuthenticationError as e:
            #Handle authentication error, e.g. check credentials or log
            print(f"OpenAI API request was not authorized: {e}")
            pass
        except openai.error.PermissionError as e:
            #Handle permission error, e.g. check scope or log
            print(f"OpenAI API request was not permitted: {e}")
            pass
        except openai.error.RateLimitError as e:
            #Handle rate limit error, e.g. wait or log
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass

        return response

    def parse_api_result(self, result, parameters):
        
        text_generated = []
        
        for idx, g in enumerate(result['choices']):
            text = g['text']
            logprob = sum(g['logprobs']['token_logprobs'])
            text_generated.append((text, logprob))
        
        # sort the text generated from model by logprobs
        text_generated = sorted(text_generated, key=lambda tup: tup[1], reverse=True)
        text_generated = [r[0] for r in text_generated]
        
        def print_multi_res(params):
            outtxt = ''
            for i in range(params['best_of']):
                outtxt += text_generated[i].strip()
                outtxt += '\n\n\n'
            return outtxt
        
        return print_multi_res(parameters)

    def query(self, query) -> str:
        response = self.request_api(query, self.params)
        parsed_response = self.parse_api_result(response, self.params)
        return parsed_response
        