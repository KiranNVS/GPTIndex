from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM = "EleutherAI/gpt-neo-125M"


class QuestionAnswering:
    def __init__(self, texts: list[str], prompt: PromptTemplate) -> None:
        self.texts = texts
        self.embeddings = self.get_embeddings()
        self.vector_store = self.get_vector_store()
        self.llm = self.get_llm()

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return embeddings

    def get_vector_store(self) -> FAISS:
        faiss = FAISS.from_texts(self.texts, self.embeddings)
        return faiss

    def get_context(self, entity) -> list[str]:
        def get_text(x): return x[0].page_content
        context_texts = [get_text(
            doc) for doc in self.vector_store.similarity_search_with_score(entity, 1)]
        context = '\n'.join(context_texts)
        return context

    def get_llm(self):
        model = AutoModelForCausalLM.from_pretrained(LLM)
        tokenizer = AutoTokenizer.from_pretrained(LLM)
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=64
                        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

    def query(self, entity: str) -> str:
        final_prompt = prompt.format(
            context=self.get_context(entity), entity=entity)
        print(f"Final prompt: {final_prompt}\n{50 * '='}")
        return self.llm(final_prompt)


if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The slow black elephant walks over the lazy dog.",
        "The white cat jumps over the lazy dog.",
    ]

    PROMPT_TEMPLATE = "{context}\n\nWhat color is the {entity}?"

    prompt = PromptTemplate(
        input_variables=["context", "entity"],
        template=PROMPT_TEMPLATE
    )

    qa = QuestionAnswering(texts, prompt)
    print(qa.query("cat"))
