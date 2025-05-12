from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from embedding_model import EmbeddingModel
from project_constants import DATABASE_PATH, LM_STUDIO_URL


class HistoryModel:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = ChatOpenAI(
            base_url=LM_STUDIO_URL,
            api_key="lm-studio",
            temperature=0.3,
            model=model_name,
        )

        self.embedding = EmbeddingModel()

        # load the existing vector database
        self.vectordb = Chroma(
            embedding_function=self.embedding,
            persist_directory=DATABASE_PATH,
        )
        print(self.vectordb._collection.count())

    def augment_last_user_message(self, chat_history: list) -> list:
        if chat_history[-1]["role"] != "user":
            raise ValueError("Last message is not from the user.")

        last_user_message = chat_history[-1]["content"]
        embedded_query = self.embedding.embed_query(last_user_message)
        retrieved_chunks = self.vectordb.similarity_search_by_vector(embedded_query)

        formatted_chunks = ""
        for i, chunk in enumerate(retrieved_chunks):
            print(f"Chunk {i}: {chunk}")
            formatted_chunks += f"- {chunk.page_content}\n"

        augmented_prompt = f"{last_user_message}\nHere are some chunks of information that could help you, they might be out of order. You should use them only if they are relevant to my question.\nThe chunks are:{formatted_chunks}"
        chat_history[-1]["content"] = augmented_prompt
        return chat_history

    def generate_response(self, chat_history: list, max_length: int = 128) -> str:
        chat_history = self.augment_last_user_message(chat_history)
        response = self.model.invoke(chat_history)
        return response.content
