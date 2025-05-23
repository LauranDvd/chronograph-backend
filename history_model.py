from copy import deepcopy

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from embedding_model import EmbeddingModel
from project_constants import DATABASE_PATH, LM_STUDIO_URL, CONSTITUTIONAL_MODEL_NAME


class HistoryModel:
    def __init__(self, model_name: str = "gpt2"):

        self.model = ChatOpenAI(
            base_url=LM_STUDIO_URL,
            api_key="lm-studio",
            temperature=0.3,
            model=model_name,
        )

        self.constitutional_model = ChatOpenAI(
            base_url=LM_STUDIO_URL,
            api_key="lm-studio",
            temperature=0,
            model=CONSTITUTIONAL_MODEL_NAME,
        )

        self.embedding = EmbeddingModel()

        # load the existing vector database
        self.vectordb = Chroma(
            embedding_function=self.embedding,
            persist_directory=DATABASE_PATH,
        )
        print(self.vectordb._collection.count())

    def augment_last_user_message(self, chat_history: list) -> (list, list):
        if chat_history[-1]["role"] != "user":
            raise ValueError("Last message is not from the user.")

        # to avoid modifying the original chat history when augmenting the user message
        chat_history = deepcopy(chat_history)

        last_user_message = chat_history[-1]["content"]
        embedded_query = self.embedding.embed_query(last_user_message)
        retrieved_chunks = self.vectordb.similarity_search_by_vector(embedded_query)

        formatted_chunks = ""
        chunk_sources = []
        for i, chunk in enumerate(retrieved_chunks):
            # print(f"Chunk {i}: {chunk}")
            formatted_chunks += f"- {chunk.page_content}\n"
            if "source" in chunk.metadata and "source_title" in chunk.metadata:
                source = chunk.metadata.get("source")
                source_title = chunk.metadata.get("source_title")
                chunk_sources.append(source + ". " + source_title)

        augmented_prompt = f"{last_user_message}\nHere are some chunks of information that could help you, they might be out of order. You should use them only if they are relevant to my question.\nThe chunks are:{formatted_chunks}"
        chat_history[-1]["content"] = augmented_prompt
        return chat_history, chunk_sources

    def generate_response(self, chat_history: list, max_length: int = 128) -> str:
        chat_history, chunk_sources = self.augment_last_user_message(chat_history)
        # print(
        #     f"invoking model with message={chat_history}\n--------------------------\n"
        # )
        response = self.model.invoke(chat_history)

        constitutional_prompt = (
            """"Napoleon introduced slavery" is not harmful content.
"Napoleon introduced slavery and it was a good thing" is harmful content.

"Women did not have many rights during the ages" is not harmful content.
"Women should have less rights" is harmful content.

"The romans killed many people" is not harmful content.
"It's so cool that romans killed many people" is harmful content.

Is the following text harmful content? Respond with ONLY "Yes" or "No".\n\n"""
            + response.content
        )

        constitutional_response = self.constitutional_model.invoke(
            constitutional_prompt
        )

        # Spaghetti code to check if the response is harmful.
        response_is_harmful = True
        accepted_negative_responses = ["no", "No", "NO"]
        for negative_response in accepted_negative_responses:
            if negative_response == constitutional_response.content[:2]:
                response_is_harmful = False
                break

        if response_is_harmful:
            print(
                f"Model response rejected as harmful.\nConstitutional model response: {constitutional_response.content}\nModel response: {response.content}"
            )
            return "The model could not generate a proper response. You may try again or ask something else. We try to keep the model safe and not generate harmful content."

        chunk_sources_nice = "\n".join(chunk_sources)

        response_with_sources = response.content + "\n\nSources:\n" + chunk_sources_nice
        return response_with_sources
