import requests
from typing import List
from langchain.embeddings.base import Embeddings
from project_constants import DATABASE_PATH, LM_STUDIO_URL, EMBEDDING_MODEL_NAME


class EmbeddingModel(Embeddings):
    def __init__(
        self,
        base_url: str = LM_STUDIO_URL,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.base_url = base_url
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(f"{self.base_url}/embeddings", json={"input": texts})
        data = response.json()

        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(f"{self.base_url}/embeddings", json={"input": [text]})
        data = response.json()
        return data["data"][0]["embedding"]
