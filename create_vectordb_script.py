from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader, TextLoader
import requests
from typing import List
from langchain.embeddings.base import Embeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from project_constants import DATABASE_PATH, LM_STUDIO_URL, EMBEDDING_MODEL_NAME


# Prepare embedding model
class LocalServerEmbeddings(Embeddings):
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


embedding = LocalServerEmbeddings()


if __name__ == "__main__":

    # Load documents
    loaders = [
        # BSHTMLLoader("docs/Napoleon.html"),
        # BSHTMLLoader("docs/Bistrita.html"),
        TextLoader("docs/napoleon_short.txt"),
        PyPDFLoader("docs/hansel.pdf"),
        PyPDFLoader("docs/red.pdf"),
        PyPDFLoader("docs/robin.pdf"),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # Create semantic chunks
    text_splitter = SemanticChunker(
        embedding
    )  # , sentence_split_regex='(?<=[.?!])\\s+|\\n')

    splits = text_splitter.split_documents(docs)
    print(f"Number of chunks: {len(splits)}")
    splits[:7]

    # Create the vector database
    vectordb = Chroma.from_documents(
        documents=splits, embedding=embedding, persist_directory=DATABASE_PATH
    )

    print(f"Number of chunks in db: {vectordb._collection.count()}")
