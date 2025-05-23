from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    TextLoader,
    JSONLoader,
)
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from project_constants import DATABASE_PATH
from embedding_model import EmbeddingModel


def metadata_func(record: dict, metadata: dict) -> dict:
    # convert lists, dicts, etc. to str because they are not supported by Chroma
    # and handle None values properly
    def to_primitive(value):
        if value is None:
            return ""  # Convert None to empty string
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    meta = record.get("metadata", {})
    metadata["country"] = to_primitive(meta.get("country"))
    metadata["source"] = to_primitive(meta.get("source"))
    metadata["source_title"] = to_primitive(meta.get("source_title"))
    metadata["year"] = to_primitive(meta.get("year"))
    metadata["all_years"] = to_primitive(meta.get("all_years"))
    metadata["time_period"] = to_primitive(meta.get("time_period"))
    metadata["section_title"] = to_primitive(meta.get("section_title"))
    metadata["language"] = to_primitive(meta.get("language"))
    return metadata


if __name__ == "__main__":

    embedding_model = EmbeddingModel()

    # Load documents
    loaders = [
        JSONLoader(
            "docs/history.json",
            jq_schema=".[]",
            content_key="content",
            metadata_func=metadata_func,
            text_content=False,
        ),
        TextLoader("docs/napoleon_short.txt"),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())  # Load only the first 10 documents for testing

    # Create semantic chunks
    text_splitter = SemanticChunker(
        embedding_model
    )  # , sentence_split_regex='(?<=[.?!])\\s+|\\n')

    splits = text_splitter.split_documents(docs)
    print(f"Number of chunks: {len(splits)}")
    # Print the first chunk for debugging
    vector_db=Chroma(
        embedding_function=embedding_model,
        persist_directory=DATABASE_PATH,
    )
    # Create the vector database
    vectordb = Chroma.from_documents(
        documents=splits, embedding=embedding_model, persist_directory=DATABASE_PATH
    )

    print(f"Number of chunks in db: {vectordb._collection.count()}")

# see some of the chunks from the database
# vector_db=Chroma(
#     embedding_function=EmbeddingModel(),
#     persist_directory=DATABASE_PATH,
# )
# all_ids = vector_db._collection.get()["ids"]
# first_ids = all_ids[:5]
#
# # Get the documents for those IDs
# docs = vector_db._collection.get(ids=first_ids)
#
# print(f"Number of chunks in db: {len(all_ids)}")
# print("First 5 chunks:")
# for i in range(5):
#     print(f"\nChunk {i + 1}:")
#     print("ID:", docs["ids"][i])
#     print("Document:", docs["documents"][i])
#     print("Metadata:", docs["metadatas"][i])
# print("Done.")