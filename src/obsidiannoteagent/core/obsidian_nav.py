from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def ingest_plain_text(
    notes_list: List[Dict[str, Any]],
    persist_dir: str | Path = "./chroma_basic",
    embedding_model: str = "bge-m3",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_text_length: int = 50,
) -> Chroma:
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url="http://localhost:11434",
    )
    vectorstore = Chroma(
        collection_name="obsiddian_plain_text",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    for note in notes_list:
        clean_text = note.get("clean_body", "").strip()
        if len(clean_text) < min_text_length:
            print(f"Skipping short/empty note: {note.get('relative_path')}")
        metadata = {
            "source": note.get("relative_path", "unkown"),
            "note_name": note.get("name", ""),
            "n_backlinks": note.get("n_backlinks", 0),
            "n_tags": note.get("n_tags", 0),
        }
        chunks = text_splitter.create_documents(
            [clean_text],
            metadatas=[metadata] * len(text_splitter.split_text(clean_text)),
        )
        documents.extend(chunks)
    if not documents:
        print("No valid text to index.")
        return vectorstore

    vectorstore.add_documents(documents)
    print(f"Ingestion complete")
    print(f"  - Total notes processed: {len(notes_list)}")
    print(f"  - Total chunks added:    {len(documents)}")
    print(f"  - Collection:             obsidian_plain_text")
    print(f"  - Persisted to:           {persist_dir}")

    return vectorstore
