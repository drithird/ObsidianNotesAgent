from os import setegid
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, cast

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class ObsidianChromaDB:
    _embeddings: OllamaEmbeddings
    _vectorstore: Chroma

    def __init__(
        self,
        notes_list: List[Dict[str, Any]],
        collection_name: str = "obsidian_notes",
        db_location_path: str | Path = "./chroma_basic",
        embedding_model: str = "bge-m3",
        ollama_url: str = "http://localhost:11434",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_text_length: int = 50,
    ):
        self.notes_list = notes_list
        self.collection_name = collection_name
        self.db_location_path: Path = Path(db_location_path).expanduser().resolve()
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_text_length = min_text_length

        self._init_embeddings()
        self._get_or_create_vectorstore()

    def _init_embeddings(self):
        """Initialize Ollama embedding model"""
        self._embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_url,
        )

    def _get_or_create_vectorstore(self):
        """Load existing DB or create an empty one"""
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self.db_location_path),
        )

    def generate_documents(self, note, text_splitter):
        clean_text = note.get("clean_body", "").strip()
        if len(clean_text) < self.min_text_length:
            print(f"Skipping short/empty note: {note.get('relative_path')}")
            return []
        metadata = {
            "source": note.get("relative_path", "unknown"),
            "note_name": note.get("name", ""),
            "n_backlinks": note.get("n_backlinks", 0),
            "n_tags": note.get("n_tags", 0),
        }
        chunks = text_splitter.create_documents(
            [clean_text],
            metadatas=[metadata] * len(text_splitter.split_text(clean_text)),
        )
        return chunks

    def _index_notes(self, notes_list):
        documents = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,
        )
        for note in notes_list:
            documents.extend(self.generate_documents(note, text_splitter))
        if not documents:
            print("No valid text to index.")
            return 0
        self._vectorstore.add_documents(documents)
        print(f"Ingestion complete")
        print(f"  - Total notes processed: {len(notes_list)}")
        print(f"  - Total chunks added:    {len(documents)}")
        print(f"  - Collection:             {self.collection_name}")
        print(f"  - Persisted to:           {self.db_location_path}")
        return len(documents)

    def check_if_existing_vectorstorage(self):
        note_count = self.count()
        if note_count > 0:
            return True
        else:
            return False

    def create_new_note_index(self):
        if self._vectorstore is None:
            self._get_or_create_vectorstore()
        else:
            self._vectorstore.delete_collection()
            self._get_or_create_vectorstore()

        self._index_notes(self.notes_list)

    def similarity_search(
        self,
        query: str,
        k: int = 6,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        return self._vectorstore.similarity_search(query, k=k, filter=filter)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 6,
        fethc_k: int = 30,
        lambda_mult: float = 0.45,
    ):
        return self._vectorstore.max_marginal_relevance_search(
            query=query, k=k, fethc_k=fethc_k, lambda_mult=lambda_mult
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        filter: Optional[dict] = None,
    ) -> List[tuple[Document, float]]:
        return self._vectorstore.similarity_search_with_score(query, k=k, filter=filter)

    def count(self) -> int:
        return self._vectorstore._collection.count()

    def add_notes_to_index(self, notes):
        if self._vectorstore is None:
            raise ValueError(
                "You need to create a new note index before you add to one"
            )
        existing_sources = {
            m.get("source", "")
            for m in self._vectorstore.get(include=["metadatas"])["metadatas"]
            if m
        }

        new_notes = [
            note
            for note in notes
            if note.get("relative_path", "unknown") not in existing_sources
        ]
        processed_notes = self._index_notes(new_notes)
        print(f"Succesfully processed {processed_notes} notes!")

    def clear_collection(self) -> None:
        """Delete everything in the collection"""
        self._vectorstore.delete_collection()
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self.db_location_path),
        )

    def delete_by_source(self, source_vaule: str) -> None:
        self._vectorstore.delete(where={"source": source_vaule})

    def get_status(self) -> dict:
        if self._vectorstore is None:
            return {"status": "not initialized", "document_count": 0}

        return {
            "status": "ready",
            "collection": self.collection_name,
            "location": str(self.db_location_path),
            "document_count": self._vectorstore._collection.count(),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
