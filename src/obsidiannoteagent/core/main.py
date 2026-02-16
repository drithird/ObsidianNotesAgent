from pathlib import Path

from langchain_core.runnables import RunnablePassthrough
from noteagent.core.chroma_db import ObsidianChromaDB
from noteagent.core.folder_navigation import load_vault

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(db: ObsidianChromaDB, llm_model: str = "qwen3:14b"):
    """
    Modern LCEL-style RAG chain (recommended in 2026)
    - Uses retriever from your ChromaDB
    - Stuff context into prompt
    - Runs Ollama for reasoning
    """
    llm = OllamaLLM(
        model=llm_model,
        temperature=0.35,  # low for factual + reasoning
        num_ctx=32000,  # adjust to your model's context limit
    )

    # Prompt tuned for your goals: categorization, connections, gaps
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an intelligent second brain assistant for my Obsidian vault.

                Use ONLY the provided context from my notes.
                Prioritize:
                - Categorizing/tagging information
                - Discovering meaningful connections between notes
                - Identifying knowledge gaps and suggesting what to add
                - Being concise, structured, and actionable

                If something is unclear, contradictory, or missing → say so honestly.
                Suggest new notes, links, or tags when useful.

                Context from notes:
                {context}

                Question: {question}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    retriever = db._vectorstore.as_retriever(
        search_kwargs={"k": 7}  # tune this
    )

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


all_notes = load_vault(Path(Path.cwd() / "data/test_vault/kepano-obsidian-main"))
chroma_db = ObsidianChromaDB(all_notes)
if not chroma_db.check_if_existing_vectorstorage():
    chroma_db.create_new_note_index()

rag = create_rag_chain(chroma_db)

query = "Tell me about Evergreen Notes"
answer = rag.invoke(query)
print(answer)
