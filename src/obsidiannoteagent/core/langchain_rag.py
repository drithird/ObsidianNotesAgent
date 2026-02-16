from langchain_core.runnables import RunnablePassthrough
from noteagent.core.chroma_db import ObsidianChromaDB

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def traditional_obsidian_retrieval_chain(db: ObsidianChromaDB, prompt_template: ChatPromptTemplate = None, llm_model: str = "qwen3:14b"):
    default_system = """
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
                    """
    llm = OllamaLLM(
        model=llm_model,
        temperature=0.35,  # low for factual + reasoning
        num_ctx=32000,  # adjust to your model's context limit
    )
    if prompt_template is None:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", default_system + "\n\nQuestion: {question}"),
                ("human", "{question}"),
            ]
        )

    retriever = db._vectorstore.as_retriever(search_kwargs={"k": 7})

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain
