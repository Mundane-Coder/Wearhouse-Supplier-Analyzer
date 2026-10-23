import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from catalog_processor import process_catalog

load_dotenv()


# --- ChromaDB setup ---

def get_chroma_client(persist_dir: str = "./chroma_db"):
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(client, collection_name: str = "catalog"):
    ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )


# --- Indexing ---

def index_catalog(pdf_path: str, collection_name: str = "catalog", persist_dir: str = "./chroma_db"):
    """
    Parses the PDF, extracts chunks and indexes them in ChromaDB.
    Returns the number of indexed chunks.
    """
    chunks = process_catalog(pdf_path)
    if not chunks:
        raise ValueError("No chunks extracted from the PDF. Check the file and parsing logic.")

    client = get_chroma_client(persist_dir)

    # Delete the old collection if it exists - recreate it for the new file
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = get_or_create_collection(client, collection_name)

    # Add chunks in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            ids=[f"chunk_{i + j}" for j, _ in enumerate(batch)]
        )

    return len(chunks)


def is_catalog_indexed(collection_name: str = "catalog", persist_dir: str = "./chroma_db") -> bool:
    """Checks if the catalog is already indexed."""
    try:
        client = get_chroma_client(persist_dir)
        collection = get_or_create_collection(client, collection_name)
        return collection.count() > 0
    except Exception:
        return False


# --- Retrieval ---

def retrieve_chunks(query: str, collection_name: str = "catalog",
                    persist_dir: str = "./chroma_db", n_results: int = 5) -> list[dict]:
    """
    Searches for relevant chunks based on the query.
    Returns a list of dictionaries with text and metadata.
    """
    client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(client, collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count())
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "metadata": meta})

    return chunks


# --- RAG Chain ---

SYSTEM_PROMPT = """You are a supplier catalog assistant.
Answer only based on the provided catalog context.
If there is not enough information, say so honestly.
Always include the SKU when available in the context.
Be concise and structured in your responses."""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Catalog context:
{context}

Question: {question}""")
])


def build_rag_chain():
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=os.getenv("GROQ_API_KEY")
        )
    else:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1024,
        )
    
    return RAG_PROMPT | llm | StrOutputParser()


def ask_catalog(question: str, collection_name: str = "catalog",
                persist_dir: str = "./chroma_db") -> dict:
    """
    Main function: takes a question, returns the answer and sources.
    """
    # Retrieval
    chunks = retrieve_chunks(question, collection_name, persist_dir)
    if not chunks:
        return {
            "answer": "The catalog is not indexed or is empty.",
            "sources": []
        }

    # Form context
    context = "\n\n".join([c["text"] for c in chunks])

    # Generation
    chain = build_rag_chain()
    answer = chain.invoke({"context": context, "question": question})

    # Sources for display
    sources = []
    for chunk in chunks:
        source = f"Page {chunk['metadata'].get('page', '?')}"
        if chunk['metadata'].get('sku'):
            source += f" | SKU: {chunk['metadata']['sku']}"
        if chunk['metadata'].get('name'):
            source += f" | {chunk['metadata']['name']}"
        sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }


def estimate_tokens(question: str, collection_name: str = "catalog",
                    persist_dir: str = "./chroma_db") -> dict:
    """Estimate token usage without calling the LLM."""
    chunks = retrieve_chunks(question, collection_name, persist_dir)
    context = "\n\n".join([c["text"] for c in chunks])
    
    prompt = f"{SYSTEM_PROMPT}\n\nCatalog context:\n{context}\n\nQuestion: {question}"
    
    # Rough estimate: 1 token ≈ 4 characters
    input_tokens = len(prompt) // 4
    
    return {
        "input_tokens": input_tokens,
        "chunks_used": len(chunks),
        "context_length": len(context),
    }


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_catalog.pdf"
    
    print("Indexing...")
    count = index_catalog(pdf_path)
    print(f"Indexed: {count} chunks\n")

    questions = [
        "What USB-C cables are available?",
        "Show chargers under $5",
        "Are there any wireless chargers?",
    ]

    for q in questions:
        est = estimate_tokens(q)
        print(f"Q: {q}")
        print(f"   tokens: {est['input_tokens']} | chunks: {est['chunks_used']}")
        result = ask_catalog(q)
        print(f"   A: {result['answer'][:150]}")
        print()