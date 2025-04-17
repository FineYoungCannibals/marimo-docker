from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings, OpenAIEmbeddings
from langchain_community.llms import Ollama
import logging
import marimo as mo
import os
import pandas as pd
from transformers import AutoTokenizer
from typing import Union, List, Literal, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_model")

TOKENIZER_MAP = {
    "mistral": "mistralai/Mistral-7B-v0.1",
    "phi4": "microsoft/phi-2",
    "gemma3:12b": "google/gemma-1.1-7b-it",
    "llama3": "meta-llama/Meta-Llama-3-8B"
}


def classify_query(query: str) -> Literal['summary', 'row', 'both']:
    lowered = query.lower()
    if any(term in lowered for term in ["most", "overall", "pattern", "trend"]):
        return "summary"
    elif any(term in lowered for term in ["specific", "what about", "tell me about", "show me"]):
        return "row"
    return "both"


def build_rag_chat_model(
    sources: List[Tuple[Union[pd.DataFrame, str, List[str]], Literal['per_row', 'join', 'chunked', 'table'], str]],
    embed_provider: Literal['huggingface', 'ollama', 'openai'] = 'huggingface',
    embed_model: str = "all-MiniLM-L6-v2",
    llm_model: str = "phi4",
    ollama_base_url: str = "http://127.0.0.1:11434",
    openai_api_key: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    k_value: Optional[int] = 4,
    force_include_labels: Optional[List[str]] = None
):
    docs = []
    for source, strategy, label in sources:
        retrieval_type = "summary" if "summary" in label else "row"
        if isinstance(source, pd.DataFrame):
            if strategy == 'per_row':
                for _, row in source.iterrows():
                    row_text = row.to_frame().T.to_markdown(index=False)
                    docs.append(Document(page_content=row_text, metadata={"source": label, "retrieval_type": retrieval_type}))
            elif strategy == 'join':
                joined = source.to_markdown(index=False)
                docs.append(Document(page_content=joined, metadata={"source": label, "retrieval_type": retrieval_type}))
            elif strategy == 'table':
                text = "### Table Format\n" + source.to_markdown(index=False)
                docs.append(Document(page_content=text, metadata={"source": label, "retrieval_type": retrieval_type}))
            elif strategy == 'chunked':
                text = source.to_markdown(index=False)
                docs.append(Document(page_content=text, metadata={"source": label, "retrieval_type": retrieval_type}))
            else:
                raise ValueError("Unsupported strategy for DataFrame.")
        elif isinstance(source, list):
            if strategy == 'join':
                joined = "\n".join(source)
                docs.append(Document(page_content=joined, metadata={"source": label, "retrieval_type": retrieval_type}))
            elif strategy in ['per_row', 'chunked']:
                for item in source:
                    docs.append(Document(page_content=item, metadata={"source": label, "retrieval_type": retrieval_type}))
            else:
                raise ValueError("Unsupported strategy for list source.")
        elif isinstance(source, str):
            docs.append(Document(page_content=source, metadata={"source": label, "retrieval_type": retrieval_type}))
        else:
            raise ValueError("Invalid source or strategy combination.")

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    logger.info(f"Generated {len(chunks)} chunks from source documents.")

    if embed_provider == 'ollama':
        embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_base_url)
    elif embed_provider == 'huggingface':
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    elif embed_provider == 'openai':
        embeddings = OpenAIEmbeddings(model=embed_model, openai_api_key=openai_api_key)
    else:
        raise ValueError("Unsupported embedding provider. Choose from 'ollama', 'huggingface', or 'openai'.")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = Ollama(model=llm_model, base_url=ollama_base_url)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})

    hf_model_name = TOKENIZER_MAP.get(llm_model, llm_model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, token=os.getenv("HUGGINGFACE_TOKEN"), trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Tokenizer for '{llm_model}' not found, falling back to bert-base-uncased: {e}")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def truncate_docs_to_fit_context(query: str, retrieved_docs: List[Document]) -> str:
        prompt = (system_prompt or "") + "\n\n" + query + "\n"
        prompt_tokens = len(tokenizer.encode(prompt))
        context = ""
        context_tokens = 0
        doc_count = 0

        logger.info("--- Retrieved Chunks ---")
        for i, doc in enumerate(retrieved_docs):
            try:
                doc_tokens = len(tokenizer.encode(doc.page_content, truncation=False))
            except Exception as e:
                logger.warning(f"Tokenization failed for chunk {i+1}: {e}")
                doc_tokens = -1

            if doc_tokens > 0:
                logger.info(f"Chunk {i+1}: {doc_tokens} tokens | Source: {doc.metadata.get('source', 'unknown')}")

            if doc_tokens > 0 and (prompt_tokens + context_tokens + doc_tokens <= max_tokens):
                context += f"\n{doc.page_content}"
                context_tokens += doc_tokens
                doc_count += 1
            else:
                logger.info(f"Chunk {i+1} skipped due to token limit.")

        logger.info(f"Token budget: {max_tokens}")
        logger.info(f"Prompt tokens: {prompt_tokens}")
        logger.info(f"Context tokens used: {context_tokens} from {doc_count} retrieved chunks")

        return f"{system_prompt or ''}\n\nContext:\n{context}\n\nUser question: {query}"

    def chat_model(messages: List[mo.ai.ChatMessage], config: mo.ai.ChatModelConfig) -> str:
        user_query = messages[-1].content
        query_type = classify_query(user_query)

        scored_docs = vectorstore.similarity_search_with_score(user_query, k=k_value)

        logger.info("--- Top Retrieved Chunks by Similarity Score ---")
        for i, (doc, score) in enumerate(scored_docs):
            snippet = doc.page_content.strip().replace("\n", " ")[:120]
            logger.info(f"#{i+1:02d} | Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')} | Preview: {snippet}...")

        retrieved_docs = [doc for doc, _ in scored_docs]

        if force_include_labels:
            forced = [doc for doc in chunks if doc.metadata.get("source") in force_include_labels]
            existing_ids = set(id(doc) for doc in retrieved_docs)
            forced_filtered = [doc for doc in forced if id(doc) not in existing_ids]
            if forced_filtered:
                logger.info(f"Injecting {len(forced_filtered)} chunks from force_include_labels: {force_include_labels}")
                retrieved_docs = forced_filtered + retrieved_docs

        if query_type in ["summary", "row"]:
            retrieved_docs = [doc for doc in retrieved_docs if doc.metadata.get("retrieval_type") == query_type]
            logger.info(f"Filtered retrieved_docs to type: {query_type}")

        prompt = truncate_docs_to_fit_context(user_query, retrieved_docs)
        return llm.invoke(prompt)

    return chat_model
