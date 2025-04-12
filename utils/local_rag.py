from typing import Union, List, Literal, Tuple, Optional
import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import marimo as mo

from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_model")


def build_rag_chat_model(
    sources: List[Tuple[Union[pd.DataFrame, str, List[str]], Literal['per_row', 'join', 'chunked', 'table'], str]],
    embed_provider: Literal['huggingface', 'ollama', 'openai'] = 'huggingface',
    embed_model: str = "all-MiniLM-L6-v2",
    llm_model: str = "phi4",
    ollama_base_url: str = "http://rtxonrtxoff:11434",
    openai_api_key: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    k_value: Optional[int] = 4,
    force_include_labels: Optional[List[str]] = None
):
    """
    Creates a RAG-powered chat model function from provided context sources.
    """

    docs = []
    for source, strategy, label in sources:
        if isinstance(source, pd.DataFrame):
            if strategy == 'per_row':
                for _, row in source.iterrows():
                    row_text = row.to_frame().T.to_markdown(index=False)
                    docs.append(Document(page_content=row_text, metadata={"source": label}))
            elif strategy == 'join':
                joined = source.to_markdown(index=False)
                docs.append(Document(page_content=joined, metadata={"source": label}))
            elif strategy == 'table':
                text = "### Table Format\n" + source.to_markdown(index=False)
                docs.append(Document(page_content=text, metadata={"source": label}))
            elif strategy == 'chunked':
                text = source.to_markdown(index=False)
                docs.append(Document(page_content=text, metadata={"source": label}))
            else:
                raise ValueError("Unsupported strategy for DataFrame.")
        elif isinstance(source, list):
            if strategy == 'join':
                joined = "\n".join(source)
                docs.append(Document(page_content=joined, metadata={"source": label}))
            elif strategy in ['per_row', 'chunked']:
                for item in source:
                    docs.append(Document(page_content=item, metadata={"source": label}))
            else:
                raise ValueError("Unsupported strategy for list source.")
        elif isinstance(source, str):
            docs.append(Document(page_content=source, metadata={"source": label}))
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

    try:
        tokenizer = AutoTokenizer.from_pretrained(embed_model)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def truncate_docs_to_fit_context(query: str, retrieved_docs: List[Document]) -> str:
        prompt = (system_prompt or "") + "\n\n" + query + "\n"
        prompt_tokens = len(tokenizer.encode(prompt))
        context = ""
        context_tokens = 0
        doc_count = 0

        logger.info("--- Retrieved Chunks ---")
        for i, doc in enumerate(retrieved_docs):
            doc_tokens = len(tokenizer.encode(doc.page_content))
            logger.info(f"Chunk {i+1}: {doc_tokens} tokens | Source: {doc.metadata.get('source', 'unknown')}")
            if prompt_tokens + context_tokens + doc_tokens <= max_tokens:
                context += f"\n{doc.page_content}"
                context_tokens += doc_tokens
                doc_count += 1
            else:
                logger.info(f"Chunk {i+1} skipped due to token limit.")
                break

        logger.info(f"Token budget: {max_tokens}")
        logger.info(f"Prompt tokens: {prompt_tokens}")
        logger.info(f"Context tokens used: {context_tokens} from {doc_count} retrieved chunks")

        return f"{system_prompt or ''}\n\nContext:\n{context}\n\nUser question: {query}"

    def chat_model(messages: List[mo.ai.ChatMessage], config: mo.ai.ChatModelConfig) -> str:
        user_query = messages[-1].content
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

        prompt = truncate_docs_to_fit_context(user_query, retrieved_docs)
        return llm.invoke(prompt)

    return chat_model