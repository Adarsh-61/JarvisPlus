import streamlit as st
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import concurrent.futures
import hashlib
import time
import logging
import subprocess
import platform
import os
import gc
import mmap
import threading
from multiprocessing import cpu_count
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    PyMuPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
import pdfplumber
import pickle
from joblib import Parallel, delayed
import psutil
import functools

os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning,ignore::DeprecationWarning"

# ======================
# CONSTANTS & CONFIG
# ======================
EMBEDDING_MODEL = "BAAI/bge-m3"
OLLAMA_MODEL = "qwen3:latest"
VISION_MODEL = "qwen3:latest"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_LENGTH = 8192
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
BATCH_SIZE = 32

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

for logger_name in [
    "httpx",
    "sentence_transformers",
    "urllib3",
    "httpcore",
    "httplib2",
    "pdfminer.pdfpage",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# ======================
# CUSTOM COMPONENTS
# ======================
class StreamHandler(BaseCallbackHandler):
    """Real-time streaming callback handler"""

    def __init__(self, container=None):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM produces a new token."""
        self.text += token
        if self.container:
            self.container.markdown(self.text + "▌")

    def on_llm_end(self, *args, **kwargs) -> None:
        """Called when LLM ends generating."""
        if self.container:
            self.container.markdown(self.text)

    def reset(self):
        """Reset the text buffer and container reference."""
        self.text = ""
        self.container = None


class DocumentProcessor:
    """Ultra-optimized document processing pipeline with advanced parallelism, memory, and error handling"""

    def __init__(self):
        self.temp_dir = Path("temp_uploads")
        self.cache_dir = Path("doc_cache")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
            keep_separator=False,
        )
        self.temp_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_workers = min(64, (cpu_count() or 1) * 4 + 8)
        self.cleanup_interval = 900
        self._start_background_cleanup()

    def _start_background_cleanup(self):
        def cleanup():
            while True:
                self._clean_temp_dir()
                self._clean_cache_dir()
                gc.collect()
                time.sleep(self.cleanup_interval)

        t = threading.Thread(target=cleanup, daemon=True)
        t.start()

    def _clean_temp_dir(self):
        for existing_file in self.temp_dir.glob("*"):
            try:
                existing_file.unlink()
            except Exception:
                pass

    def _clean_cache_dir(self):
        now = time.time()
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                if now - cache_file.stat().st_mtime > 2 * 86400:
                    cache_file.unlink()
                else:
                    with open(cache_file, "rb") as f:
                        pickle.load(f)
            except Exception:
                try:
                    cache_file.unlink()
                except Exception:
                    pass

    def get_file_hash(self, file) -> str:
        file_bytes = file.getbuffer()
        return hashlib.md5(file_bytes).hexdigest()

    def process_uploads(self, uploaded_files: List) -> List[Document]:
        """Maximized parallel processing, deduplication, and memory efficiency"""
        start_time = time.time()
        self._clean_temp_dir()
        all_documents = []
        cache_hits = 0
        file_hashes = [self.get_file_hash(file) for file in uploaded_files]
        cache_files = [self.cache_dir / f"{h}.pkl" for h in file_hashes]

        def process_or_load(file, file_hash, cache_file):
            if cache_file.exists():
                nonlocal cache_hits
                try:
                    cache_hits += 1
                    return self._load_from_cache(file_hash)
                except Exception:
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
                    return self._process_file(file, file_hash)
            else:
                return self._process_file(file, file_hash)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="DocProc"
        ) as executor:
            futures = [
                executor.submit(process_or_load, file, file_hash, cache_file)
                for file, file_hash, cache_file in zip(
                    uploaded_files, file_hashes, cache_files
                )
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    docs = future.result()
                    if docs:
                        all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error in file processing: {str(e)}")

        def doc_hash(doc):
            return hashlib.md5(
                doc.page_content.encode("utf-8", errors="ignore")
            ).hexdigest()

        num_hash_jobs = min(self.max_workers, cpu_count() or 1, 32)
        try:
            hashes = Parallel(n_jobs=num_hash_jobs, backend="threading")(
                delayed(doc_hash)(doc) for doc in all_documents
            )
        except Exception as hash_error:
            logger.error(
                f"Parallel hashing failed: {hash_error}. Falling back to sequential.",
                exc_info=True,
            )
            hashes = [doc_hash(doc) for doc in all_documents]

        unique_docs = {}
        for h, doc in zip(hashes, all_documents):
            if h not in unique_docs:
                unique_docs[h] = doc

        processing_time = time.time() - start_time
        logger.info(
            f"Document processing completed in {processing_time:.2f}s. "
            f"{len(unique_docs)} unique docs from {len(uploaded_files)} files ({cache_hits} cache hits). "
            f"RAM: {psutil.virtual_memory().percent}%"
        )
        del all_documents
        gc.collect()
        return list(unique_docs.values())

    def _load_from_cache(self, file_hash: str) -> List[Document]:
        try:
            cache_path = self.cache_dir / f"{file_hash}.pkl"
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Cache loading error: {str(e)}")
            return []

    def _save_to_cache(self, docs: List[Document], file_hash: str) -> None:
        try:
            cache_path = self.cache_dir / f"{file_hash}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(docs, f)
        except Exception as e:
            logger.error(f"Cache saving error: {str(e)}")

    def _process_file(self, file, file_hash: str) -> List[Document]:
        start_time = time.time()
        file_path = self.temp_dir / file.name
        mm = None
        try:
            file_path.write_bytes(file.getbuffer())
            file_size = file_path.stat().st_size
            use_mmap = file_size > 1 * 1024 * 1024

            if use_mmap:
                with open(file_path, "rb") as f:
                    try:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        if file_path.suffix.lower() == ".pdf":
                            docs = self._process_pdf(file_path)
                        elif file_path.suffix.lower() == ".txt":
                            docs = self._process_text(file_path, mmap_obj=mm)
                        else:
                            logger.warning(
                                f"Unsupported file type skipped: {file.name}"
                            )
                            return []
                    finally:
                        if mm:
                            mm.close()
            else:
                if file_path.suffix.lower() == ".pdf":
                    docs = self._process_pdf(file_path)
                elif file_path.suffix.lower() == ".txt":
                    docs = self._process_text(file_path)
                else:
                    logger.warning(f"Unsupported file type skipped: {file.name}")
                    return []

            if not docs:
                logger.warning(f"No content extracted from file: {file.name}")
                return []

            num_split_jobs = min(self.max_workers // 2, cpu_count() or 1, 16)
            chunked_docs = []
            valid_docs = [doc for doc in docs if doc and doc.page_content]
            if not valid_docs:
                logger.warning(
                    f"No valid document content after initial processing for: {file.name}"
                )
                return []

            try:
                chunk_lists = Parallel(n_jobs=num_split_jobs, backend="threading")(
                    delayed(self.splitter.split_documents)([doc]) for doc in valid_docs
                )
                for chunk_list in chunk_lists:
                    if chunk_list:
                        chunked_docs.extend(chunk_list)
            except Exception as split_error:
                logger.error(
                    f"Parallel splitting failed for {file.name}: {split_error}. Falling back to sequential.",
                    exc_info=True,
                )
                try:
                    chunked_docs = self.splitter.split_documents(valid_docs)
                except Exception as seq_split_error:
                    logger.error(
                        f"Sequential splitting also failed for {file.name}: {seq_split_error}",
                        exc_info=True,
                    )
                    return []

            if not chunked_docs:
                logger.warning(f"No chunks generated after splitting file: {file.name}")
                return []

            for doc in chunked_docs:
                doc.metadata["file_hash"] = file_hash
                doc.metadata["file_name"] = file.name
                doc.metadata["chunk_length"] = len(doc.page_content)

            self._save_to_cache(chunked_docs, file_hash)
            processing_time = time.time() - start_time
            logger.info(
                f"Processed and cached {file.name} ({len(chunked_docs)} chunks) in {processing_time:.2f}s"
            )
            del docs
            gc.collect()
            return chunked_docs
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {str(e)}", exc_info=True)
            return []
        finally:
            if mm and not mm.closed:
                mm.close()
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError as unlink_error:
                logger.warning(
                    f"Could not delete temp file {file_path}: {unlink_error}"
                )

    def _process_pdf(self, path: Path, mmap_obj=None) -> List[Document]:
        start_time = time.time()
        try:
            loader = PyMuPDFLoader(str(path))
            docs = loader.load()
            for i, doc in enumerate(docs):
                doc.metadata.update(
                    {
                        "page_number": doc.metadata.get("page", i),
                        "page_context": f"Page {doc.metadata.get('page', i) + 1}",
                        "doc_type": "PDF",
                        "filename": path.name,
                        "source": str(path),
                    }
                )
                doc.metadata.pop("total_pages", None)
                doc.metadata.pop("file_path", None)

            processing_time = time.time() - start_time
            logger.debug(
                f"PyMuPDF processed {path.name} ({len(docs)} pages) in {processing_time:.2f}s"
            )
            return docs
        except ImportError:
            logger.warning(
                f"PyMuPDF package not found, please install it with `pip install pymupdf`. Trying pdfplumber fallback."
            )
            raise
        except Exception as e_pymupdf:
            logger.warning(
                f"PyMuPDF failed for {path}: {str(e_pymupdf)}. Trying pdfplumber fallback."
            )
            try:
                all_texts = []
                with pdfplumber.open(path) as pdf:
                    num_pages = len(pdf.pages)
                    num_extract_jobs = min(8, cpu_count() or 1) if num_pages > 10 else 1

                    page_texts = list(
                        Parallel(n_jobs=num_extract_jobs)(
                            delayed(lambda p: (p.page_number, p.extract_text() or ""))(
                                page
                            )
                            for page in pdf.pages
                        )
                    )

                    page_texts = [pt for pt in page_texts if pt is not None]
                    page_texts.sort(key=lambda x: x[0])

                    docs = []
                    for page_num, text in page_texts:
                        if text.strip():
                            docs.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "page_number": page_num - 1,
                                        "page_context": f"Page {page_num}",
                                        "doc_type": "PDF (Fallback)",
                                        "filename": path.name,
                                        "source": str(path),
                                    },
                                )
                            )

                processing_time = time.time() - start_time
                if docs:
                    logger.info(
                        f"pdfplumber fallback succeeded for {path.name} ({len(docs)} pages) in {processing_time:.2f}s"
                    )
                    return docs
                else:
                    logger.error(
                        f"pdfplumber fallback failed to extract text from {path.name}"
                    )
                    return []
            except Exception as e_fallback:
                logger.error(
                    f"Fallback PDF extraction failed for {path}: {str(e_fallback)}",
                    exc_info=True,
                )
                return []

    def _process_text(self, path: Path, mmap_obj=None) -> List[Document]:
        start_time = time.time()
        try:
            encodings_to_try = ["utf-8", "latin-1", "cp1252"]
            text_content = None
            detected_encoding = None

            if mmap_obj:
                for enc in encodings_to_try:
                    try:
                        text_content = mmap_obj.read().decode(enc)
                        detected_encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
                    finally:
                        mmap_obj.seek(0)
                if text_content is None:
                    mmap_obj.seek(0)
                    text_content = mmap_obj.read().decode("utf-8", errors="replace")
                    detected_encoding = "utf-8 (replaced errors)"

            else:
                for enc in encodings_to_try:
                    try:
                        text_content = path.read_text(encoding=enc)
                        detected_encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as read_err:
                        logger.error(
                            f"Error reading text file {path} directly: {read_err}"
                        )
                        break

                if text_content is None and path.exists():
                    try:
                        text_content = path.read_bytes().decode(
                            "utf-8", errors="replace"
                        )
                        detected_encoding = "utf-8 (replaced errors)"
                    except Exception as fallback_read_err:
                        logger.error(
                            f"Fallback text read failed for {path}: {fallback_read_err}"
                        )
                        return []

            if text_content is None:
                logger.warning(f"Could not read or decode text file: {path}")
                return []

            doc = Document(
                page_content=text_content,
                metadata={
                    "doc_type": "TXT",
                    "filename": path.name,
                    "source": str(path),
                    "encoding": detected_encoding,
                },
            )
            processing_time = time.time() - start_time
            logger.debug(
                f"Processed {path.name} (Encoding: {detected_encoding}) in {processing_time:.2f}s"
            )
            return [doc]

        except Exception as e:
            logger.error(f"Text processing error for {path}: {str(e)}", exc_info=True)
            return []


# ======================
# CORE FUNCTIONALITY
# ======================
def batch_embed(batch, embeddings):
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(batch, embeddings)


class AIAssistant:
    """Enhanced main application controller"""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.initialize_states()
        self.initialize_llm()

    def initialize_states(self):
        """Initialize session state variables"""
        defaults = {
            "message_history": [],
            "vector_store": None,
            "document_chain": None,
            "general_chain": None,
            "conversion_mode": "Text-To-Text",
            "welcome_shown": False,
            "current_docs_hash": None,
            "stream_handler": None,
            "llm": None,
            "embeddings": None,
            "processed_docs": [],
            "general_memory": None,
            "document_memory": None,
            "conversation_context": [],
        }
        for key, val in defaults.items():
            st.session_state.setdefault(key, val)

        st.session_state.stream_handler = StreamHandler()

        if not st.session_state.general_memory:
            st.session_state.general_memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
            )

        if not st.session_state.document_memory:
            st.session_state.document_memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
            )

    def initialize_llm(self):
        """Initialize LLM instance with better error handling"""
        try:
            mode = st.session_state.get("conversion_mode", "Text-To-Text")
            model_name = VISION_MODEL if mode == "Files-To-Text" else OLLAMA_MODEL
            logger.info(f"Initializing LLM with model: {model_name} for mode: {mode}")

            st.session_state.llm = OllamaLLM(
                model=model_name,
                temperature=0.7,
                num_predict=8192,
                callbacks=[st.session_state.stream_handler],
            )
            if mode != "Files-To-Text" and not st.session_state.embeddings:
                logger.info(f"Initializing embeddings model: {EMBEDDING_MODEL}")
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={"device": DEVICE},
                    encode_kwargs={
                        "batch_size": BATCH_SIZE,
                        "normalize_embeddings": True,
                    },
                )
        except Exception as e:
            logger.error(
                f"LLM/Embeddings initialization failed: {str(e)}", exc_info=True
            )
            st.error(
                f"Model initialization failed: {str(e)}. Please ensure Ollama is running and models are available."
            )
            st.session_state.llm = None
            st.session_state.embeddings = None

    def reinitialize_llm_with_handler(self, stream_handler):
        """Reinitialize LLM with current stream handler to enable streaming for each response"""
        try:
            mode = st.session_state.get("conversion_mode", "Text-To-Text")
            model_name = VISION_MODEL if mode == "Files-To-Text" else OLLAMA_MODEL
            logger.debug(f"Reinitializing LLM with model: {model_name}")
            st.session_state.llm = OllamaLLM(
                model=model_name,
                temperature=0.7,
                num_predict=8192,
                callbacks=[stream_handler],
            )
            st.session_state.general_chain = None
            st.session_state.document_chain = None
        except Exception as e:
            logger.error(f"LLM reinitialization failed: {str(e)}", exc_info=True)
            st.error(f"LLM reinitialization failed: {str(e)}")
            st.session_state.llm = None

    def initialize_chains(self) -> None:
        """Initialize conversation chains lazily with proper prompts and error checks"""
        if not st.session_state.llm:
            logger.warning("LLM not initialized. Attempting initialization.")
            self.initialize_llm()
            if not st.session_state.llm:
                st.error("Cannot create chains: LLM initialization failed.")
                return

        try:
            if not st.session_state.general_chain:
                general_prompt = PromptTemplate(
                    input_variables=["chat_history", "question"],
                    template="""You are a helpful AI assistant. Answer questions conversationally based on the context of our conversation.

                    When answering follow-up questions, always refer back to our previous conversation to maintain context.
                    If a question seems vague or references something not explicitly mentioned in the current question,
                    look at our conversation history to understand what the human is referring to.

                    Current conversation:
                    {chat_history}

                    Human: {question}
                    AI:""",
                )

                if not st.session_state.general_memory:
                    st.session_state.general_memory = ConversationBufferMemory(
                        return_messages=True,
                        memory_key="chat_history",
                        input_key="question",
                        output_key="answer",
                    )
                memory = st.session_state.general_memory

                st.session_state.general_chain = (
                    {
                        "question": RunnablePassthrough(),
                        "chat_history": lambda x: self._format_chat_history(memory),
                    }
                    | general_prompt
                    | st.session_state.llm
                    | StrOutputParser()
                )
                logger.info("General conversation chain initialized.")

            if st.session_state.get("conversion_mode") != "Files-To-Text":
                if not st.session_state.embeddings:
                    logger.warning(
                        "Embeddings not initialized. Cannot create document chain yet."
                    )
                    self.initialize_llm()
                    if not st.session_state.embeddings:
                        st.error(
                            "Cannot create document chain: Embeddings initialization failed."
                        )
                        return

                if (
                    st.session_state.vector_store
                    and not st.session_state.document_chain
                ):
                    qa_prompt = PromptTemplate(
                        input_variables=["context", "question", "chat_history"],
                        template="""You are a helpful AI assistant that answers questions based on the provided documents.
                        Use the following context to answer the question. If you don't know the answer from the context, say so politely.

                        IMPORTANT: Pay close attention to the chat history to maintain context. If the question refers to something from previous
                        messages (like "it", "that", "this topic", etc.), use the chat history to understand what the human is asking about.

                        Context:
                        {context}

                        Chat History:
                        {chat_history}

                        Human: {question}

                        Remember: Maintain continuity with previous questions. If the question seems vague, check the chat history
                        to understand what the human is referring to. Answer precisely and stay on topic.

                        Answer:""",
                    )

                    if not st.session_state.document_memory:
                        st.session_state.document_memory = ConversationBufferMemory(
                            return_messages=True,
                            memory_key="chat_history",
                            input_key="question",
                            output_key="answer",
                        )
                    qa_memory = st.session_state.document_memory

                    base_retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity", search_kwargs={"k": 6}
                    )
                    relevance_filter = EmbeddingsFilter(
                        embeddings=st.session_state.embeddings, similarity_threshold=0.6
                    )
                    compressed_retriever = ContextualCompressionRetriever(
                        base_compressor=relevance_filter, base_retriever=base_retriever
                    )
                    st.session_state.retriever = compressed_retriever

                    def format_docs(docs):
                        formatted = []
                        for i, doc in enumerate(docs):
                            source_info = f"Source {i+1} ({doc.metadata.get('filename', 'Unknown')}, {doc.metadata.get('page_context', 'N/A')}):"
                            formatted.append(f"{source_info}\n{doc.page_content}")
                        return "\n\n---\n\n".join(formatted)

                    st.session_state.document_chain = (
                        {
                            "context": st.session_state.retriever | format_docs,
                            "question": RunnablePassthrough(),
                            "chat_history": lambda x: self._format_chat_history(
                                qa_memory
                            ),
                        }
                        | qa_prompt
                        | st.session_state.llm
                        | StrOutputParser()
                    )
                    logger.info("Document RAG chain initialized.")

        except Exception as e:
            logger.error(f"Chain initialization failed: {str(e)}", exc_info=True)
            st.error(f"Chain initialization failed: {str(e)}")
            st.session_state.general_chain = None
            st.session_state.document_chain = None

    def _format_chat_history(self, memory) -> str:
        try:
            history = memory.load_memory_variables({}).get("chat_history", "")

            if isinstance(history, list):
                formatted_history = ""
                for message in history:
                    if hasattr(message, "type") and hasattr(message, "content"):
                        role = "Human" if message.type == "human" else "AI"
                        formatted_history += f"{role}: {message.content}\n"
                return formatted_history
            elif isinstance(history, str) and history:
                return history

            if len(st.session_state.history) > 0:
                formatted_history = ""
                for msg in st.session_state.history:
                    role = "Human" if msg["role"] == "user" else "AI"
                    formatted_history += f"{role}: {msg['content']}\n"
                return formatted_history

            return ""
        except Exception as e:
            logger.error(f"Error formatting chat history: {str(e)}")
            return ""

    def _is_followup_question(self, query: str) -> bool:
        query_lower = query.lower().strip()
        pronouns = [
            " it",
            " this",
            " that",
            " these",
            " those",
            " its",
            " their",
            " them",
        ]
        short_question = len(query_lower.split()) <= 5

        starts_like_followup = query_lower.startswith(
            ("and ", "what about", "how about", "so ", "then ")
        )

        has_pronoun = any(pronoun in query_lower for pronoun in pronouns)

        is_short_after_history = short_question and len(st.session_state.history) > 1

        return starts_like_followup or has_pronoun or is_short_after_history

    def _handle_general_query(self, query: str) -> str:
        if not st.session_state.general_chain:
            self.initialize_chains()
            if not st.session_state.general_chain:
                return "Error: Could not initialize the conversation chain. Please check logs."

        if (
            any(
                term in query.lower()
                for term in ["document", "pdf", "file", "uploaded", "text"]
            )
            and st.session_state.conversion_mode == "Text-To-Text"
        ):
            return "It looks like you're asking about documents, but you're currently in Text-To-Text mode. Switch to Files-To-Text mode in the sidebar and upload documents to ask questions about them."

        try:
            response = st.session_state.general_chain.invoke(query)

            if st.session_state.general_memory:
                st.session_state.general_memory.save_context(
                    {"question": query}, {"answer": response}
                )
            else:
                logger.warning("General memory not available to save context.")

            return response
        except Exception as e:
            logger.error(f"General query failed: {str(e)}", exc_info=True)
            return f"Sorry, I encountered an error processing your request. Please try again. (Error: {str(e)})"

    def _handle_document_query(self, query: str) -> str:
        if not st.session_state.processed_docs:
            return "No documents have been uploaded or processed. Please upload documents first."

        mode = st.session_state.get("conversion_mode")

        if mode == "Files-To-Text":
            if not st.session_state.llm:
                logger.error("Files-To-Text query failed: LLM not initialized.")
                return "Error: The required vision model is not available."
            try:
                context = "\n\n".join(
                    doc.page_content for doc in st.session_state.processed_docs
                )

                MAX_VISION_CONTEXT_CHARS = 100000
                if len(context) > MAX_VISION_CONTEXT_CHARS:
                    logger.warning(
                        f"Files-To-Text context length ({len(context)} chars) exceeds limit ({MAX_VISION_CONTEXT_CHARS}). Truncating."
                    )
                    context = (
                        context[:MAX_VISION_CONTEXT_CHARS]
                        + "\n\n... [Content Truncated] ..."
                    )
                    st.warning(
                        "The combined content of the files is very large and has been truncated to fit the model's context limit."
                    )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "--- Document Content Start ---"},
                            {"type": "text", "text": context},
                            {"type": "text", "text": "--- Document Content End ---"},
                            {
                                "type": "text",
                                "text": f"\nBased on the document content above, answer the following question: {query}",
                            },
                        ],
                    }
                ]
                response = st.session_state.llm.invoke(str(messages))
                return response
            except Exception as e:
                logger.error(
                    f"Vision model document query failed: {str(e)}", exc_info=True
                )
                return f"Sorry, I encountered an error processing your request with the vision model. (Error: {str(e)})"

        else:
            if not st.session_state.document_chain:
                self.initialize_chains()
                if not st.session_state.document_chain:
                    return "Error: Could not initialize the document analysis chain. Please check logs or try reloading documents."

            try:
                start_time = time.time()

                answer = st.session_state.document_chain.invoke(query)

                if st.session_state.document_memory:
                    st.session_state.document_memory.save_context(
                        {"question": query}, {"answer": answer}
                    )
                else:
                    logger.warning("Document memory not available to save context.")

                retrieved_docs = []
                if hasattr(st.session_state, "retriever"):
                    try:
                        retrieved_docs = (
                            st.session_state.retriever.get_relevant_documents(query)
                        )
                    except Exception as retr_err:
                        logger.warning(
                            f"Could not retrieve documents for citation: {retr_err}"
                        )

                query_time = time.time() - start_time
                logger.info(f"Document RAG query processed in {query_time:.2f}s")

                sources_text = ""
                if retrieved_docs:
                    sources = []
                    for i, doc in enumerate(retrieved_docs[:3]):
                        source_desc = doc.metadata.get("filename", "Unknown Source")
                        page_ctx = doc.metadata.get("page_context", None)
                        if page_ctx:
                            source_desc += f" ({page_ctx})"
                        score = doc.metadata.get("relevance_score", None)
                        if score is not None:
                            source_desc += f" (Relevance: {score:.2f})"
                        sources.append(f"• {source_desc}")
                    if sources:
                        sources_text = "\n\n**Sources:**\n" + "\n".join(sources)
                else:
                    logger.warning(
                        f"No documents retrieved for query: '{query}'. Answer might be from history."
                    )

                return f"{answer}{sources_text}"

            except Exception as e:
                logger.error(f"Document RAG query failed: {str(e)}", exc_info=True)
                if "retriever" in str(e).lower():
                    return f"Sorry, I had trouble searching the documents. Please try rephrasing your question or reloading the files. (Error: {str(e)})"
                else:
                    return f"Sorry, I encountered an error generating the answer from the documents. Please try again. (Error: {str(e)})"

    def _generate_simple_doc_summary(self) -> str:
        if not st.session_state.processed_docs:
            return "No documents have been uploaded yet."

        filenames = {}
        for doc in st.session_state.processed_docs:
            filename = doc.metadata.get("filename", "Unknown")
            if filename not in filenames:
                filenames[filename] = 0
            filenames[filename] += 1

        summary = "**Document Summary**\n\n"
        summary += f"I found {len(filenames)} documents with {len(st.session_state.processed_docs)} total chunks.\n\n"
        summary += "**Files:**\n"

        for filename, count in filenames.items():
            summary += f"• {filename} - {count} chunks\n"

        return summary

    def _generate_document_summary(self) -> str:
        if not st.session_state.processed_docs:
            return "No documents have been uploaded yet."

        file_stats = {}
        content_samples = {}

        try:
            for doc in st.session_state.processed_docs:
                filename = doc.metadata.get("filename", "Unnamed Document")
                file_type = doc.metadata.get("doc_type", "Unknown Type")

                if filename not in file_stats:
                    file_stats[filename] = {
                        "count": 0,
                        "type": file_type,
                        "pages": set(),
                        "word_count": 0,
                    }
                    content_samples[filename] = (
                        doc.page_content[:200] + "..."
                        if len(doc.page_content) > 200
                        else doc.page_content
                    )

                file_stats[filename]["count"] += 1
                file_stats[filename]["word_count"] += len(doc.page_content.split())

                page = doc.metadata.get("page_number")
                if page is not None:
                    file_stats[filename]["pages"].add(page)

            summary = "**Document Summary**\n\n"
            summary += f"Total documents loaded: {len(file_stats)} files with {len(st.session_state.processed_docs)} total chunks.\n\n"

            summary += "**Files:**\n"
            for filename, stats in file_stats.items():
                page_info = ""
                if stats["pages"]:
                    page_info = f" ({len(stats['pages'])} pages)"
                summary += f"• {filename} - {stats['type']}{page_info} - {stats['count']} chunks - ~{stats['word_count']} words\n"

            summary += "\n**Content Previews:**\n"
            for filename, preview in content_samples.items():
                preview_clean = preview.replace("\n", " ")
                summary += f'• {filename}:\n  "{preview_clean}"\n\n'

            summary += "**Usage Guide:**\n"
            summary += "You can ask specific questions about these documents such as:\n"
            summary += "• Summarize the main points of [document name]\n"
            summary += "• What does [document name] say about [topic]?\n"
            summary += "• Compare the information about [topic] across documents\n"
            summary += "• Find all mentions of [specific term] in the documents\n"

            return summary
        except Exception as e:
            logger.error(f"Error generating document summary: {str(e)}")
            return self._generate_simple_doc_summary()

    def handle_query(self, query: str) -> str:
        mode = st.session_state.get("conversion_mode", "Text-To-Text")
        logger.info(f"Handling query in mode: {mode}")

        if not st.session_state.llm:
            logger.error(
                f"Query handling aborted: LLM not initialized for mode {mode}."
            )
            self.initialize_llm()
            if not st.session_state.llm:
                st.error(
                    "The AI model is not available. Please check the setup and ensure Ollama is running."
                )
                return "Error: AI model unavailable."

        if mode == "Text-To-Text":
            return self._handle_general_query(query)
        else:
            return self._handle_document_query(query)

    def create_vector_store(self, docs: List[Document]) -> Any:
        start_time = time.time()
        if not docs:
            logger.warning("No documents provided to create vector store.")
            return None
        if not st.session_state.embeddings:
            logger.error("Embeddings not initialized. Cannot create vector store.")
            st.error("Embedding model not available. Cannot process documents for Q&A.")
            self.initialize_llm()
            if not st.session_state.embeddings:
                return None

        try:
            vector_store = None
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            if available_ram_gb > 30 and len(docs) > 5000:
                batch_size = 2048
            elif available_ram_gb > 15 and len(docs) > 1000:
                batch_size = 1024
            elif available_ram_gb > 7:
                batch_size = 512
            else:
                batch_size = 128
            logger.info(
                f"Using batch size: {batch_size} for embedding {len(docs)} documents."
            )

            batches = [
                docs[i : i + batch_size] for i in range(0, len(docs), batch_size)
            ]
            embeddings = st.session_state.embeddings
            batch_embed_partial = functools.partial(batch_embed, embeddings=embeddings)

            num_embed_jobs = min(cpu_count() or 1, 16, len(batches))
            logger.info(
                f"Starting parallel embedding with {num_embed_jobs} jobs using threading backend."
            )

            stores = Parallel(n_jobs=num_embed_jobs, backend="threading")(
                delayed(batch_embed_partial)(batch) for batch in batches
            )

            stores = list(stores)
            logger.info(
                f"Embedding batches completed. Merging {len(stores)} FAISS indices."
            )
            merge_start_time = time.time()
            for i, store in enumerate(stores):
                if store:
                    if vector_store is None:
                        vector_store = store
                    else:
                        try:
                            vector_store.merge_from(store)
                        except Exception as merge_err:
                            logger.error(
                                f"Error merging FAISS index {i+1}/{len(stores)}: {merge_err}",
                                exc_info=True,
                            )
                            st.warning(
                                f"Warning: Could not merge part of the document index. Search results might be incomplete."
                            )
                else:
                    logger.warning(
                        f"Batch {i+1}/{len(stores)} resulted in an empty vector store, skipping merge."
                    )

            merge_time = time.time() - merge_start_time
            total_time = time.time() - start_time
            if vector_store:
                index_size = vector_store.index.ntotal
                logger.info(
                    f"FAISS index merged in {merge_time:.2f}s. Total creation time: {total_time:.2f}s. Index size: {index_size} vectors."
                )
            else:
                logger.error(
                    "Vector store creation resulted in an empty store after merging."
                )
                st.error("Failed to create a searchable index from the documents.")

            gc.collect()
            return vector_store
        except Exception as e:
            logger.error(f"Vector store creation failed: {str(e)}", exc_info=True)
            st.error(f"Error creating document index: {str(e)}")
            return None


def launch_terminal_with_script():
    """Launch a new terminal window and run cli.py inside it"""
    try:
        system = platform.system()
        script_path = "X:/JarvisPlus/cli.py"

        if system == "Windows":
            git_bash_paths = [
                r"C:\Program Files\Git\git-bash.exe",
            ]

            git_bash_path = None
            for path in git_bash_paths:
                if os.path.exists(path):
                    git_bash_path = path
                    break

            if git_bash_path:
                if "git-bash.exe" in git_bash_path.lower():
                    subprocess.Popen(
                        [git_bash_path, "-c", f'python "{script_path}"; exec bash']
                    )
                else:
                    subprocess.Popen(
                        f'start "" "{git_bash_path}" -c "python {script_path}"',
                        shell=True,
                    )
            else:
                try:
                    subprocess.Popen(
                        [
                            "powershell",
                            "-Command",
                            f'Start-Process powershell -ArgumentList "-NoExit", "-Command", "python \'{script_path}\'"',
                        ]
                    )
                except Exception:
                    subprocess.Popen(
                        f'start cmd.exe /k python "{script_path}"', shell=True
                    )
        elif system == "Darwin":
            os.system(
                f"""
                osascript -e 'tell app "Terminal"
                    do script "python {script_path}"
                    activate
                end tell'
            """
            )
        elif system == "Linux":
            for terminal_cmd in [
                ["gnome-terminal", "--", "python", script_path],
                ["xterm", "-e", f'python "{script_path}"'],
                ["konsole", "--new-tab", "-e", f'python "{script_path}"'],
                ["xfce4-terminal", "--command", f'python "{script_path}"'],
            ]:
                try:
                    subprocess.Popen(terminal_cmd)
                    break
                except FileNotFoundError:
                    continue
        else:
            return False
        return True
    except Exception as e:
        print(f"Error launching terminal: {e}")
        return False


def main():
    """Main application with enhanced UI"""

    st.set_page_config(page_title="Jarvis AI", page_icon="🤖", layout="centered")

    if "history" not in st.session_state:
        st.session_state.history = []

    if (
        "message_history" in st.session_state
        and len(st.session_state.message_history) > 0
    ):
        st.session_state.history = st.session_state.message_history
        st.session_state.message_history = []

    assistant = AIAssistant()

    st.markdown(
        """
        <style>
            /* Apply font to all elements */
            * {
                 font-family: 'Comic Sans MS', 'Cursive', sans-serif !important;
            }
            
            /* Main app container */
            .stApp {
                font-family: 'Comic Sans MS', 'Cursive', sans-serif !important;
                min-height: 100vh;
                padding: 20px;
                margin: 0;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] > div:first-child {
                display: flex;
                flex-direction: column;
                height: 100%;
                box-shadow: 2px 0 5px rgba(0,0,0,0.1);
                font-family: 'Comic Sans MS', 'Cursive', sans-serif !important;
            }
            
            /* Headings */
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Comic Sans MS', 'Cursive', sans-serif !important;
            }
            
            h1 {
                font-size: 3em;
                text-align: center;
                margin-bottom: 20px;
                background: linear-gradient(45deg, #3a7bd5, #00d2ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                padding: 10px;
            }
            
            /* Input elements */
            input, textarea, select, button, .stSelectbox, .stTextInput, .stTextArea {
                font-family: 'Comic Sans MS', 'Cursive', sans-serif !important;
            }
            
            /* Chat messages */
            [data-testid="stChatMessageContent"] {
                font-family: 'Comic Sans MS', 'Cursive', sans-serif !important;
            }
            
            /* All paragraphs and text */
            p, span, div, label, a {
                font-family: 'Comic Sans MS', 'Cursive', sans-serif !important;
            }
            
            /* Markdown content */
            .stMarkdown {
                font-family: 'Comic Sans MS', 'Cursive', sans-serif !important;
            }
            
            .stMarkdown p {
                line-height: 1.6;
                font-family: 'Comic Sans MS', 'Cursive', sans-serif !important;
            }
            
            /* Existing styles */
            .footer {
                margin-top: auto;
                padding: 1rem;
                text-align: center;
                width: 100%;
                position: relative;
            }
            
            .mode-indicator {
                padding: 8px 12px;
                border-radius: 20px;
                margin: 10px 0;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }
            
            .document-mode {
                background-color: #e3f2fd;
                color: #1976d2;
                border-left: 4px solid #1976d2;
            }
            .text-mode {
                background-color: #f0f4c3;
                color: #827717;
                border-left: 4px solid #827717;
            }
            .stButton>button {
                border-radius: 20px;
                transition: all 0.3s ease;
                font-weight: 500;
                padding: 0.5rem 2rem;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .center-button {
                display: flex;
                justify-content: center;
                margin-top: 30px;
                margin-bottom: 30px;
            }
            .green-button {
                background-color: #4CAF50 !important;
                color: white !important;
                padding: 10px 24px !important;
                border-radius: 20px !important;
                border: none !important;
                font-size: 16px !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
            }
            .green-button:hover {
                background-color: #45a049 !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
                transform: translateY(-2px) !important;
            }
            .stMarkdown p {
                line-height: 1.6;
            }
            /* Animate transitions */
            .stAnimatedDiv {
                animation: fadeIn 0.5s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            /* Style file uploader */
            [data-testid="stFileUploader"] {
                padding: 15px;
                border-radius: 10px;
                border: 2px dashed #ccc;
                transition: all 0.3s ease;
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #3a7bd5;
            }
            /* Add responsive layout adjustments */
            @media (max-width: 768px) {
                h1 {
                    font-size: 2em;
                }
                .stButton>button {
                    width: 100%;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        if not st.session_state.get("welcome_shown", False):
            st.toast(
                "Hi! I'm Adarsh Pandey, the creator of this AI assistant. I'm glad you're here! I built this AI to help with your questions and ideas. My goal is to use technology to make a positive impact on your life. Let's explore together!"
            )
            st.session_state.welcome_shown = True
    except Exception as e:
        logger.error(f"Error showing welcome message: {e}")

    with st.sidebar:
        st.title("Menu")

        try:
            new_mode = st.selectbox(
                "Select Option:",
                options=["Text-To-Text", "Files-To-Text", "Any-To-Any"],
                index=0
                if st.session_state.conversion_mode == "Text-To-Text"
                else 1
                if st.session_state.conversion_mode == "Files-To-Text"
                else 2,
            )

            if st.session_state.conversion_mode != new_mode:
                st.session_state.conversion_mode = new_mode
                st.session_state.history = []
                st.rerun()
        except Exception as e:
            logger.error(f"Error changing modes: {e}")
            st.error("Failed to change modes. Please refresh the page.")

        if st.session_state.conversion_mode == "Files-To-Text":
            uploaded_files = st.file_uploader(
                "📄 Upload TXT or PDF files:",
                type=["txt", "pdf"],
                accept_multiple_files=True,
            )

            if uploaded_files:
                sorted_files = sorted(uploaded_files, key=lambda f: f.name)
                current_files_hash = hashlib.md5(
                    str([(f.name, f.size) for f in sorted_files]).encode()
                ).hexdigest()

                if current_files_hash != st.session_state.get("current_docs_hash"):
                    with st.spinner(
                        "I'm working on your request, please give me a moment..."
                    ):
                        try:
                            docs = assistant.processor.process_uploads(uploaded_files)
                            if docs:
                                st.session_state.processed_docs = docs
                                st.session_state.vector_store = (
                                    assistant.create_vector_store(docs)
                                )
                                if st.session_state.vector_store:
                                    st.session_state.document_chain = None
                                    assistant.initialize_chains()
                                    st.session_state.current_docs_hash = (
                                        current_files_hash
                                    )
                                    st.success(
                                        f"Successfully processed {len(uploaded_files)} files!"
                                    )
                                else:
                                    st.error(
                                        "Could not create a searchable index from the documents."
                                    )
                                    st.session_state.processed_docs = []
                                    st.session_state.current_docs_hash = None
                            else:
                                st.error(
                                    "Sorry, I am experiencing issues right now. Please try again later."
                                )
                                st.session_state.processed_docs = []
                                st.session_state.current_docs_hash = None
                        except Exception as e:
                            logger.error(f"Document processing error: {e}")
                            st.error(f"Error processing documents: {str(e)}")

    if st.session_state.get("conversion_mode", "Text-To-Text") != "Any-To-Any":
        st.markdown("<h1>Jarvis AI</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style='text-align:center; font-size: 1.4em;'>
                Hi there! I'm an AI assistant created by Adarsh Pandey. I can understand text and files, so feel free to ask anything. I'll give you clear, creative answers to help with whatever you need.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("""---""")

        for i, message in enumerate(st.session_state.history):
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Type a message"):
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.history.append({"role": "user", "content": prompt})

            current_memory = (
                st.session_state.general_memory
                if st.session_state.conversion_mode == "Text-To-Text"
                else st.session_state.document_memory
            )

            if current_memory and len(st.session_state.history) > 0:
                for i in range(0, len(st.session_state.history) - 1, 2):
                    if i + 1 < len(st.session_state.history):
                        user_msg = st.session_state.history[i]
                        if user_msg["role"] == "user":
                            ai_msg = st.session_state.history[i + 1]
                            if ai_msg["role"] == "assistant":
                                current_memory.save_context(
                                    {"question": user_msg["content"]},
                                    {"answer": ai_msg["content"]},
                                )

            with st.chat_message("assistant"):
                response_container = st.empty()
                st.session_state.stream_handler = StreamHandler(
                    container=response_container
                )
                assistant.reinitialize_llm_with_handler(st.session_state.stream_handler)

                response = assistant.handle_query(prompt)

                if not st.session_state.stream_handler.text:
                    response_container.markdown(response)

                st.session_state.history.append(
                    {"role": "assistant", "content": response}
                )

                current_memory = (
                    st.session_state.general_memory
                    if st.session_state.conversion_mode == "Text-To-Text"
                    else st.session_state.document_memory
                )
                current_memory.save_context({"question": prompt}, {"answer": response})

    else:
        st.markdown("<h1>Jarvis+</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <p style='text-align:center; font-size: 1.4em;'>
                Welcome to Jarvis+! My agent-based system is still under development. It might sometimes make mistakes, crash, or be slow. I am continuously working to improve it, so please be patient.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("""---""")
        st.markdown(
            """                    
This AI assistant can:

- 🏠 **Fully Local Execution**: Runs entirely on your local machine without cloud dependency.
- 🗣️ **Voice Interaction**: Supports speech-to-text and text-to-speech.
- 🌐 **Autonomous Web Browsing**: Searches and navigates websites automatically.
- 📂 **File System Management**: Executes bash commands for file navigation and manipulation.
- 💻 **Code Writing & Execution**: Supports Python, C, Java and Golang with debugging capabilities.
- 🛠️ **Self-Correction**: Detects and fixes errors in execution.
- 📝 **Task Planning & Execution**: Uses multiple agents to plan and execute complex tasks.
- 🔄 **Agent Routing**: Automatically selects the best agent for a given task.
- 🧠 **Memory Management**: Maintains session context efficiently.
- ⚙️ **Customizable Configuration**: Modify settings via config.ini file.
- 📡 **Remote Execution**: Can run models on a remote server.

*⚠️ Note: This model is still in development, and I am continuously improving agent routing, multi-language coding support, and system performance.*
"""
        )
        st.markdown("""---""")
        st.markdown(
            "<div style='text-align: center;'>Press the <b>Launch</b> button below to open the Jarvis+ interface.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            try:
                if st.button(
                    "Launch",
                    key="start_multimodal",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Launching Jarvis+..."):
                        success = launch_terminal_with_script()
                        if success:
                            st.balloons()
                        else:
                            st.error("Failed to launch. Please try again later.")
            except Exception as e:
                logger.error(f"Error starting Any-To-Any mode: {e}")
                st.error(
                    "There was a problem launching Jarvis+. Please try again later."
                )
        st.markdown("""---""")


if __name__ == "__main__":
    try:
        import streamlit.watcher.local_sources_watcher

        original_get_module_paths = (
            streamlit.watcher.local_sources_watcher.get_module_paths
        )

        def patched_get_module_paths(module):
            try:
                return original_get_module_paths(module)
            except (RuntimeError, AttributeError):
                return []

        streamlit.watcher.local_sources_watcher.get_module_paths = (
            patched_get_module_paths
        )
    except Exception as e:
        pass

    main()
