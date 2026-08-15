import streamlit as st
import os
import tempfile
import requests
import json
import time
from pathlib import Path
import pandas as pd
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import shutil
from dotenv import load_dotenv

# Configure logging
def setup_logging():
    """Setup comprehensive logging for the application"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger('SarvamQA')
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on rerun
    if logger.handlers:
        return logger

    file_handler = RotatingFileHandler(
        log_dir / f"sarvam_qa_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=10_000_000,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    error_handler = RotatingFileHandler(
        log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=5_000_000,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()

# Load .env file
load_dotenv()

logger.info("=" * 60)
logger.info("Sarvam QA Application Starting")
logger.info(f"Python version: {sys.version}")
logger.info(f"Streamlit version: {st.__version__}")
logger.info("=" * 60)

try:
    logger.info("Attempting to import LlamaIndex components...")
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.readers.file import DocxReader, CSVReader, PptxReader
    from llama_index.core.llms import LLM, CompletionResponse, LLMMetadata
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse
    import llama_index.core
    logger.info("Successfully imported all LlamaIndex components")
except ImportError as e:
    logger.error(f"Import error: {e}", exc_info=True)
    sys.exit(1)

st.set_page_config(page_title="Document Q&A with Sarvam AI", page_icon="📚", layout="wide")
logger.info("Page configuration set")

# ============================================================
# Custom Sarvam LLM for LlamaIndex (no OpenAI dependency)
# ============================================================

SARVAM_CHAT_ENDPOINT = "/v1/chat/completions"
DEFAULT_BASE_URL = "https://api.sarvam.ai"
SARVAM_MODEL = "sarvam-105b"


class SarvamLLM(LLM):
    """LlamaIndex LLM wrapper for Sarvam AI Chat Completions API."""

    def __init__(self, api_key: str = "", base_url: str = DEFAULT_BASE_URL,
                 model: str = SARVAM_MODEL, temperature: float = 0.1,
                 max_tokens: int = 1024, top_p: float = 0.9,
                 reasoning_effort: str = "low", callback_manager=None,
                 system_prompt: str = "", **kwargs):
        super().__init__(callback_manager=callback_manager, **kwargs)
        object.__setattr__(self, '_api_key', api_key)
        object.__setattr__(self, '_base_url', base_url.rstrip("/"))
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_temperature', temperature)
        object.__setattr__(self, '_max_tokens', max_tokens)
        object.__setattr__(self, '_top_p', top_p)
        object.__setattr__(self, '_reasoning_effort', reasoning_effort)
        object.__setattr__(self, '_system_prompt', system_prompt)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128000,
            num_output=4096,
            model_name=object.__getattribute__(self, '_model'),
            is_chat_model=True,
        )

    def _build_messages(self, prompt: str):
        """Build chat messages from a prompt string."""
        messages = []
        system_prompt = object.__getattribute__(self, '_system_prompt')
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        logger.info(f"[{request_id}] SarvamLLM complete called, prompt len={len(prompt)}")

        try:
            api_key = object.__getattribute__(self, '_api_key')
            base_url = object.__getattribute__(self, '_base_url')
            model = object.__getattribute__(self, '_model')
            max_tokens = kwargs.get("max_tokens", object.__getattribute__(self, '_max_tokens'))
            temperature = kwargs.get("temperature", object.__getattribute__(self, '_temperature'))
            top_p = kwargs.get("top_p", object.__getattribute__(self, '_top_p'))
            reasoning_effort = kwargs.get("reasoning_effort", object.__getattribute__(self, '_reasoning_effort'))

            headers = {
                "Content-Type": "application/json",
                "api-subscription-key": api_key,
            }

            payload = {
                "model": model,
                "messages": self._build_messages(prompt),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
                "reasoning_effort": reasoning_effort,
            }

            start_time = time.time()
            response = requests.post(
                f"{base_url}{SARVAM_CHAT_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=120,
            )
            elapsed = time.time() - start_time
            logger.info(f"[{request_id}] API response in {elapsed:.2f}s, status={response.status_code}")

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"] or ""
                logger.info(f"[{request_id}] Got response ({len(content)} chars)")
                return CompletionResponse(text=content)
            else:
                error_text = response.text[:500]
                logger.error(f"[{request_id}] API error {response.status_code}: {error_text}")
                return CompletionResponse(text=f"API Error ({response.status_code}): {error_text}")

        except requests.exceptions.Timeout:
            logger.error(f"[{request_id}] API request timed out", exc_info=True)
            return CompletionResponse(text="Error: Request timed out. Please try again.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{request_id}] Connection error: {e}", exc_info=True)
            return CompletionResponse(text="Error: Cannot connect to Sarvam API.")
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
            return CompletionResponse(text=f"Error: {str(e)}")

    def stream_complete(self, prompt: str, **kwargs):
        request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        logger.info(f"[{request_id}] SarvamLLM stream_complete called")

        try:
            api_key = object.__getattribute__(self, '_api_key')
            base_url = object.__getattribute__(self, '_base_url')
            model = object.__getattribute__(self, '_model')
            max_tokens = kwargs.get("max_tokens", object.__getattribute__(self, '_max_tokens'))
            temperature = kwargs.get("temperature", object.__getattribute__(self, '_temperature'))
            top_p = kwargs.get("top_p", object.__getattribute__(self, '_top_p'))
            reasoning_effort = kwargs.get("reasoning_effort", object.__getattribute__(self, '_reasoning_effort'))

            headers = {
                "Content-Type": "application/json",
                "api-subscription-key": api_key,
            }

            payload = {
                "model": model,
                "messages": self._build_messages(prompt),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
                "reasoning_effort": reasoning_effort,
            }

            response = requests.post(
                f"{base_url}{SARVAM_CHAT_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=120,
                stream=True,
            )

            if response.status_code != 200:
                error_text = response.text[:500]
                logger.error(f"API error {response.status_code}: {error_text}")
                yield CompletionResponse(text=f"API Error ({response.status_code}): {error_text}")
                return

            full_text = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            for choice in choices:
                                delta = choice.get("delta", {})
                                if "content" in delta and delta["content"]:
                                    token = delta["content"]
                                    full_text += token
                                    yield CompletionResponse(text=token, delta=True)
                        except json.JSONDecodeError:
                            pass

            logger.info(f"[{request_id}] Stream complete, total {len(full_text)} chars")

        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
            yield CompletionResponse(text=f"Error: {str(e)}")

    def chat(self, messages):
        """Chat interface — convert messages to prompt and call complete."""
        prompt = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        resp = self.complete(prompt)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=resp.text))

    def stream_chat(self, messages):
        """Stream chat interface."""
        prompt = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        full_text = ""
        for resp in self.stream_complete(prompt):
            full_text += resp.text
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=full_text), delta=resp.delta)

    async def acomplete(self, prompt: str, **kwargs):
        """Async complete — delegates to sync complete."""
        return self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs):
        """Async stream complete — delegates to sync stream_complete."""
        for resp in self.stream_complete(prompt, **kwargs):
            yield resp

    async def achat(self, messages):
        """Async chat — delegates to sync chat."""
        return self.chat(messages)

    async def astream_chat(self, messages):
        """Async stream chat — delegates to sync stream_chat."""
        for resp in self.stream_chat(messages):
            yield resp


def call_sarvam_chat(messages, api_key, base_url, model=SARVAM_MODEL,
                     temperature=0.1, max_tokens=1024, top_p=0.9,
                     stream=False, reasoning_effort="low"):
    """Direct HTTP call to Sarvam API (used for streaming in the chat UI)."""
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.info(f"[{request_id}] Direct Sarvam chat call, model={model}")

    try:
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": api_key,
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        start_time = time.time()
        response = requests.post(
            f"{base_url}{SARVAM_CHAT_ENDPOINT}",
            headers=headers,
            json=payload,
            timeout=120,
            stream=stream,
        )

        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] API response in {elapsed:.2f}s, status={response.status_code}")

        if response.status_code == 200:
            if stream:
                return _parse_streaming_response(response)
            else:
                result = response.json()
                content = result["choices"][0]["message"]["content"] or ""
                logger.info(f"[{request_id}] Got response ({len(content)} chars)")
                return content
        else:
            error_text = response.text[:500]
            logger.error(f"[{request_id}] API error {response.status_code}: {error_text}")
            return f"API Error ({response.status_code}): {error_text}"

    except requests.exceptions.Timeout:
        logger.error(f"[{request_id}] API request timed out", exc_info=True)
        return "Error: Request timed out. Please try again."
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[{request_id}] Connection error: {e}", exc_info=True)
        return "Error: Cannot connect to Sarvam API. Check your internet connection."
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        return f"Error: {str(e)}"


def _parse_streaming_response(response):
    """Parse SSE streaming response from Sarvam API."""
    full_content = ""
    reasoning_content = ""

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta", {})
                        if "content" in delta and delta["content"]:
                            full_content += delta["content"]
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            reasoning_content += delta["reasoning_content"]
                except json.JSONDecodeError:
                    pass

    result = {"content": full_content}
    if reasoning_content:
        result["reasoning"] = reasoning_content
    return result


def test_sarvam_api(api_key: str, base_url: str) -> tuple:
    """Test if Sarvam API is accessible. Returns (success, message)."""
    try:
        headers = {
            "Content-Type": "application/json",
            "api-subscription-key": api_key,
        }
        payload = {
            "model": SARVAM_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        }

        start_time = time.time()
        response = requests.post(
            f"{base_url}{SARVAM_CHAT_ENDPOINT}",
            headers=headers,
            json=payload,
            timeout=15,
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            msg = f"API connection successful! (response time: {elapsed:.2f}s, model: {result.get('model', 'unknown')})"
            logger.info(f"API test successful in {elapsed:.2f}s: {reply}")
            return True, msg
        else:
            error_text = response.text[:300]
            msg = f"API test failed (HTTP {response.status_code}): {error_text}"
            logger.warning(f"API test failed: {msg}")
            return False, msg
    except requests.exceptions.Timeout:
        msg = "API test timeout after 15 seconds"
        logger.error(msg)
        return False, msg
    except Exception as e:
        msg = f"API test failed: {str(e)}"
        logger.error(f"API test exception: {e}", exc_info=True)
        return False, msg


# ============================================================
# Document processing
# ============================================================

def save_uploaded_files(uploaded_files):
    """Save uploaded files to a temporary directory. Returns (paths, temp_dir)."""
    logger.info(f"Saving {len(uploaded_files)} uploaded files")
    saved_paths = []
    temp_dir = tempfile.mkdtemp(prefix="sarvam_qa_")
    logger.debug(f"Created temporary directory: {temp_dir}")

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
        logger.debug(f"Saved: {uploaded_file.name} ({uploaded_file.size} bytes)")

    return saved_paths, temp_dir


def load_documents(file_paths):
    """Load and chunk documents from various formats. Returns list of Documents."""
    all_documents = []
    failed_files = []

    for idx, file_path in enumerate(file_paths, 1):
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name
        file_size = Path(file_path).stat().st_size

        logger.info(f"[{idx}/{len(file_paths)}] Processing: {file_name} ({file_ext}, {file_size} bytes)")

        try:
            if file_ext in (".txt", ".md"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                from llama_index.core.schema import Document
                doc = Document(text=content, metadata={"file_name": file_name, "file_type": file_ext})
                all_documents.append(doc)
                logger.info(f"Processed {file_name} ({len(content)} chars)")

            elif file_ext == ".docx":
                reader = DocxReader()
                docs = reader.load_data(file=file_path)
                for doc in docs:
                    doc.metadata["file_name"] = file_name
                all_documents.extend(docs)
                logger.info(f"Processed {file_name} ({len(docs)} chunks)")

            elif file_ext == ".csv":
                reader = CSVReader()
                docs = reader.load_data(file=file_path)
                for doc in docs:
                    doc.metadata["file_name"] = file_name
                all_documents.extend(docs)
                logger.info(f"Processed {file_name} ({len(docs)} chunks)")

            elif file_ext == ".pptx":
                reader = PptxReader()
                docs = reader.load_data(file=file_path)
                for doc in docs:
                    doc.metadata["file_name"] = file_name
                all_documents.extend(docs)
                logger.info(f"Processed {file_name} ({len(docs)} chunks)")

            elif file_ext == ".xlsx":
                df = pd.read_excel(file_path)
                content = df.to_string()
                from llama_index.core.schema import Document
                doc = Document(text=content, metadata={"file_name": file_name, "file_type": file_ext})
                all_documents.append(doc)
                logger.info(f"Processed {file_name} (Excel: {len(df)} rows)")

            else:  # PDF and others
                docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                for doc in docs:
                    if not hasattr(doc, "metadata"):
                        doc.metadata = {}
                    doc.metadata["file_name"] = file_name
                all_documents.extend(docs)
                logger.info(f"Processed {file_name} ({len(docs)} chunks)")

        except Exception as e:
            error_msg = f"Error processing {file_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            failed_files.append(file_name)

    return all_documents, failed_files


# ============================================================
# Session state initialization
# ============================================================

logger.debug("Initializing session state")
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("SARVAM_API_KEY", "")
if "base_url" not in st.session_state:
    st.session_state.base_url = os.getenv("SARVAM_BASE_URL", DEFAULT_BASE_URL)
if "index" not in st.session_state:
    st.session_state.index = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are a helpful Q&A assistant. Answer questions based only on the provided documents. "
        "If the answer is not in the documents, say \"I cannot find this information in the provided documents.\" "
        "Provide clear, concise answers with relevant details from the documents."
    )
if "model" not in st.session_state:
    st.session_state.model = SARVAM_MODEL
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# Main app
# ============================================================

def main():
    logger.info("Starting main application")

    st.title("Document Q&A with Sarvam AI")
    st.markdown(
        "Upload your documents and ask questions. Powered by **Sarvam AI Sarvam-105B** "
        "with vector search for accurate, source-grounded answers."
    )
    st.markdown(
        'Get your free API key at '
        '[Sarvam AI Dashboard](https://dashboard.sarvam.ai/key-management) — '
        '1000 free credits available.'
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")

        # API Key status from .env
        api_key = st.session_state.api_key
        if api_key:
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "****"
            st.success(f"API Key loaded from .env ({masked_key})")
        else:
            st.error("API Key not found! Set SARVAM_API_KEY in .env file")

        # Base URL status from .env
        base_url = st.session_state.base_url
        st.caption(f"API URL: {base_url}")

        if api_key:
            if st.button("Test API Connection", type="secondary"):
                with st.spinner("Testing connection..."):
                    success, msg = test_sarvam_api(api_key, base_url)
                if success:
                    st.success(f"OK: {msg}")
                else:
                    st.error(f"Failed: {msg}")

        st.divider()

        st.subheader("Model")
        st.markdown("**Sarvam-105B** (Best Quality)")
        st.caption("Flagship 105B parameter model for highest quality answers")

        st.divider()

        st.subheader("Parameters")
        max_tokens = st.slider("Max Response Tokens", 64, 4096, 1024, 64,
                               help="Max tokens in the response")
        chunk_size = st.slider("Chunk Size", 256, 4096, 1024, 256,
                               help="Size of document chunks for indexing")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05,
                                help="0 = deterministic, higher = more creative")
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05,
                          help="Nucleus sampling parameter")

        st.divider()

        st.subheader("Assistant Behavior")
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=150,
        )
        st.session_state.system_prompt = system_prompt

        st.divider()

        if st.button("Clear Session & Files", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            logger.info("Session state cleared")
            st.rerun()

    # --- Main content ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Documents")
        ALLOWED_TYPES = ["pdf", "docx", "txt", "md", "csv", "xlsx", "pptx"]
        uploaded_files = st.file_uploader(
            "Choose files",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            help="Supported: PDF, Word, Text, Markdown, CSV, Excel, PowerPoint",
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.write("**Selected files:**")
            for file in uploaded_files:
                ext = Path(file.name).suffix.lower()
                icon_map = {
                    ".pdf": "📄", ".docx": "📝", ".txt": "📃", ".md": "📋",
                    ".csv": "📊", ".xlsx": "📈", ".pptx": "📽️",
                }
                icon = icon_map.get(ext, "📎")
                st.write(f"{icon} {file.name} ({file.size / 1024:.1f} KB)")

        if st.session_state.uploaded_files and st.session_state.api_key:
            if st.button("Process Documents", type="primary", use_container_width=True):
                logger.info("Process Documents button clicked")
                with st.spinner("Processing documents..."):
                    file_paths, temp_dir = save_uploaded_files(st.session_state.uploaded_files)

                    progress_bar = st.progress(0, text="Loading documents...")
                    all_documents, failed_files = load_documents(file_paths)
                    progress_bar.progress(40, text="Creating embeddings...")

                    # Set up LlamaIndex Settings with our custom SarvamLLM
                    llm = SarvamLLM(
                        api_key=st.session_state.api_key,
                        base_url=st.session_state.base_url,
                        model=SARVAM_MODEL,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        system_prompt=st.session_state.system_prompt,
                    )
                    Settings.llm = llm
                    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
                    Settings.embed_model = embed_model
                    Settings.chunk_size = chunk_size
                    Settings.chunk_overlap = 200

                    progress_bar.progress(70, text="Building vector index...")

                    if not all_documents:
                        st.error("No documents could be processed successfully.")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        progress_bar.empty()
                    else:
                        index = VectorStoreIndex.from_documents(all_documents, show_progress=True)
                        progress_bar.progress(100, text="Index ready!")

                        query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")

                        st.session_state.index = index
                        st.session_state.query_engine = query_engine
                        st.session_state.processing_complete = True
                        st.session_state.temp_dir = temp_dir

                        st.success(
                            f"Loaded {len(all_documents)} chunks from {len(file_paths)} files. "
                            "You can now ask questions!"
                        )

                        if failed_files:
                            st.warning(f"Failed to process: {', '.join(failed_files)}")

                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temp directory: {temp_dir}")

        elif st.session_state.uploaded_files and not st.session_state.api_key:
            st.warning("Please enter your Sarvam API key in the sidebar.")

    with col2:
        st.header("Ask Questions")

        if st.session_state.processing_complete:
            for i, chat in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
                    if chat.get("sources"):
                        with st.expander("View Sources"):
                            for j, src in enumerate(chat["sources"]):
                                st.write(f"**Source {j + 1}:**")
                                st.text(src[:500] + ("..." if len(src) > 500 else ""))
                                st.divider()

            question = st.chat_input("Ask anything about your documents...")

            with st.expander("Advanced Options"):
                retrieval_k = st.slider("Top K Retrieval", 1, 10, 3,
                                        help="Number of document chunks to retrieve")
                reasoning = st.selectbox(
                    "Reasoning Effort",
                    options=["low", "medium", "high", "none"],
                    index=0,
                    help="Higher reasoning = better answers but slower & more credits",
                )

            if question:
                logger.info(f"User asked: {question[:100]}...")

                recent_history = st.session_state.chat_history[-4:]

                # Update query engine with new top_k
                st.session_state.query_engine = (
                    st.session_state.index.as_query_engine(
                        similarity_top_k=retrieval_k, response_mode="compact"
                    )
                )

                with st.spinner("Thinking..."):
                    try:
                        # Get context from vector search
                        search_response = st.session_state.query_engine.query(question)
                        context_text = str(search_response)

                        # Build final messages for Sarvam API
                        final_messages = [
                            {"role": "system", "content": st.session_state.system_prompt},
                            {"role": "user", "content": f"Context from documents:\n{context_text}\n\nQuestion: {question}"}
                        ]

                        for chat in recent_history:
                            final_messages.append({"role": "user", "content": chat["question"]})
                            final_messages.append({"role": "assistant", "content": chat["answer"]})
                        final_messages.append({"role": "user", "content": question})

                        # Stream the response
                        with st.container():
                            response_container = st.empty()
                            streaming_text = ""

                            result = call_sarvam_chat(
                                messages=final_messages,
                                api_key=st.session_state.api_key,
                                base_url=st.session_state.base_url,
                                model=st.session_state.model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                stream=True,
                                reasoning_effort=None if reasoning == "none" else reasoning,
                            )

                            if isinstance(result, dict):
                                streaming_text = result["content"]
                                response_container.markdown(streaming_text)

                                if "reasoning" in result:
                                    with st.expander("Reasoning Steps"):
                                        st.text(result["reasoning"])
                            else:
                                streaming_text = result
                                response_container.markdown(streaming_text)

                        # Show sources
                        sources = []
                        if hasattr(search_response, "source_nodes") and search_response.source_nodes:
                            for node in search_response.source_nodes:
                                sources.append(node.text)

                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": streaming_text,
                            "sources": sources[:3],
                        })

                    except Exception as e:
                        error_msg = f"Error getting answer: {str(e)}"
                        logger.error(error_msg, exc_info=True)
                        st.error(error_msg)
        else:
            st.info(
                "Upload documents and click **Process Documents** to get started. "
                "Then ask questions about your documents!"
            )

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; padding: 15px;'>
            <p style='color: #666;'>Built with Sarvam AI + Streamlit + LlamaIndex</p>
            <p style='color: #888; font-size: 0.85em;'>Upload Documents • Ask Questions • Get Source-Grounded Answers</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Debug logs
    with st.sidebar.expander("Debug Information", expanded=False):
        st.json({
            "model": st.session_state.model,
            "base_url": st.session_state.base_url,
            "processing_complete": st.session_state.processing_complete,
            "chat_history_count": len(st.session_state.chat_history),
            "chunk_size": Settings.chunk_size if hasattr(Settings, 'chunk_size') else "N/A",
        })
        if st.button("Show Recent Logs"):
            log_files = sorted(Path("logs").glob("*.log"))
            if log_files:
                latest_log = log_files[-1]
                with open(latest_log, 'r') as f:
                    lines = f.readlines()[-50:]
                    st.code(''.join(lines))
            else:
                st.info("No log files found")


if __name__ == "__main__":
    logger.info("Application instance starting")
    main()
    logger.info("Application shutdown")
