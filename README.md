# Document Q&A with Sarvam AI

An interactive Streamlit application for intelligent question-answering over documents using **Sarvam AI's Sarvam-105B** flagship model. Upload your documents, ask questions, and get accurate, source-grounded answers with streaming responses.

## Features

- **Multi-Format Support**: PDF, DOCX, TXT, MD, CSV, XLSX, PPTX
- **Sarvam-105B Model**: Flagship 105B parameter model for highest quality answers
- **Streaming Responses**: See answers generated in real-time
- **Vector Search**: Semantic retrieval from your documents using BAAI/bge-small-en-v1.5 embeddings
- **Chat History**: Conversational context over your documents
- **Source Tracking**: View which document chunks informed each answer
- **Configurable**: Model selection, temperature, chunk size, retrieval depth, reasoning effort
- **System Prompt**: Customize assistant behavior

## Prerequisites

- Python 3.9+
- Sarvam AI API key ([Get 1000 free credits](https://dashboard.sarvam.ai/key-management))
- Internet connection

## Installation

```bash
# Clone the repository
git clone https://github.com/PredictiveManish/Q-A-with-sarvam.git
cd Q-A-with-sarvam

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Copy and configure .env
cp .env.example .env
```

## Usage

```bash
streamlit run app.py
```

1. Enter your Sarvam API key in the sidebar
2. (Optional) Test the API connection
3. The app uses **Sarvam-105B** (best quality) by default
4. Upload documents
5. Click **Process Documents**
6. Ask questions and get streaming answers

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Model | sarvam-30b | Choose between fast (30B) and best (105B) |
| Max Tokens | 1024 | Response length cap |
| Chunk Size | 1024 | Document chunk size for indexing |
| Temperature | 0.1 | Response randomness (0-1) |
| Top P | 0.9 | Nucleus sampling parameter |
| Top K Retrieval | 3 | Document chunks retrieved per query |
| Reasoning Effort | low | none, low, medium, high |

## Architecture

```
Upload → Chunk & Embed → Vector Index → Retrieve Context → Stream Answer
```

- **Embeddings**: FastEmbed with `BAAI/bge-small-en-v1.5`
- **Index**: LlamaIndex `VectorStoreIndex` (in-memory)
- **LLM**: Sarvam AI Chat Completions API (`/v1/chat/completions`)
- **UI**: Streamlit with chat-style interface

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API connection failed | Verify API key at [dashboard.sarvam.ai](https://dashboard.sarvam.ai/key-management) |
| Slow responses | Use sarvam-30b, lower max_tokens, or disable reasoning |
| Memory issues | Process fewer/larger chunks, reduce Top K |
| Import errors | Run `pip install -r requirements.txt` in a fresh virtual environment |

## License

MIT

## Contact

**Manish Tiwari**
- Twitter: [@compmanish](https://x.com/compmanish)
- Email: manish.tiwari.09@zohomail.in
