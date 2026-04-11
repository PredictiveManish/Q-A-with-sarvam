# 📚 PDF Q&A with Sarvam AI

An interactive Streamlit application that enables intelligent question-answering over PDF documents using Sarvam AI's powerful language models. Upload your PDFs, ask questions, and get accurate answers based on the document content.

> ### 💡 Note

This application requires a **Sarvam AI API key**. You can get **1000 free credits** by signing up at the [Sarvam AI Dashboard](https://dashboard.sarvam.ai).

## ✨ Features

- **PDF Document Processing**: Upload and process multiple PDF files
- **Intelligent Q&A**: Ask questions about your documents and get contextual answers
- **Sarvam AI Integration**: Leverages Sarvam AI's language models for high-quality responses
- **Vector Search**: Uses embeddings for semantic search and relevant content retrieval
- **Customizable Settings**: 
  - Adjust context window size
  - Control response token length
  - Modify chunk size for document processing
  - Configure retrieval parameters (Top K)
  - Set temperature for response creativity
- **Source Tracking**: View source documents and relevant passages
- **System Prompt Customization**: Define assistant behavior and response guidelines

## 🚀 Live

[Use here](https://sarvam-pdf-bot.streamlit.app/)
<img width="1919" height="887" alt="Screenshot 2026-03-22 225658" src="https://github.com/user-attachments/assets/2d88dd28-50d8-4f9f-98fb-31247413f6a4" />

## 📋 Prerequisites

- Python 3.8 or higher
- Sarvam AI API key ([Get your free credits here](https://dashboard.sarvam.ai/key-management))
- Internet connection for API access

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pdf-qa-sarvam.git
cd pdf-qa-sarvam
```
2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Dependencies
The application requires the following packages:

text
streamlit
llama-index
llama-index-embeddings-fastembed
requests
PyPDF2
You can install them directly:
```bash
pip install streamlit llama-index llama-index-embeddings-fastembed requests
```
Usage
Start the Streamlit application
```bash
streamlit run app.py
```
### Configure the application (sidebar)
- Enter your Sarvam AI API key
- Optionally modify the base URL (default: https://api.sarvam.ai)
- Adjust model settings as needed
- Customize the system prompt for assistant behavior
- Upload Documents
- Click "Browse files" to select PDF documents
- Multiple files can be uploaded simultaneously
- Process Documents
- Click "Process Documents" to index your PDFs
- Wait for the processing to complete
- Ask Questions
- Enter your question in the text input field
- Click "Get Answer" to receive a response
- View source documents in the expandable section

### ⚙️ Configuration Options

**Model Settings**
- Context Window Size: 1024-8192 tokens (default: 4500)
- Max Response Tokens: 64-2048 tokens (default: 512)
- Chunk Size: 256-4096 tokens (default: 1024)

**Advanced Options**
- Temperature: 0.0-1.0 (default: 0.1) - Controls response randomness
- Top K Retrieval: 1-10 (default: 3) - Number of document chunks to retrieve

**System Prompt**
Customize the assistant's behavior and response guidelines. Default prompt:

```text
You are a helpful Q&A assistant. Answer questions based only on the provided documents. 
If the answer is not in the documents, say "I cannot find this information in the provided documents."
Provide clear, concise answers with relevant details from the documents.
```
### 🏗️ Architecture

The application is built using:

- **Frontend**: Streamlit for interactive UI
- **Document Processing**: LlamaIndex for document indexing and retrieval
- **Embeddings**: FastEmbed with BAAI/bge-small-en-v1.5
- **Language Model**: Custom SarvamAI LLM wrapper for Sarvam AI API
- **Vector Storage**: In-memory vector store

#### Workflow

| Step | Description |
|------|-------------|
| **Upload** | PDF files are uploaded and temporarily stored |
| **Processing** | Documents are chunked and embedded using FastEmbed |
| **Indexing** | VectorStoreIndex creates searchable embeddings |
| **Querying** | User questions trigger semantic search and LLM response generation |
| **Response** | Answers with source references are displayed |

---

### 🔧 Troubleshooting

#### Common Issues

<details>
<summary><b>API Connection Error</b></summary>

- Verify your API key is correct
- Check internet connection
- Ensure the base URL is correct (default: `https://api.sarvam.ai`)
</details>

<details>
<summary><b>Document Processing Failed</b></summary>

- Ensure PDF files are not corrupted
- Check that the file size is reasonable (< 10MB recommended)
- Verify chunk size isn't too large for your system memory
</details>

<details>
<summary><b>Slow Responses</b></summary>

- Reduce chunk size for faster processing
- Lower the Top K retrieval value
- Decrease max response tokens
</details>

<details>
<summary><b>Memory Issues</b></summary>

- Process documents in smaller batches
- Reduce chunk size
- Clear session and restart the app
</details>

---

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/AmazingFeature

# Commit your changes
git commit -m 'Add some AmazingFeature'

# Push to the branch
git push origin feature/AmazingFeature

# Open a Pull Request
```
### 🙏 Acknowledgments

- **[Sarvam AI](https://sarvam.ai)** — for providing the language model API
- **[LlamaIndex](https://www.llamaindex.ai)** — for document indexing and retrieval framework
- **[Streamlit](https://streamlit.io)** — for the interactive web interface
- **[FastEmbed](https://github.com/qdrant/fastembed)** — for efficient embeddings

---

### 📧 Contact

**Manish Tiwari**

- 🐦 Twitter: [@compmanish](https://x.com/compmanish)
- 📧 Email: [Mail](mailto:manish.tiwari.09@zohomail.in)
- 🔗 Project Link: [https://github.com/predictivemanish/pdf-qa-sarvam](https://github.com/predictivemanish/pdf-qa-sarvam)

---


