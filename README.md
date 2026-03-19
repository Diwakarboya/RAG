## LangChain + FAISS RAG App

This project is a minimal Retrieval-Augmented Generation (RAG) example using:

- **LangChain** for document loading, chunking, and retrieval
- **FAISS** as the in-memory vector database
- **Gemini** (Google Generative AI) for generation and embeddings

### 1. Setup

From the `c:\\RAG` directory:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Set your Google Generative AI API key in a `.env` file (not committed to git):

```bash
echo GOOGLE_API_KEY=your_api_key_here > .env
```

Or create `.env` manually with:

```text
GOOGLE_API_KEY=your_api_key_here
```

### 2. Usage

Prepare a plain text / markdown document, for example:

```text
docs\\example.txt
```

Run a single query:

```bash
.venv\\Scripts\\activate
python rag_app.py docs\\example.txt -q "What is this document about?"
```

Or start an interactive QA session:

```bash
.venv\\Scripts\\activate
python rag_app.py docs\\example.txt
```

### 3. How it works

1. **Load**: Reads the document from disk using `PyPDFLoader`.
2. **Chunk**: Splits it into overlapping chunks with `RecursiveCharacterTextSplitter`.
3. **Embed + Index**: Builds Gemini embeddings (`text-embedding-004`) and stores them in a FAISS vector store.
4. **Retrieve**: On each question, retrieves the most similar chunks from FAISS.
5. **Generate**: Uses `ChatGoogleGenerativeAI` (`gemini-1.5-flash`) to answer the question based on retrieved context.



