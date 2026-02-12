# RAG Chat Application with LandingAI ADE

A full-stack chat application that combines Retrieval-Augmented Generation (RAG) with document processing using LandingAI ADE.

## 🎯 Features

- **💬 Interactive Chat Interface** - Beautiful React-based chat UI
- **🔍 Hybrid Search** - Combines semantic and keyword search for better results
- **📄 Document Upload** - Process PDFs and images with LandingAI ADE
- **🤖 Groq LLM** - Fast and accurate responses using Llama 3.3
- **📊 Table Parsing** - Extracts data from HTML tables in documents
- **💾 Vector Database** - ChromaDB for efficient document storage and retrieval

## 📁 Project Structure

```
Improve_RAG_system_with_ADE/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── rag_system.py        # RAG logic
│   ├── hybrid_search.py     # Hybrid search module
│   ├── markdown_loader.py   # Document loader
│   ├── requirements.txt     # Python dependencies
│   ├── .env.example         # Environment variables template
│   └── chroma_db/          # Vector database (created on first run)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBox.jsx  # Main chat interface
│   │   │   └── Message.jsx  # Message component
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── api.js           # Backend API client
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   GROQ_API_KEY=your-groq-api-key-here
   LANDINGAI_ADE_API_KEY=your-landingai-ade-api-key-here
   ```
   
   - Get Groq API key: https://console.groq.com/keys
   - Get LandingAI ADE API key: Contact LandingAI

5. **Run the backend:**
   ```bash
   uvicorn main:app --reload
   ```
   
   Backend will run at: http://localhost:8000

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```
   
   Frontend will run at: http://localhost:5173

## 📖 Usage

### 1. Start Both Servers

Make sure both backend (port 8000) and frontend (port 5173) are running.

### 2. Upload a Document

- Click the 📎 button in the chat interface
- Select a PDF or image file
- Wait for processing (LandingAI ADE will parse the document)
- Document chunks will be stored in the vector database

### 3. Ask Questions

Type your questions in the chat input and press Enter. The system will:
1. Search the vector database using hybrid search
2. Retrieve relevant chunks
3. Send to Groq LLM for answer generation
4. Display the answer with sources

### Example Questions

```
What is MGRRN?
What is the SSIM value of MGRRN on Snow100K-M?
Explain the Mask Generation Module
What datasets were used for evaluation?
```

## 🔧 API Endpoints

### Backend (http://localhost:8000)

- `GET /` - API information
- `POST /chat` - Send message, get RAG response
  ```json
  {
    "message": "What is MGRRN?"
  }
  ```
- `POST /upload` - Upload document for processing
- `GET /history` - Get chat history
- `GET /stats` - Get collection statistics

## 🎨 Features Explained

### Hybrid Search

Combines two search methods:
- **Semantic Search** (Vector similarity) - Finds documents with similar meaning
- **Keyword Search** (BM25) - Finds exact keyword matches
- **RRF (Reciprocal Rank Fusion)** - Combines scores from both methods

Adjustable `alpha` parameter:
- `alpha=1.0` → Pure semantic
- `alpha=0.5` → Balanced (default)
- `alpha=0.0` → Pure keyword

### Table Parsing

Custom prompt helps the LLM parse HTML tables:
- Extracts values from `<table>`, `<tr>`, `<td>` tags
- Matches rows and columns correctly
- Handles complex table structures

### Document Processing Flow

1. User uploads file → Frontend
2. Frontend sends to `/upload` → Backend
3. Backend saves file temporarily
4. Backend calls LandingAI ADE Parse() API
5. ADE returns chunks with markdown content
6. Backend generates embeddings
7. Backend stores in ChromaDB
8. Success response to frontend

## 🛠️ Development

### Backend Development

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --log-level debug
```

### Frontend Development

```bash
cd frontend
npm run dev
```

### Build for Production

Frontend:
```bash
cd frontend
npm run build
```

## 📝 Environment Variables

### Backend (.env)

```bash
# Required
GROQ_API_KEY=your-groq-api-key
LANDINGAI_ADE_API_KEY=your-ade-api-key

# Optional (defaults shown)
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## 🐛 Troubleshooting

### Backend Issues

**"Module not found" error:**
```bash
pip install -r requirements.txt
```

**"API key not configured":**
- Check `.env` file exists
- Verify API keys are set correctly
- Restart backend server

**ChromaDB errors:**
- Delete `chroma_db/` folder and restart

### Frontend Issues

**"Cannot connect to backend":**
- Verify backend is running on port 8000
- Check CORS settings in `main.py`

**Build errors:**
```bash
rm -rf node_modules package-lock.json
npm install
```

## 📊 Tech Stack

### Backend
- **FastAPI** - Web framework
- **LangChain** - RAG orchestration
- **ChromaDB** - Vector database
- **Groq** - LLM provider
- **LandingAI ADE** - Document processing
- **HuggingFace** - Embeddings

### Frontend
- **React** - UI framework
- **Vite** - Build tool
- **Axios** - HTTP client

## 🎯 Next Steps

- [ ] Add chat history persistence (database)
- [ ] Add user authentication
- [ ] Support multiple document collections
- [ ] Add streaming responses
- [ ] Deploy to production

## 📄 License

MIT

## 👥 Authors

Ahmad Raza

---

**Built with ❤️ using RAG, Groq, and LandingAI ADE**
