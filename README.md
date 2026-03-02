# IN'SMART

Educational RAG-based assistant platform for processing and querying course materials.

## Overview

IN'SMART is a Retrieval-Augmented Generation (RAG) system designed to help students interact with their course materials through an intelligent chatbot interface. The platform supports multiple input formats (PDFs, audio recordings, handwritten notes) and provides semantic search capabilities over the processed content.

## Architecture

The system is built with a microservices architecture using Docker:

- **FastAPI Backend**: RESTful API for document processing, ingestion, and chat functionality
- **Streamlit Frontend**: Web interface for document upload and chat interaction
- **Weaviate**: Vector database for storing and searching document embeddings
- **MinIO**: Object storage for original document files

## Project Structure

```
IN-SMART/
├── api/                    # FastAPI backend
│   ├── storage/            # Database connectors (Weaviate, MinIO)
│   ├── processing/         # Document processing (audio, text, OCR)
│   ├── services/           # Business logic (chunking, embeddings, ingestion, LLM, retrieval)
│   └── models/             # Tests and evaluation scripts
├── client/                 # Streamlit frontend
├── docker/                 # Dockerfiles for services
├── docker-compose.yml      # Docker orchestration
└── pyproject.toml          # UV workspace configuration
```

## Prerequisites

- Docker and Docker Compose
- UV (Python package manager)
- Python 3.10 or higher
- Mistral API key

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd IN-SMART
```

### 2. Configure environment variables

Create a `.env` file by copying the example template:

```bash
cp .env.example .env
```

Then edit `.env` and replace `your_mistral_api_key_here` with your actual Mistral API key. The file should contain:

```bash
# Mistral API Configuration
MISTRAL_API_KEY=your_mistral_api_key_here

# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_CLASS_NAME=DocumentChunk

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=in-smart-documents
MINIO_SECURE=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Client Configuration
API_URL=http://localhost:8000

# Chunking Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Retrieval Configuration
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# LLM Configuration
MISTRAL_MODEL=mistral-medium
MISTRAL_TEMPERATURE=0.7
MISTRAL_MAX_TOKENS=2000

# Embedding Configuration
EMBEDDING_MODEL=mistral-embed
```

**Important**: Replace `your_mistral_api_key_here` with your actual Mistral API key.

### 3. Start services with Docker Compose

```bash
docker-compose up -d
```

This will start:
- Weaviate on `http://localhost:8080`
- MinIO on `http://localhost:9000` (API) and `http://localhost:9001` (Console)
- FastAPI on `http://localhost:8000`
- Streamlit on `http://localhost:8501`

### 4. Initialize MinIO bucket

Access MinIO console at `http://localhost:9001` (login: minioadmin/minioadmin) and create the bucket specified in your `.env` file (default: `in-smart-documents`).

## Development

### Local Development Setup

#### API Development

```bash
cd api
uv sync
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Client Development

```bash
cd client
uv sync
uv run streamlit run main.py --server.port 8501
```

### Using UV

This project uses UV for dependency management. Key commands:

```bash
# Install dependencies
uv sync

# Add a dependency
uv add <package-name>

# Run a command in the environment
uv run <command>

# Activate virtual environment
source .venv/bin/activate  # After uv sync
```

## API Endpoints

### Health Check
- `GET /health` - API health status

### Document Management
- `POST /api/documents/upload` - Upload a document (PDF, audio, image)
- `GET /api/documents` - List all documents
- `GET /api/documents/{document_id}` - Get document metadata
- `DELETE /api/documents/{document_id}` - Delete a document

### Chat
- `POST /api/chat` - Send a message and get AI response
- `GET /api/chat/history` - Get chat history

### Ingestion
- `POST /api/ingest/{document_id}` - Process and ingest a document

## Usage

1. **Upload Documents**: Navigate to the upload page and upload your course materials (PDFs, audio files, or images of handwritten notes)

2. **Wait for Processing**: Documents are automatically processed (STT for audio, OCR for images, text extraction for PDFs)

3. **Chat**: Use the chat interface to ask questions about your course materials. The system will retrieve relevant chunks and generate answers using the Mistral LLM

## Testing

Run tests from the API directory:

```bash
cd api
uv run pytest models/tests/
```

## Evaluation

Evaluation scripts are available in `api/models/eval/`:

- `eval_retrieval.py`: Evaluate retrieval quality (Precision@K, Recall@K)
- `eval_generation.py`: Evaluate LLM response quality and document usage

## Technologies

- **FastAPI**: Web framework for building APIs
- **Streamlit**: Frontend framework for web applications
- **Weaviate**: Vector database for semantic search
- **MinIO**: Object storage for documents
- **Mistral AI**: LLM, embeddings, and OCR/STT services
- **UV**: Python package and environment manager
- **Docker**: Containerization and orchestration

## Contributing

This project follows a Git flow workflow:
- `main`: Production-ready code
- `dev`: Development branch
- `feature/*`: Feature branches
- `fix/*`: Bug fix branches

## License

[Specify license]

## Authors

EL HABI Aymane, ESSALHI Ayoub, ESSAKI Mehdi, FOUSSARD Nicolas, KEOVILAY Lyam, TRICOT Baptiste
