# Deep Learning RAG System

A cloud-based **Retrieval-Augmented Generation (RAG)** system for answering questions about Deep Learning using a curated collection of educational notes. Built with **Qdrant Cloud** for vector storage, **Hybrid Search (Dense + Sparse)** for information retrieval, **Reciprocal Rank Fusion (RRF)** and **Cross-Encoder Reranking** to improve retrieval relevance, and **Google Gemini Flash 3.8** to generate context-aware answers. The system exposes its question-answering functionality through a **FastAPI REST API**.

## Overview

The system implements a multi-stage RAG pipeline to answer questions about Deep Learning using a curated collection of educational notes.

- **Document Processing & Chunking:** Load and prepare Deep Learning notes, then split them using three strategies: fixed-size, sentence-based, and paragraph-based chunking.
- **Cloud Vector Storage:** Store document embeddings in a Qdrant Cloud Cluster.
- **Hybrid Retrieval:** Combine dense semantic search and sparse keyword-based search to retrieve relevant documents.
- **Result Fusion:** Apply Reciprocal Rank Fusion (RRF) to merge and rank results from both retrieval methods.
- **Cross-Encoder Reranking:** Evaluate query-document relevance using a pretrained Cross-Encoder model and rerank retrieved candidates.
- **Answer Generation:** Pass the top relevant documents as context to Google Gemini Flash 3.8 to generate answers.
- **REST API:** Expose the complete question-answering pipeline through a FastAPI application.

## Architecture

```text
              Deep Learning Notes
                       |
                       v
               Document Processing
                       |
                       v
                Text Chunking
                       |
          +------------+------------+
          |            |            |
        Fixed       Sentence     Paragraph
          |            |            |
          +------------+------------+
                       |
                       v
              Embedding Generation
                       |
                       v
               Qdrant Cloud Cluster
                       |
                       v
                   User Query
                       |
              +--------+--------+
              |                 |
         Dense Search      Sparse Search
              |                 |
              +--------+--------+
                       |
                       v
          Reciprocal Rank Fusion (RRF)
                       |
                       v
              Candidate Documents
                       |
                       v
             Cross-Encoder Reranking
                       |
                       v
                Top Relevant Docs
                       |
                       v
                Gemini Flash 3.8
                       |
                       v
                Generated Answer
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Package Management | uv |
| API Framework | FastAPI |
| Vector Database | Qdrant Cloud |
| Embedding Models | Sentence Transformers |
| Dense Retrieval | Vector Similarity Search |
| Sparse Retrieval | BM25 |
| Retrieval Strategy | Hybrid Search |
| Result Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Sentence Transformers Cross-Encoder |
| Reranker Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Google Gemini Flash 3.8 |


## Chunking Strategies

The project supports three chunking approaches to explore how document segmentation affects retrieval:

| Strategy | Description |
|---|---|
| Fixed-Size Chunking | Splits text into chunks based on a predefined size. |
| Sentence-Based Chunking | Groups text according to sentence boundaries. |
| Paragraph-Based Chunking | Preserves paragraph-level context when splitting documents. |

## Project Structure

```text
deep-learning-rag/
│
├── data/
│
├── scripts/
│
├── src/
│   └── rag_app/
│       ├── api/
│       ├── core/
│       ├── generation/
│       ├── ingestion/
│       ├── retrieval/
│       ├── schemas/
│       ├── services/
│       └── main.py
│
├── tests/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting Started

### Prerequisites

- Python
- [uv](https://docs.astral.sh/uv/)
- Qdrant Cloud account and API credentials
- Google Gemini API key

### 1. Clone the Repository

```bash
git clone https://github.com/MohamedAkramZidan/deep-learning-rag-assistant.git

cd deep-learning-rag
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Run the Application

```bash
uv run uvicorn rag_app.main:app --reload
```

### 5. Access API Documentation

Once the application is running, open:

http://127.0.0.1:8000/docs

## Future Improvements

- Evaluate retrieval quality using Recall@K, MRR, and MAP.
- Compare embedding models and retrieval configurations.
- Experiment with different chunking strategies.
- Evaluate answer quality and response faithfulness.
- Add a user-friendly interface.


*Built to explore practical RAG architectures, hybrid information retrieval, reranking, vector databases, and LLM integration.*