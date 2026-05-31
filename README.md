# RAG Model - Retrieval-Augmented Generation

A comprehensive implementation of a Retrieval-Augmented Generation (RAG) system that combines embeddings, vector database search, context retrieval, and LLM responses.

## 🎯 Overview

This project implements a complete RAG pipeline with the following components:

- **Embeddings**: Text vectorization using SentenceTransformer
- **Vector Store**: Efficient similarity search using ChromaDB
- **Context Retrieval**: Intelligent document ranking and context building
- **LLM Response**: Response generation using OpenAI API
- **Evaluation**: RAG performance evaluation metrics

## 📁 Project Structure

```
rag-model/
├── embeddings.py          # Text embedding generation
├── vector_store.py        # Vector database operations
├── context_retrieval.py   # Document retrieval and ranking
├── llm_response.py        # LLM response generation
├── rag_pipeline.py        # Main orchestration and evaluation
├── main.py                # Example usage and demonstration
├── sample_data.json       # Sample documents for testing
├── requirements.txt       # Project dependencies
├── .env.example           # Environment configuration template
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yoursgaurav95-sudo/Rag-model.git
cd Rag-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Usage

Run the demonstration:
```bash
python main.py
```

## 🔧 Components

### 1. Embeddings (`embeddings.py`)
Converts text to vector representations using pre-trained models.

```python
from embeddings import EmbeddingModel

model = EmbeddingModel("all-MiniLM-L6-v2")
embeddings = model.encode(["Hello world", "How are you?"])
```

### 2. Vector Store (`vector_store.py`)
Stores and retrieves embeddings efficiently using ChromaDB.

```python
from vector_store import VectorStore

store = VectorStore(collection_name="documents")
store.add_documents([
    {"id": "1", "content": "Document content", "metadata": {}}
])
results = store.search("query", top_k=5)
```

### 3. Context Retrieval (`context_retrieval.py`)
Retrieves relevant documents and builds context for the LLM.

```python
from context_retrieval import ContextRetriever

retriever = ContextRetriever(vector_store)
context, sources = retriever.retrieve_and_build_context("query", top_k=5)
```

### 4. LLM Response (`llm_response.py`)
Generates responses using OpenAI API with retrieved context.

```python
from llm_response import ResponseGenerator

generator = ResponseGenerator(api_key="sk-...")
response = generator.generate_response(query, context)
```

### 5. RAG Pipeline (`rag_pipeline.py`)
Orchestrates all components into a unified pipeline.

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(openai_api_key="sk-...")
pipeline.ingest_documents(documents)
result = pipeline.query("What is machine learning?")
```

## 📊 Features

### Document Ingestion
- Add documents from Python lists
- Import from JSON files
- Batch processing support

### Retrieval
- Semantic similarity search
- Metadata filtering
- Result re-ranking
- Relevance threshold filtering

### Response Generation
- Context-aware LLM responses
- Source attribution
- Customizable system prompts
- Multiple prompt templates (QA, Summary, Extraction, Classification)

### Evaluation
- Precision, Recall, F1-Score metrics
- Batch evaluation support
- Performance tracking

## 💡 Example Usage

```python
import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline, RAGEvaluator

load_dotenv()

# Initialize pipeline
pipeline = RAGPipeline(os.getenv("OPENAI_API_KEY"))

# Ingest documents
documents = [
    {
        "id": "1",
        "content": "Python is a programming language",
        "metadata": {"category": "programming"}
    }
]
pipeline.ingest_documents(documents)

# Query the pipeline
result = pipeline.query("What is Python?")
print(result["response"])
print(f"Sources: {result['sources']}")

# Evaluate performance
evaluator = RAGEvaluator(pipeline)
metrics = evaluator.evaluate_retrieval("Python", ["1"], top_k=5)
print(f"F1 Score: {metrics['f1_score']}")
```

## 🎓 Advanced Usage

### Custom LLM Configuration
```python
from llm_response import LLMConfig, ResponseGenerator

config = LLMConfig(
    model="gpt-4",
    temperature=0.5,
    max_tokens=1000
)
generator = ResponseGenerator(api_key, config)
```

### Batch Processing
```python
queries = ["Query 1", "Query 2", "Query 3"]
results = pipeline.batch_query(queries, top_k=5)
```

### Using Prompt Templates
```python
from llm_response import PromptTemplate

summary = PromptTemplate.summary_template(context)
classification = PromptTemplate.classification_template(context, ["category1", "category2"])
```

## 🔍 Performance Optimization

- **Embedding Caching**: Avoid recomputing embeddings
- **Batch Ingestion**: Process multiple documents efficiently
- **Vector Index**: HNSW indexing for fast similarity search
- **Context Limiting**: Manage token limits with max_length parameter

## 📈 Metrics

The system provides evaluation metrics:
- **Precision**: Relevance of retrieved documents
- **Recall**: Coverage of relevant documents
- **F1 Score**: Harmonic mean of precision and recall

## 🛠️ Customization

### Add Custom Embedding Models
```python
from embeddings import EmbeddingModel

# Use different pre-trained models
model = EmbeddingModel("sentence-transformers/all-mpnet-base-v2")
```

### Implement Custom Retrievers
Extend `ContextRetriever` for domain-specific retrieval logic.

### Create Custom Prompts
Use `PromptTemplate` class to define custom prompt templates.

## 📝 Dependencies

- `langchain`: LLM orchestration
- `openai`: OpenAI API access
- `chromadb`: Vector database
- `sentence-transformers`: Embedding models
- `pydantic`: Data validation
- `python-dotenv`: Environment configuration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note**: Make sure to set up your OPENAI_API_KEY in the `.env` file before running the pipeline.
