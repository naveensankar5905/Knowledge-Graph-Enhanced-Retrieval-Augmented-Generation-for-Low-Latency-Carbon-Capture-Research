# Knowledge Graph-Enhanced Retrieval Augmented Generation for Low-Latency Carbon Capture Research

A sophisticated Knowledge Graph-based Question Answering (QA) system that extracts information from research papers, builds knowledge graphs, generates Q&A pairs, and evaluates performance using comprehensive metrics.

## 🚀 Features

- **PDF Text Extraction**: Extracts and chunks text from research papers with section-aware processing
- **Knowledge Graph Construction**: Builds entity-relationship graphs using Google Gemini AI
- **Automated Q&A Generation**: Creates both short (20 words) and long (60-90 words) Q&A pairs
- **Advanced Retrieval**: Multi-stage retrieval with semantic search and graph expansion
- **Comprehensive Evaluation**: Dual-phase evaluation (Pre-RAG and Post-RAG) with 7 metrics
- **Interactive Visualization**: HTML-based knowledge graph visualization

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Documentation](#documentation)

## 🏗️ Architecture

```
┌─────────────────┐
│  PDF Documents  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  1. Text Extraction     │  → extracted_texts.json
│     (extract_text.py)   │  → extracted_texts_chunked.json
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  2. Knowledge Graph     │  → knowledge_graph.json
│     Build (build_graph) │  → knowledge_graph_visualization.html
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  3. Q&A Generation      │  → qa_short.json
│     (generate_qa.py)    │  → qa_long.json
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  4. Evaluation          │  → evaluations.json
│     (evaluate.py)       │  → evaluation_details.json
└─────────────────────────┘
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Google Gemini API Key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/naveensankar5905/Knowledge-Graph-Enhanced-Retrieval-Augmented-Generation-for-Low-Latency-Carbon-Capture-Research.git
cd Knowledge-Graph-Enhanced-Retrieval-Augmented-Generation-for-Low-Latency-Carbon-Capture-Research
```

2. Install required packages:
```bash
pip install pymupdf google-generativeai python-dotenv sentence-transformers scikit-learn rouge-score nltk numpy
```

3. Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

4. Create a `raw/` folder and place your PDF research papers:
```bash
mkdir raw
# Copy your PDF files to the raw/ folder
```

## 🎯 Usage

### Run Complete Pipeline

Execute all steps sequentially:

```bash
python main.py
```

This will:
1. Extract text from PDFs
2. Build knowledge graph
3. Generate Q&A pairs
4. Evaluate performance

### Run Individual Steps

```bash
# Step 1: Extract text from PDFs
python extract_text.py

# Step 2: Build knowledge graph
python build_graph.py

# Step 3: Generate Q&A pairs
python generate_qa.py

# Step 4: Evaluate system performance
python evaluate.py
```

## 🔄 Pipeline Overview

### 1. Text Extraction (`extract_text.py`)
- Extracts text from PDF documents
- Performs section-aware chunking (500-word chunks)
- Preserves document structure and metadata

### 2. Knowledge Graph Building (`build_graph.py`)
- Extracts entities and relationships using Gemini AI
- Constructs entity-relationship graph
- Generates interactive HTML visualization

### 3. Q&A Generation (`generate_qa.py`)
- Creates diverse question types (factual, comparative, analytical)
- Generates both short and long answers
- Covers multiple sections per document

### 4. Evaluation (`evaluate.py`)
- **Pre-RAG**: Evaluates retrieval performance
- **Post-RAG**: Evaluates generation quality
- Uses 7 comprehensive metrics

## 📊 Evaluation Metrics

The system uses 7 complementary metrics to assess performance:

1. **Semantic Similarity** (0.0-1.0): Measures conceptual similarity using embeddings
2. **ROUGE-1 F1**: Measures word-level overlap
3. **Precision**: Ratio of relevant content retrieved
4. **Recall**: Coverage of reference answer
5. **F1 Score**: Harmonic mean of precision and recall
6. **BLEU Score**: N-gram overlap quality
7. **Exact Match**: Binary exact match score

### Two-Phase Evaluation

- **Pre-RAG (Retrieval)**: Compares retrieved chunks against reference answers
- **Post-RAG (Generation)**: Compares LLM-generated answers against reference answers

## 📁 Output Files

All output files are saved in the `output/` directory:

| File | Description |
|------|-------------|
| `extracted_texts.json` | Raw extracted text from PDFs |
| `extracted_texts_chunked.json` | Chunked text with metadata |
| `knowledge_graph.json` | Entity-relationship graph structure |
| `knowledge_graph_visualization.html` | Interactive graph visualization |
| `qa_short.json` | Short Q&A pairs (max 20 words) |
| `qa_long.json` | Long Q&A pairs (60-90 words) |
| `evaluations.json` | Aggregate evaluation metrics |
| `evaluation_details.json` | Per-question detailed results |

## 📂 Project Structure

```
.
├── main.py                          # Pipeline orchestrator
├── extract_text.py                  # PDF text extraction
├── build_graph.py                   # Knowledge graph construction
├── generate_qa.py                   # Q&A pair generation
├── evaluate.py                      # Performance evaluation
├── EVALUATION_DOCUMENTATION.txt     # Detailed evaluation docs
├── .env                            # API keys (not in repo)
├── .gitignore                      # Git ignore rules
├── raw/                            # Input PDF files
│   └── *.pdf
└── output/                         # Generated outputs
    ├── extracted_texts.json
    ├── extracted_texts_chunked.json
    ├── knowledge_graph.json
    ├── knowledge_graph_visualization.html
    ├── qa_short.json
    ├── qa_long.json
    ├── evaluations.json
    └── evaluation_details.json
```

## 📖 Documentation

For detailed documentation on the evaluation system, see:
- [`EVALUATION_DOCUMENTATION.txt`](EVALUATION_DOCUMENTATION.txt) - Comprehensive evaluation guide

## 🔍 Key Features

### Advanced Retrieval System
- **Multi-stage retrieval**: Entity matching → Semantic search → Graph expansion
- **Graph expansion**: Traverses knowledge graph edges for related information
- **Semantic embeddings**: Uses SentenceTransformer for meaning-based retrieval

### LLM-Powered Generation
- **Model**: Google Gemini 2.0 Flash
- **Context-aware**: Generates answers based on retrieved chunks
- **Length control**: Separate strategies for short vs long answers

### Comprehensive Evaluation
- Evaluates both retrieval and generation
- Multiple metrics for holistic assessment
- Per-question detailed analysis for debugging

## 🛠️ Technologies Used

- **Python 3.8+**
- **PyMuPDF**: PDF text extraction
- **Google Gemini AI**: Entity extraction and answer generation
- **SentenceTransformers**: Semantic embeddings
- **scikit-learn**: Similarity computations
- **ROUGE & NLTK**: Evaluation metrics
- **NumPy**: Numerical operations

## 📝 Configuration

Key parameters can be adjusted in `evaluate.py`:

```python
# Retrieval settings
top_k = 5                    # Number of chunks to retrieve
similarity_threshold = 0.3   # Minimum relevance score
use_graph_expansion = True   # Enable graph traversal
max_hops = 1                 # Graph expansion depth

# Generation settings
temperature = 0.2            # LLM randomness (lower = more deterministic)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Naveen Sankar**

- GitHub: [@naveensankar5905](https://github.com/naveensankar5905)

## 🙏 Acknowledgments

- Google Gemini API for entity extraction and answer generation
- Research community for carbon capture domain knowledge
- Open-source libraries that made this project possible

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ **Star this repository if you find it helpful!**
