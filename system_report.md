# SAP SOP AI Trainer: System Operation Report

## 1. System Overview
The SAP SOP AI Trainer is a **Multimodal Retrieval-Augmented Generation (RAG)** system designed to assist users with Standard Operating Procedures (SOPs). Unlike traditional chatbots, this system can "see" and understand both the text content and the screenshots/diagrams within PDF documents to provide accurate, guided assistance.

## 2. Core Functionality
- **Document Ingestion**: Automatically parses PDF files, extracting native text and embedded images.
- **Multimodal Indexing**: Stores text and image "embeddings" (mathematical representations) in a high-performance vector database (**ChromaDB**).
- **Intelligent Retrieval**: When a question is asked, the system searches for the most relevant text fragments and images.
- **AI Synthesis**: Uses advanced local Large Language Models (LLMs) to read the retrieved context and generate a human-like response.

## 3. Technology Stack
| Component | Technology | Role |
| :--- | :--- | :--- |
| **Orchestration** | LlamaIndex | Handles the RAG pipeline and model integration. |
| **Local LLM** | Ollama (Llama 3) | Generates text-based answers and reasoning. |
| **Vision LLM** | Ollama (LLaVA) | Analyzes images and screenshots from the PDFs. |
| **Embeddings** | Nomic-Embed-Text | Converts text into searchable vectors. |
| **Database** | ChromaDB | Persistent storage for document indices. |
| **Frontend** | Streamlit | Provides the web-based chat interface. |

## 4. Performance & Response Time
Performance is heavily dependent on the local hardware (GPU and RAM) as the entire system runs locally for maximum privacy.

- **Ingestion Time**: ~10–30 seconds per PDF (depends on page count and image density).
- **Embedding Generation**: Fast (milliseconds per chunk).
- **Query Response Time**: **~1.5 to 3 minutes** per question.
    - *Note*: This includes the time for the Vision model to analyze retrieved screenshots and the Text model to synthesize the final answer.
- **Reliability**: The system is configured with a **600-second timeout** to ensure complex queries complete even on standard hardware.

## 5. System Requirements & Maintenance
- **Local Privacy**: No data leaves your machine; all processing is handled by Ollama.
- **Storage**: The database is stored in the `./chroma_db` folder.
- **Source Files**: Original PDFs and extracted images are kept in the `./data` folder for reference.

---
*This report was generated on 2026-03-11 based on current system configuration and verification tests.*
