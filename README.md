# CLI-based LLM Powered Search Engine

This project implements a command-line LLM-powered search engine that uses Retrieval-Augmented Generation (RAG) to process and search through multimedia content. It supports images, text documents, audio, and video files, generating transcripts and embeddings to enable natural language queries across all content. Users can place mixed media files (e.g., audio, video, text, PDFs, images) into a directory and perform searches ranging from exact filenames to vague contextual queries with the system returning the top five most relevant results. All processing is performed locally using open-source models via Ollama or other supported providers, without relying on external APIs or OC infrastructure for inference.

## Features

- **Multi-modal Content Processing**: Handle images, audio, video, text, and PDF files
- **RAG System**: Uses embeddings and vector similarity for intelligent search
- **Natural Language Queries**: Ask questions in plain English
- **OCR and Vision**: Extract text and descriptions from images
- **Speech-to-Text**: Transcribe audio and video content
- **Open Source Models**: Uses only open source models via Ollama and Hugging Face

## Supported File Types

- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`
- **Audio**: `.wav`, `.mp3`, `.m4a`, `.flac`
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv` (audio extraction)
- **Text**: `.txt`, `.md`, `.py`, `.js`, `.html`, `.css`
- **PDF**: Via pre-converted images in `pdf_images/` directory

## Setup

### Prerequisites

1. **Install Ollama**: Visit [ollama.ai](https://ollama.ai) and install Ollama
2. **Pull Required Models**:
   ```bash
   ollama pull gemma2:2b
   ollama pull minicpm-v:8b
   ```

### Installation

1. **Clone/Navigate to the project directory**
2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
3. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

## Usage

### 1. Index Your Data

First, index all files in your data directory:

```bash
python search_engine.py index
```

Or index a specific directory:

```bash
python search_engine.py index --directory /path/to/your/files
```

### 2. Search Your Content

Search with natural language queries:

```bash
# Search for animals in images
python search_engine.py search "what animals are in the images"

# Search for dialogue from audio
python search_engine.py search "dialogue from audio"

# Search for specific content
python search_engine.py search "zebra"

# Get more results
python search_engine.py search "content about Harvard" --top-k 10

# Get raw results without LLM processing
python search_engine.py search "animals" --raw
```

## Example Queries

- `"what animals can you see in the images"`
- `"dialogue from the audio files"`
- `"text content about Harvard"`
- `"zebra"`
- `"what is discussed in the audio"`
- `"content from PDF documents"`

## Sample Data

The project includes sample data in the `data/` directory:

- `animal_images/`: Images of animals (zebra, animals grid)
- `harvard.wav`: Audio file for transcription testing
- `pdf_images/`: PDF pages converted to images for OCR

## Architecture

### Components

1. **Content Processors**:
   - Image processing with MiniCPM-V vision model
   - Audio transcription with Whisper
   - Text file reading
   - PDF OCR via image conversion

2. **RAG System**:
   - Sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2`
   - Vector similarity search with cosine similarity
   - Persistent index storage

3. **LLM Integration**:
   - Gemma2:2b for generating contextual responses
   - Context-aware answer generation


## How It Works

1. **Indexing Phase**:
   - Scans all files in the specified directory
   - Extracts content based on file type
   - Generates embeddings for each document
   - Stores embeddings and metadata in a search index

2. **Search Phase**:
   - Converts user query to embedding
   - Finds most similar documents using cosine similarity
   - Uses LLM to generate contextual response based on results


## Extension Ideas

- Add support for more file types
- Implement video frame analysis
- Add support for real PDF parsing
- Multi-language support
- Web interface

