#!/usr/bin/env python3
"""
CLI-based LLM Powered Search Engine
A RAG system that processes multimedia content and enables natural language search
"""

import argparse
import os
import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Import existing modules
from transformer_embedding import get_embedding
from transformer_cosine_similarity import get_embedding as get_embedding_similarity, cosine_similarity
from transformer_whisper import transcribe
from pdf_processor import extract_text_from_pdf_images
import ollama
import base64
import io
from PIL import Image
import numpy as np
from transformers import pipeline

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self, data_dir: str = "data", index_file: str = "search_index.pkl"):
        self.data_dir = Path(data_dir)
        self.index_file = index_file
        self.index = self.load_index()
        self.embedding_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
        self.llm_model = "gemma3:1b"  # Using a lightweight model for responses, suggested by Sr. Developer
        
    def load_index(self) -> Dict:
        """Load existing search index or create new one"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    logger.info(f"Loading existing index from {self.index_file}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load index: {e}. Creating new index.")
        return {"documents": [], "embeddings": [], "metadata": []}
    
    def save_index(self):
        """Save search index to disk"""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.index, f)
            logger.info(f"Index saved to {self.index_file}")
        except Exception as e:
            logger.error(f"Could not save index: {e}")
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for vision model"""
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def process_image(self, file_path: str) -> str:
        """Extract text/description from images using vision model"""
        try:
            image_base64 = self.image_to_base64(file_path)
            response = ollama.generate(
                model='minicpm-v:8b',
                prompt='Describe this image in detail, including any text, objects, people, and scenes you can see.',
                images=[image_base64]
            )
            return response['response']
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return f"Image file: {os.path.basename(file_path)}"
    
    def process_audio(self, file_path: str) -> str:
        """Transcribe audio files using Whisper"""
        try:
            asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
            result = asr(file_path)
            return result["text"]
        except Exception as e:
            logger.error(f"Error processing audio {file_path}: {e}")
            return f"Audio file: {os.path.basename(file_path)}"
    
    def process_text(self, file_path: str) -> str:
        """Read text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return f"Text file: {os.path.basename(file_path)}"
    
    def process_video(self, file_path: str) -> str:
        """Extract audio from video and transcribe (simplified approach)"""
        # Returning metadata
        # But would actually need to extract audio track and process it
        return f"Video file: {os.path.basename(file_path)} - Video content processing not fully implemented"
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the sentence transformer"""
        embeddings = self.embedding_model(text)
        return np.mean(embeddings[0], axis=0)
    
    def process_file(self, file_path: str) -> Tuple[str, str]:
        """Process a file and return its content and type"""
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
            content = self.process_image(str(file_path))
            file_type = 'image'
        elif file_ext in ['.wav', '.mp3', '.m4a', '.flac']:
            content = self.process_audio(str(file_path))
            file_type = 'audio'
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            content = self.process_video(str(file_path))
            file_type = 'video'
        elif file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css']:
            content = self.process_text(str(file_path))
            file_type = 'text'
        elif file_ext == '.pdf':
            # For PDFs, look for  images in pdf_images/ directory
            pdf_images_dir = self.data_dir / "pdf_images"
            if pdf_images_dir.exists():
                content = extract_text_from_pdf_images(str(pdf_images_dir))
            else:
                content = f"PDF file: {file_path.name} (images not found in pdf_images/)"
            file_type = 'pdf'
        else:
            content = f"Unknown file type: {file_path.name}"
            file_type = 'unknown'
        
        return content, file_type
    
    def index_directory(self, directory: str = None):
        """Index all files in the data directory"""
        if directory is None:
            directory = self.data_dir
        
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory {directory} does not exist")
            return
        
        logger.info(f"Indexing directory: {directory}")
        
        # Clears existing index
        self.index = {"documents": [], "embeddings": [], "metadata": []}
        
        # Process all files recursively
        for file_path in directory.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                logger.info(f"Processing: {file_path}")
                
                try:
                    content, file_type = self.process_file(file_path)
                    
                    # Get embedding for the content
                    embedding = self.get_text_embedding(content)
                    
                    # Add to index
                    self.index["documents"].append(content)
                    self.index["embeddings"].append(embedding)
                    self.index["metadata"].append({
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_type,
                        "file_size": file_path.stat().st_size
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Indexed {len(self.index['documents'])} documents")
        self.save_index()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for documents similar to the query"""
        if not self.index["documents"]:
            logger.warning("No documents in index. Run 'index' command first.")
            return []
        
        # Get query embedding
        query_embedding = self.get_text_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.index["embeddings"]):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Get top k results
        results = []
        for similarity, idx in similarities[:top_k]:
            results.append({
                "similarity": similarity,
                "content": self.index["documents"][idx],
                "metadata": self.index["metadata"][idx]
            })
        
        return results
    
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate a response using the LLM based on search results"""
        if not search_results:
            return "No relevant documents found for your query."
        
        # Prepare context from search results
        context = "Based on the following relevant documents:\n\n"
        for i, result in enumerate(search_results, 1):
            metadata = result["metadata"]
            content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
            context += f"{i}. File: {metadata['file_name']} (Type: {metadata['file_type']})\n"
            context += f"   Content: {content_preview}\n"
            context += f"   Similarity: {result['similarity']:.3f}\n\n"
        
        prompt = f"""
{context}

User Query: {query}

Please provide a helpful response based on the relevant documents above. 
If the query asks for specific information, extract it from the documents.
If no relevant information is found, say so clearly.
Keep your response concise but informative.
"""
        
        try:
            response = ollama.generate(model=self.llm_model, prompt=prompt)
            return response['response']
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error generating response. Here are the relevant files found:\n" + "\n".join([
                f"- {r['metadata']['file_name']} (similarity: {r['similarity']:.3f})" 
                for r in search_results
            ])

def main():
    parser = argparse.ArgumentParser(
        description="CLI-based LLM Powered Search Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s index                                    # Index all files in data/
  %(prog)s index --directory /path/to/files        # Index specific directory
  %(prog)s search "what animals are in the images" # Search with natural language
  %(prog)s search "dialogue from audio"            # Find audio transcripts
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index files for searching')
    index_parser.add_argument('--directory', '-d', default='data', 
                            help='Directory to index (default: data)')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search indexed files')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', '-k', type=int, default=5,
                             help='Number of results to return (default: 5)')
    search_parser.add_argument('--raw', action='store_true',
                             help='Return raw search results without LLM response')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize search engine
    engine = SearchEngine()
    
    if args.command == 'index':
        engine.index_directory(args.directory)
        print(f"Indexing completed. {len(engine.index['documents'])} documents indexed.")
    
    elif args.command == 'search':
        results = engine.search(args.query, args.top_k)
        
        if not results:
            print("No relevant documents found.")
            return
        
        if args.raw:
            # Print raw results
            print(f"Top {len(results)} results for: '{args.query}'\n")
            for i, result in enumerate(results, 1):
                metadata = result["metadata"]
                print(f"{i}. {metadata['file_name']} (Type: {metadata['file_type']})")
                print(f"   Similarity: {result['similarity']:.3f}")
                print(f"   Path: {metadata['file_path']}")
                content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
                print(f"   Content: {content_preview}")
                print()
        else:
            # Now generate and print LLM response
            response = engine.generate_response(args.query, results)
            print(f"Query: {args.query}\n")
            print("Response:")
            print(response)
            print(f"\n--- Based on {len(results)} relevant documents ---")

if __name__ == "__main__":
    main()
