#!/usr/bin/env python3
"""
PDF Processor for the Search Engine
Processes PDF files by extracting text from converted images
"""

import os
import ollama
import base64
from PIL import Image
import io
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 for vision model"""
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_text_from_pdf_images(pdf_images_dir: str) -> str:
    """Extract text from PDF page images using OCR"""
    pdf_images_path = Path(pdf_images_dir)
    
    if not pdf_images_path.exists():
        return f"PDF images directory not found: {pdf_images_dir}"
    
    all_text = []
    image_files = sorted([f for f in pdf_images_path.iterdir() 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    for image_file in image_files:
        try:
            image_base64 = image_to_base64(str(image_file))
            response = ollama.generate(
                model='minicpm-v:8b',
                prompt='Extract all text from this image. Return only the text content, no descriptions.',
                images=[image_base64]
            )
            page_text = response['response']
            all_text.append(f"Page {image_file.stem}: {page_text}")
            logger.info(f"Extracted text from {image_file}")
        except Exception as e:
            logger.error(f"Error processing PDF page {image_file}: {e}")
            all_text.append(f"Page {image_file.stem}: [Error extracting text]")
    
    return "\n\n".join(all_text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pdf_processor.py <pdf_images_directory>")
        sys.exit(1)
    
    pdf_dir = sys.argv[1]
    text = extract_text_from_pdf_images(pdf_dir)
    print(text)
