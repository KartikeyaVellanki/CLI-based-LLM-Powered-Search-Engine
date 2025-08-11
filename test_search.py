#!/usr/bin/env python3
"""
Test script for the CLI Search Engine
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Command: {cmd}")
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def test_search_engine():
    """Test the search engine functionality"""
    print("=" * 50)
    print("Testing CLI Search Engine")
    print("=" * 50)
    
    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("Data directory not found. Please ensure data/ directory exists.")
        return False
    
    print("✅ Data directory found")
    
    # Test indexing
    print("\n1. Testing indexing...")
    if run_command("python search_engine.py index"):
        print("Indexing completed successfully")
    else:
        print("Indexing failed")
        return False
    
    # Check if index file was created
    if Path("search_index.pkl").exists():
        print("Search index file created")
    else:
        print("Search index file not found")
        return False
    
    # Test search queries
    test_queries = [
        "animals",
        "what animals are in the images",
        "audio content",
        "zebra"
    ]
    
    print("\n2. Testing search queries...")
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        if run_command(f'python search_engine.py search "{query}" --raw'):
            print(f"Query '{query}' executed successfully")
        else:
            print(f"Query '{query}' failed")
    
    print("\n3. Testing LLM response...")
    if run_command('python search_engine.py search "what animals can you see" --top-k 3'):
        print("LLM response test completed")
    else:
        print("LLM response test failed")
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_search_engine()
