#!/usr/bin/env python3
"""
Demo script showing the CLI Search Engine capabilities
"""

import subprocess
import time

def run_demo():
    """Run a demonstration of the search engine"""
    
    print("CLI-based LLM Powered Search Engine Demo")
    print("=" * 50)
    
    demo_queries = [
        ("Finding animals in images", "what animals are in the images"),
        ("Searching for specific animal", "zebra"),
        ("Audio content search", "what is said in the audio"),
        ("Physics content search", "physics formulas"),
        ("Chemistry content search", "periodic table")
    ]
    
    for description, query in demo_queries:
        print(f"\n {description}")
        print(f"Query: '{query}'")
        print("-" * 30)
        
        try:
            result = subprocess.run(
                f'python search_engine.py search "{query}"',
                shell=True,
                capture_output=True,
                text=True,
                cwd="/Users/kartikeyavellanki/Desktop/CLI-LLM Search Engine"
            )
            
            if result.returncode == 0:
                # Extract just the response part
                output_lines = result.stdout.strip().split('\n')
                response_started = False
                response_lines = []
                
                for line in output_lines:
                    if line.startswith("Response:"):
                        response_started = True
                        continue
                    elif line.startswith("--- Based on"):
                        break
                    elif response_started:
                        response_lines.append(line)
                
                response = '\n'.join(response_lines).strip()
                print(f" {response}")
            else:
                print(f"Error: {result.stderr}")
        
        except Exception as e:
            print(f"Error running query: {e}")
        
        time.sleep(1)  # incase of delay during queries
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nTry your own queries with:")
    print('python search_engine.py search "your query here"')

if __name__ == "__main__":
    run_demo()
