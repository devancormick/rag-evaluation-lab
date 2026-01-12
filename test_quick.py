#!/usr/bin/env python3
"""
Quick structure and import tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Testing basic structure...")

# Test 1: Config system
try:
    from src.config import load_config
    config_file = Path(__file__).parent / "experiments" / "baseline.yaml"
    if config_file.exists():
        config = load_config(str(config_file))
        print(f"✅ Config system works: {config.experiment_name}")
    else:
        print("⚠️  Config file not found")
except Exception as e:
    print(f"❌ Config test failed: {e}")

# Test 2: Fixed chunker (simplest)
try:
    from src.chunking import FixedSizeChunker
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    text = "A" * 100
    chunks = chunker.chunk(text)
    print(f"✅ FixedSizeChunker works: {len(chunks)} chunks")
except Exception as e:
    print(f"❌ FixedSizeChunker failed: {e}")

# Test 3: Results analyzer structure
try:
    from src.pipeline.results_analyzer import ResultsAnalyzer
    analyzer = ResultsAnalyzer()
    print("✅ ResultsAnalyzer imports successfully")
except Exception as e:
    print(f"❌ ResultsAnalyzer import failed: {e}")

print("\n✅ Quick tests completed!")
