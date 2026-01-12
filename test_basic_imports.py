#!/usr/bin/env python3
"""
Basic import test to verify code structure without heavy dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    errors = []
    
    try:
        from src.chunking import FixedSizeChunker, TextTilingChunker, C99Chunker, LayoutAwareChunker
        print("✅ Chunking modules imported successfully")
    except Exception as e:
        errors.append(f"Chunking imports failed: {e}")
    
    try:
        from src.config import load_config, ExperimentConfig
        print("✅ Config modules imported successfully")
    except Exception as e:
        errors.append(f"Config imports failed: {e}")
    
    try:
        from src.datasets import QASPERLoader, HotpotQALoader
        print("✅ Dataset modules imported successfully")
    except Exception as e:
        errors.append(f"Dataset imports failed: {e}")
    
    try:
        from src.evaluation import compute_precision_at_k, compute_recall_at_k, compute_mrr
        print("✅ Evaluation modules imported successfully")
    except Exception as e:
        errors.append(f"Evaluation imports failed: {e}")
    
    try:
        from src.pipeline import ExperimentRunner, ResultsAnalyzer
        print("✅ Pipeline modules imported successfully")
    except Exception as e:
        errors.append(f"Pipeline imports failed: {e}")
    
    try:
        from src.vectordb import FAISSVectorDB
        print("✅ VectorDB modules imported successfully (FAISS)")
    except Exception as e:
        errors.append(f"VectorDB imports failed: {e}")
    
    if errors:
        print("\n❌ Errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("\n✅ All basic imports successful!")
    return True

def test_chunking_basic():
    """Test basic chunking functionality."""
    try:
        from src.chunking import FixedSizeChunker
        
        text = "This is a test document. " * 20
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0, "Should produce at least one chunk"
        assert all("text" in chunk for chunk in chunks), "All chunks should have text"
        assert all("start_idx" in chunk for chunk in chunks), "All chunks should have start_idx"
        
        print("✅ Basic chunking test passed")
        return True
    except Exception as e:
        print(f"❌ Basic chunking test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    try:
        from src.config import load_config
        from pathlib import Path
        
        config_file = Path(__file__).parent / "experiments" / "baseline.yaml"
        if config_file.exists():
            config = load_config(str(config_file))
            assert config.experiment_name is not None
            print("✅ Config loading test passed")
            return True
        else:
            print("⚠️  Config file not found, skipping config test")
            return True
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running basic import and structure tests...\n")
    
    all_passed = True
    all_passed &= test_imports()
    print()
    all_passed &= test_chunking_basic()
    print()
    all_passed &= test_config_loading()
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All basic tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
