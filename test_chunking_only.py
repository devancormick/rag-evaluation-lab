#!/usr/bin/env python3
"""
Test chunking strategies without heavy dependencies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_all_chunkers():
    """Test all chunking strategies."""
    from src.chunking import (
        FixedSizeChunker,
        TextTilingChunker,
        C99Chunker,
        LayoutAwareChunker
    )
    
    test_text = "This is a test document with multiple sentences. " * 30
    test_text += "Here is a new topic that is different. " * 20
    test_text += "And another section with different content. " * 20
    
    results = {}
    
    # Test FixedSizeChunker
    try:
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(test_text)
        results['FixedSize'] = {
            'success': True,
            'num_chunks': len(chunks),
            'config': chunker.get_config()
        }
        print(f"✅ FixedSizeChunker: {len(chunks)} chunks")
    except Exception as e:
        results['FixedSize'] = {'success': False, 'error': str(e)}
        print(f"❌ FixedSizeChunker failed: {e}")
    
    # Test TextTilingChunker
    try:
        chunker = TextTilingChunker(w=10, k=5)
        chunks = chunker.chunk(test_text)
        results['TextTiling'] = {
            'success': True,
            'num_chunks': len(chunks),
            'config': chunker.get_config()
        }
        print(f"✅ TextTilingChunker: {len(chunks)} chunks")
    except Exception as e:
        results['TextTiling'] = {'success': False, 'error': str(e)}
        print(f"❌ TextTilingChunker failed: {e}")
    
    # Test C99Chunker
    try:
        chunker = C99Chunker(window_size=50, similarity_threshold=0.5)
        chunks = chunker.chunk(test_text)
        results['C99'] = {
            'success': True,
            'num_chunks': len(chunks),
            'config': chunker.get_config()
        }
        print(f"✅ C99Chunker: {len(chunks)} chunks")
    except Exception as e:
        results['C99'] = {'success': False, 'error': str(e)}
        print(f"❌ C99Chunker failed: {e}")
    
    # Test LayoutAwareChunker
    try:
        chunker = LayoutAwareChunker(max_chunk_size=300, preserve_structure=True)
        chunks = chunker.chunk(test_text)
        results['LayoutAware'] = {
            'success': True,
            'num_chunks': len(chunks),
            'config': chunker.get_config()
        }
        print(f"✅ LayoutAwareChunker: {len(chunks)} chunks")
    except Exception as e:
        results['LayoutAware'] = {'success': False, 'error': str(e)}
        print(f"❌ LayoutAwareChunker failed: {e}")
    
    return results

def test_config_system():
    """Test configuration system."""
    from src.config import load_config, ExperimentConfig
    from pathlib import Path
    
    config_file = Path(__file__).parent / "experiments" / "baseline.yaml"
    if not config_file.exists():
        print("⚠️  Config file not found")
        return False
    
    try:
        config = load_config(str(config_file))
        print(f"✅ Config loaded: {config.experiment_name}")
        print(f"   - Chunking: {config.chunking.strategy}")
        print(f"   - Embedding: {config.embedding.model_family}/{config.embedding.model_name}")
        print(f"   - VectorDB: {config.vectordb.backend}")
        print(f"   - Dataset: {config.dataset.name}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Chunking Strategies\n" + "="*50)
    chunking_results = test_all_chunkers()
    
    print("\n" + "="*50)
    print("Testing Configuration System\n" + "="*50)
    config_ok = test_config_system()
    
    print("\n" + "="*50)
    all_passed = all(r.get('success', False) for r in chunking_results.values()) and config_ok
    
    if all_passed:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
