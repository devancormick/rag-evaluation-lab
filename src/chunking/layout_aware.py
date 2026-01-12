"""
Layout-aware chunking for structured documents.
"""

import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from .base import ChunkingStrategy


class LayoutAwareChunker(ChunkingStrategy):
    """Chunk text using layout-aware approach for structured documents."""

    def __init__(
        self,
        max_chunk_size: int = 512,
        preserve_structure: bool = True,
        respect_paragraphs: bool = True
    ):
        """
        Initialize layout-aware chunker.

        Args:
            max_chunk_size: Maximum characters per chunk
            preserve_structure: Whether to preserve document structure (headers, lists, etc.)
            respect_paragraphs: Whether to respect paragraph boundaries
        """
        self.max_chunk_size = max_chunk_size
        self.preserve_structure = preserve_structure
        self.respect_paragraphs = respect_paragraphs

    def _extract_structure(self, text: str) -> List[Dict[str, Any]]:
        """Extract structural elements from text."""
        # Try to parse as HTML if possible
        try:
            soup = BeautifulSoup(text, 'html.parser')
            elements = []
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div']):
                elements.append({
                    'type': element.name,
                    'text': element.get_text().strip(),
                    'level': int(element.name[1]) if element.name.startswith('h') else 0
                })
            return elements
        except:
            pass

        # Fallback: parse as plain text with paragraph detection
        elements = []
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Detect headers (lines that are short and end without punctuation)
            lines = para.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Simple heuristic: headers are short lines without sentence-ending punctuation
                if len(line) < 100 and not re.search(r'[.!?]$', line):
                    elements.append({
                        'type': 'header',
                        'text': line,
                        'level': 1
                    })
                else:
                    elements.append({
                        'type': 'paragraph',
                        'text': line,
                        'level': 0
                    })

        return elements

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text using layout-aware approach."""
        if metadata is None:
            metadata = {}

        if self.preserve_structure:
            elements = self._extract_structure(text)
        else:
            # Fallback to paragraph-based chunking
            paragraphs = re.split(r'\n\s*\n', text)
            elements = [{'type': 'paragraph', 'text': p.strip(), 'level': 0} 
                       for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        start_idx = 0

        for element in elements:
            element_text = element['text']
            element_size = len(element_text)

            # If adding this element would exceed max size, finalize current chunk
            if current_size + element_size > self.max_chunk_size and current_chunk:
                chunk_text = '\n\n'.join([e['text'] for e in current_chunk])
                end_idx = start_idx + len(chunk_text)
                
                chunks.append({
                    "text": chunk_text,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "metadata": {
                        **metadata,
                        "chunk_index": len(chunks),
                        "structure_types": [e['type'] for e in current_chunk]
                    }
                })

                start_idx = end_idx
                current_chunk = []
                current_size = 0

            # Add element to current chunk
            current_chunk.append(element)
            current_size += element_size + 2  # +2 for '\n\n'

        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join([e['text'] for e in current_chunk])
            chunks.append({
                "text": chunk_text,
                "start_idx": start_idx,
                "end_idx": start_idx + len(chunk_text),
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "structure_types": [e['type'] for e in current_chunk]
                }
            })

        return chunks if chunks else [{
            "text": text,
            "start_idx": 0,
            "end_idx": len(text),
            "metadata": {**metadata, "chunk_index": 0}
        }]

    def get_config(self) -> Dict[str, Any]:
        """Return chunker configuration."""
        return {
            "strategy": "layout_aware",
            "max_chunk_size": self.max_chunk_size,
            "preserve_structure": self.preserve_structure,
            "respect_paragraphs": self.respect_paragraphs
        }

