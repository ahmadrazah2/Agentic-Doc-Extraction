"""
Markdown Loader for RAG Pipeline

This module loads markdown files with anchor IDs and converts them into
structured chunks suitable for vector database storage.

The markdown format has chunks separated by anchor tags:
<a id='chunk-id'></a>
Content here...
"""

import re
from typing import List, Dict, Any
from pathlib import Path
from langchain_core.documents import Document


class MarkdownChunkLoader:
    """
    Load markdown files with anchor-based chunks.
    
    Parses markdown files where chunks are separated by anchor tags
    like: <a id='chunk-id'></a>
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the loader.
        
        Args:
            file_path: Path to the markdown file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def load(self) -> List[Document]:
        """
        Load and parse the markdown file into Document objects.
        
        Returns:
            List of Document objects, one per chunk
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by anchor tags
        chunks = self._split_by_anchors(content)
        
        # Convert to Document objects
        documents = []
        for chunk_id, text in chunks:
            if text.strip():  # Skip empty chunks
                doc = Document(
                    page_content=text.strip(),
                    metadata={
                        'chunk_id': chunk_id,
                        'source': str(self.file_path.name),
                        'file_path': str(self.file_path)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _split_by_anchors(self, content: str) -> List[tuple]:
        """
        Split content by anchor tags.
        
        Args:
            content: Full markdown content
            
        Returns:
            List of (chunk_id, text) tuples
        """
        # Pattern to match anchor tags: <a id='...'></a>
        anchor_pattern = r"<a id='([^']+)'></a>"
        
        # Find all anchor positions
        chunks = []
        matches = list(re.finditer(anchor_pattern, content))
        
        for i, match in enumerate(matches):
            chunk_id = match.group(1)
            start_pos = match.end()
            
            # End position is the start of next anchor, or end of file
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)
            
            # Extract text between anchors
            text = content[start_pos:end_pos].strip()
            chunks.append((chunk_id, text))
        
        return chunks
    
    def load_with_metadata(self, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Load documents with additional custom metadata.
        
        Args:
            metadata: Additional metadata to add to all documents
            
        Returns:
            List of Document objects with merged metadata
        """
        documents = self.load()
        
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded markdown file.
        
        Returns:
            Dictionary with statistics
        """
        documents = self.load()
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        return {
            'num_chunks': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_chunk_length': total_chars / len(documents) if documents else 0,
            'avg_words_per_chunk': total_words / len(documents) if documents else 0,
            'file_path': str(self.file_path),
            'file_size_bytes': self.file_path.stat().st_size
        }


def load_markdown_chunks(file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
    """
    Convenience function to load markdown chunks.
    
    Args:
        file_path: Path to markdown file
        metadata: Optional additional metadata
        
    Returns:
        List of Document objects
    """
    loader = MarkdownChunkLoader(file_path)
    if metadata:
        return loader.load_with_metadata(metadata)
    return loader.load()


def print_markdown_stats(file_path: str):
    """
    Print statistics about a markdown file.
    
    Args:
        file_path: Path to markdown file
    """
    loader = MarkdownChunkLoader(file_path)
    stats = loader.get_stats()
    
    print(f"\n📄 Markdown File Statistics")
    print(f"{'='*60}")
    print(f"File: {stats['file_path']}")
    print(f"File Size: {stats['file_size_bytes']:,} bytes")
    print(f"\n📊 Content:")
    print(f"  Chunks: {stats['num_chunks']}")
    print(f"  Total Characters: {stats['total_characters']:,}")
    print(f"  Total Words: {stats['total_words']:,}")
    print(f"\n📏 Averages:")
    print(f"  Avg Chunk Length: {stats['avg_chunk_length']:.1f} characters")
    print(f"  Avg Words/Chunk: {stats['avg_words_per_chunk']:.1f} words")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "ade_outputs/desnow.md"
    
    print(f"Loading markdown file: {file_path}\n")
    
    try:
        # Load documents
        loader = MarkdownChunkLoader(file_path)
        documents = loader.load()
        
        # Print stats
        print_markdown_stats(file_path)
        
        # Show first few chunks
        print("📝 First 3 Chunks Preview:\n")
        for i, doc in enumerate(documents[:3], 1):
            print(f"Chunk {i}:")
            print(f"  ID: {doc.metadata['chunk_id']}")
            print(f"  Length: {len(doc.page_content)} chars")
            print(f"  Preview: {doc.page_content[:100]}...")
            print()
        
        print(f"✅ Successfully loaded {len(documents)} chunks!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
