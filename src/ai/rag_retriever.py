"""
RAG (Retrieval-Augmented Generation) System
Uses Amazon Bedrock Titan embeddings to retrieve relevant documentation chunks
"""

import json
import numpy as np
from typing import List, Tuple
import boto3
from pathlib import Path


class RAGRetriever:
    """Lightweight RAG system using Bedrock Titan embeddings"""
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize RAG retriever
        
        Args:
            region: AWS region for Bedrock
        """
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        self.embedding_model = 'amazon.titan-embed-text-v2:0'
        
        # Document store: [(chunk_text, embedding_vector)]
        self.documents: List[Tuple[str, np.ndarray]] = []
        self.indexed = False
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Create embedding for text using Bedrock Titan
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": 256,
                "normalize": True
            })
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.embedding_model,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            embedding = np.array(response_body['embedding'])
            return embedding
            
        except Exception as e:
            print(f"Error creating embedding: {e}")
            # Return zero vector on error
            return np.zeros(256)
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            
            if current_size + line_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                
                # Keep last few lines for overlap
                overlap_lines = []
                overlap_size = 0
                for prev_line in reversed(current_chunk):
                    if overlap_size < overlap:
                        overlap_lines.insert(0, prev_line)
                        overlap_size += len(prev_line)
                    else:
                        break
                
                current_chunk = overlap_lines
                current_size = overlap_size
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def index_readme(self, readme_path: str = 'README.md'):
        """
        Index README.md for RAG retrieval
        
        Args:
            readme_path: Path to README file
        """
        print("🔍 Indexing README for RAG...")
        
        # Read README
        readme_file = Path(readme_path)
        if not readme_file.exists():
            print(f"Warning: {readme_path} not found, RAG will be limited")
            return
        
        readme_text = readme_file.read_text(encoding='utf-8')
        
        # Create chunks
        chunks = self._chunk_text(readme_text, chunk_size=600, overlap=150)
        print(f"  Created {len(chunks)} chunks from README")
        
        # Embed each chunk
        for i, chunk in enumerate(chunks):
            # Clean chunk (remove markdown code blocks, excessive whitespace)
            clean_chunk = chunk.replace('```', '').strip()
            if len(clean_chunk) < 50:  # Skip very short chunks
                continue
            
            embedding = self._embed_text(clean_chunk)
            self.documents.append((clean_chunk, embedding))
            
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(chunks)} chunks...")
        
        self.indexed = True
        print(f"✅ Indexed {len(self.documents)} document chunks")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant document chunks for query
        
        Args:
            query: User's question
            top_k: Number of top chunks to return
            
        Returns:
            List of relevant text chunks
        """
        if not self.indexed or not self.documents:
            return []
        
        # Embed query
        query_embedding = self._embed_text(query)
        
        # Calculate cosine similarity with all documents
        similarities = []
        for chunk, doc_embedding in self.documents:
            # Cosine similarity (embeddings are normalized)
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((chunk, similarity))
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, score in similarities[:top_k]]
        
        return top_chunks
    
    def get_context(self, query: str) -> str:
        """
        Get formatted context string for query
        
        Args:
            query: User's question
            
        Returns:
            Formatted context string
        """
        relevant_chunks = self.retrieve(query, top_k=3)
        
        if not relevant_chunks:
            return ""
        
        context = "RELEVANT DOCUMENTATION:\n\n"
        for i, chunk in enumerate(relevant_chunks, 1):
            context += f"[Source {i}]\n{chunk}\n\n"
        
        return context.strip()
