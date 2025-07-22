"""
PDF RAG Module for DSPy
A reusable DSPy module for PDF retrieval and question answering.
"""

import os
import dspy
from typing import Optional, Union, Type
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


class PDFQASignature(dspy.Signature):
    """Default signature for PDF question answering."""
    context = dspy.InputField(desc="relevant sections from the PDF")
    query = dspy.InputField(desc="user question about the PDF content")
    answer = dspy.OutputField(desc="comprehensive answer based on the PDF content")


class PDFRAGModule(dspy.Module):
    """
    A reusable DSPy module for PDF retrieval and question answering.
    
    This module handles PDF loading, embedding generation, caching, retrieval,
    and structured extraction using DSPy signatures.
    
    Args:
        pdf_path: Path to the PDF file
        signature: Optional DSPy signature for custom extraction (defaults to PDFQASignature)
        embed_model: Optional embedding model (defaults to GTE-large)
        collection_name: Optional ChromaDB collection name (auto-generated from PDF name if None)
        chunk_size: Size of text chunks (default: 1024)
        chunk_overlap: Overlap between chunks (default: 200)
        retrieve_k: Number of chunks to retrieve (default: 5)
        chroma_path: Path to ChromaDB storage (default: "./chroma_db")
    """
    
    def __init__(
        self,
        pdf_path: str,
        signature: Optional[Type[dspy.Signature]] = None,
        embed_model: Optional[HuggingFaceEmbedding] = None,
        collection_name: Optional[str] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        retrieve_k: int = 5,
        chroma_path: str = "./chroma_db",
    ):
        super().__init__()
        
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieve_k = retrieve_k
        
        # Use provided signature or default to PDFQASignature
        self.signature = signature or PDFQASignature
        
        # Initialize embedding model if not provided
        if embed_model is None:
            self.embed_model = HuggingFaceEmbedding(
                model_name="Alibaba-NLP/gte-large-en-v1.5",
                embed_batch_size=4,
                trust_remote_code=True,
                device="cuda" if __import__('torch').cuda.is_available() else "cpu"
            )
        else:
            self.embed_model = embed_model
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Generate collection name from PDF if not provided
        if collection_name is None:
            pdf_name = os.path.basename(pdf_path).replace('.pdf', '').replace(' ', '_')
            self.collection_name = f"pdf_{pdf_name}"
        else:
            self.collection_name = collection_name
        
        # Load and index the PDF
        self.index = self._load_and_index_pdf()
        
        # Create the extraction module with the signature
        self.extract = dspy.ChainOfThought(self.signature)
    
    def _load_and_index_pdf(self):
        """Load PDF and create vector index with ChromaDB, reusing existing embeddings if available."""
        
        # Check if collection already exists and has data
        try:
            existing_collection = self.chroma_client.get_collection(self.collection_name)
            if existing_collection.count() > 0:
                print(f"Reusing existing embeddings for {self.collection_name} ({existing_collection.count()} chunks)")
                # Create vector store from existing collection
                vector_store = ChromaVectorStore(chroma_collection=existing_collection)
                # Create index from existing vector store
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=self.embed_model
                )
                return index
        except:
            print(f"No existing embeddings found for {self.collection_name}, creating new ones...")
        
        # Create new collection
        collection = self.chroma_client.get_or_create_collection(self.collection_name)
        
        # Load PDF
        print(f"Loading PDF: {self.pdf_path}")
        documents = SimpleDirectoryReader(
            input_files=[self.pdf_path]
        ).load_data()
        
        print(f"Loaded {len(documents)} pages from PDF")
        
        # Adjust batch size for very large documents
        if len(documents) > 500 and hasattr(self.embed_model, 'embed_batch_size'):
            print("Large document detected. Using batch size of 2 for embedding generation.")
            original_batch_size = self.embed_model.embed_batch_size
            self.embed_model.embed_batch_size = 2
        else:
            original_batch_size = None
        
        # Split into chunks
        text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        # Create vector store with storage context
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index with storage context
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.embed_model,
            storage_context=storage_context,
            text_splitter=text_splitter,
            show_progress=True
        )
        
        # Reset batch size if it was changed
        if original_batch_size is not None:
            self.embed_model.embed_batch_size = original_batch_size
        
        print(f"Created and stored {collection.count()} embeddings for {self.collection_name}")
        return index
    
    def retrieve(self, query: str, k: Optional[int] = None) -> list:
        """Retrieve relevant chunks for a query."""
        k = k or self.retrieve_k
        retriever = self.index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(query)
        return results
    
    def forward(self, query: str, **kwargs) -> dspy.Prediction:
        """
        Process a query against the PDF.
        
        Args:
            query: The question or query about the PDF content
            **kwargs: Additional arguments to pass to the signature
        
        Returns:
            dspy.Prediction with the extraction results
        """
        # Retrieve relevant chunks
        results = self.retrieve(query, k=kwargs.get('retrieve_k', self.retrieve_k))
        context = "\n\n".join([r.text for r in results])
        
        # Prepare inputs for the signature
        signature_inputs = {
            'query': query,
            'context': context,
            **kwargs  # Allow passing additional fields
        }
        
        # Extract information using the signature
        prediction = self.extract(**signature_inputs)
        
        # Add retrieval results to the prediction for transparency
        prediction.retrieved_chunks = [r.text for r in results]
        prediction.num_chunks = len(results)
        
        return prediction
    
    def __call__(self, query: str, **kwargs) -> dspy.Prediction:
        """Allow calling the module directly."""
        return self.forward(query, **kwargs)