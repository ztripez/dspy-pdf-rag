"""
CLI demonstration of the PDF RAG Module.
This script shows how to use the PDFRAGModule for pinout extraction from datasheets.

For the reusable module, see pdf_rag_module.py
For usage examples, see examples/
"""

import sys
import os
import dspy
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from typing import Optional

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Configure ModernBERT embeddings
embed_model = HuggingFaceEmbedding(
    model_name="Alibaba-NLP/gte-large-en-v1.5",  # Using GTE-large as ModernBERT Embed isn't on HF yet
    embed_batch_size=4,  # Reduced batch size for very large documents on 4GB VRAM
    trust_remote_code=True,
    device="cuda"  # Use GPU for faster embeddings
)

# Configure LLM provider - supports any LiteLLM provider
# Examples:
# - OpenAI: "openai/gpt-4-turbo", "openai/gpt-4o-mini"
# - Anthropic: "anthropic/claude-3-opus-20240229", "anthropic/claude-3-sonnet-20240229"
# - Google: "gemini/gemini-pro", "gemini/gemini-1.5-pro"
# - Local: "ollama/llama2", "ollama/mistral"
# - Azure: "azure/<deployment-name>"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4-turbo")
OPTIMIZE_MODEL = os.getenv("OPTIMIZE_MODEL", "openai/gpt-4o-mini")

# Configure DSPy with LiteLLM
dspy.settings.configure(lm=dspy.LM(model=DEFAULT_MODEL))

class PinoutSignature(dspy.Signature):
    """Extract pinout information from technical documentation."""
    
    context = dspy.InputField(desc="relevant sections from the datasheet")
    query = dspy.InputField(desc="user query about pinout")
    
    pin_number = dspy.OutputField(desc="pin number or range")
    pin_name = dspy.OutputField(desc="pin name/label")
    pin_function = dspy.OutputField(desc="pin function description")
    electrical_characteristics = dspy.OutputField(desc="voltage, current, or other electrical specs")

class PinoutExtractor(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.extract = dspy.ChainOfThought(PinoutSignature)
    
    def forward(self, query):
        # Retrieve relevant chunks
        results = self.retriever.retrieve(query, k=5)
        context = "\n\n".join([r.text for r in results])
        
        # Extract pinout information
        prediction = self.extract(context=context, query=query)
        return prediction

class LlamaIndexRetriever:
    def __init__(self, index):
        self.index = index
        self.retriever = index.as_retriever(similarity_top_k=5)
    
    def retrieve(self, query: str, k: Optional[int] = 5):
        retriever = self.index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(query)
        return results

def load_and_index_pdf(pdf_path: str):
    """Load PDF and create vector index with ChromaDB, reusing existing embeddings if available."""
    
    # Create collection name based on PDF filename
    import os
    pdf_name = os.path.basename(pdf_path).replace('.pdf', '').replace(' ', '_')
    collection_name = f"datasheet_{pdf_name}"
    
    # Check if collection already exists and has data
    try:
        existing_collection = chroma_client.get_collection(collection_name)
        if existing_collection.count() > 0:
            print(f"Reusing existing embeddings for {pdf_name} ({existing_collection.count()} chunks)")
            # Create vector store from existing collection
            vector_store = ChromaVectorStore(chroma_collection=existing_collection)
            # Create index from existing vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model
            )
            return index
    except:
        print(f"No existing embeddings found for {pdf_name}, creating new ones...")
    
    # Create new collection if doesn't exist or is empty
    collection = chroma_client.get_or_create_collection(collection_name)
    
    # Load PDF
    documents = SimpleDirectoryReader(
        input_files=[pdf_path]
    ).load_data()
    
    print(f"Loaded {len(documents)} pages from PDF")
    
    # Adjust batch size for very large documents
    if len(documents) > 500:
        print("Large document detected. Using batch size of 2 for embedding generation.")
        embed_model.embed_batch_size = 2
    
    # Split into chunks
    text_splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200,
    )
    
    # Create vector store with storage context
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index with storage context
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        storage_context=storage_context,
        text_splitter=text_splitter,
        show_progress=True
    )
    
    # Reset batch size
    embed_model.embed_batch_size = 4
    
    print(f"Created and stored {collection.count()} embeddings for {pdf_name}")
    return index

def create_training_examples():
    """Create a few training examples for MIPRO optimization."""
    examples = [
        dspy.Example(
            query="What is the function of pin 23?",
            pin_number="23",
            pin_name="GPIO_15",
            pin_function="General Purpose Input/Output pin 15",
            electrical_characteristics="3.3V logic level, max 12mA"
        ),
        dspy.Example(
            query="Which pins are for power supply?",
            pin_number="1, 17",
            pin_name="VDD, VSS",
            pin_function="Power supply and ground pins",
            electrical_characteristics="VDD: 3.3V ±10%, VSS: Ground"
        ),
        dspy.Example(
            query="What are the I2C pins?",
            pin_number="9, 10",
            pin_name="SDA, SCL",
            pin_function="I2C data and clock lines",
            electrical_characteristics="3.3V, open-drain, requires external pull-up resistors"
        )
    ]
    return examples

def main():
    # Get PDF path from command line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf> [--no-optimize]")
        print("Example: python main.py docs/ESP-32\\ Dev\\ Kit\\ C\\ V2_EN.pdf")
        print("         python main.py docs/datasheet.pdf --no-optimize")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    skip_optimization = "--no-optimize" in sys.argv
    
    print(f"Using LLM: {DEFAULT_MODEL}")
    print(f"Loading and indexing PDF: {pdf_path}...")
    index = load_and_index_pdf(pdf_path)
    
    # Create DSPy retriever wrapper
    retriever = LlamaIndexRetriever(index)
    
    # Create pinout extractor
    extractor = PinoutExtractor(retriever)
    
    # Create training examples
    train_examples = create_training_examples()
    
    # Configure MIPRO optimizer
    optimize = not skip_optimization
    
    if optimize:
        from dspy.teleprompt import MIPROv2
        
        # Define metric for optimization
        def pinout_metric(example, prediction, trace=None):
            # Check if all fields are present and non-empty
            return all([
                prediction.pin_number,
                prediction.pin_name,
                prediction.pin_function,
                prediction.electrical_characteristics
            ])
        
        print(f"\nOptimizing with MIPRO using {OPTIMIZE_MODEL}...")
        teleprompter = MIPROv2(
            metric=pinout_metric,
            prompt_model=dspy.LM(model=OPTIMIZE_MODEL),
            task_model=dspy.LM(model=OPTIMIZE_MODEL),
            auto='light'  # Use automatic settings for light optimization
        )
        
        # Compile with MIPRO
        optimized_extractor = teleprompter.compile(
            extractor,
            trainset=train_examples,
            requires_permission_to_run=False  # Disable confirmation prompt for automation
        )
    else:
        print("\nSkipping MIPRO optimization (set optimize=True to enable)")
        optimized_extractor = extractor
    
    # Test queries
    test_queries = [
        "What is pin 42 used for?",
        "Show me the UART pins",
        "Which pins handle the SPI interface?",
        "What are the analog input pins?"
    ]
    
    print("\n" + "="*50)
    print("Testing Pinout Extraction")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = optimized_extractor(query)
        print(f"Pin Number: {result.pin_number}")
        print(f"Pin Name: {result.pin_name}")
        print(f"Function: {result.pin_function}")
        print(f"Electrical Specs: {result.electrical_characteristics}")
        print("-"*50)

if __name__ == "__main__":
    main()
