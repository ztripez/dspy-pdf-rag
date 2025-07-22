# DSPy PDF RAG Module

A reusable DSPy module for PDF retrieval and question answering, with special support for electronics datasheets. This module can be integrated into any DSPy pipeline for document analysis, information extraction, and knowledge retrieval.

## Features

- **Reusable DSPy Module**: Integrate PDF RAG into any DSPy pipeline or chain
- **Flexible Signatures**: Use built-in or custom signatures for different extraction tasks
- **Intelligent PDF Processing**: Handles large technical PDFs (1000+ pages) with tables, figures, and complex layouts
- **Persistent Embeddings**: ChromaDB storage for instant retrieval on subsequent runs
- **Automatic Optimization**: MIPRO optimizer support for improved accuracy
- **GPU Acceleration**: Uses CUDA for fast embedding generation with adaptive batch sizing
- **Multi-Provider Support**: Works with any LLM supported by LiteLLM

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (4GB+ VRAM recommended) or CPU fallback
- API key for your chosen LLM provider (OpenAI, Anthropic, Google, etc.)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dspy-pdf.git
cd dspy-pdf
```

2. Install dependencies using uv:
```bash
pip install uv
uv sync
```

3. Create a `.env` file with your API keys:
```bash
# For OpenAI (default)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# For other providers, add the appropriate key:
# ANTHROPIC_API_KEY=your-anthropic-key
# GEMINI_API_KEY=your-google-key
# AZURE_API_KEY=your-azure-key
# GROQ_API_KEY=your-groq-key
```

## Usage

### As a Reusable Module

```python
from pdf_rag_module import PDFRAGModule
import dspy

# Configure your LLM
dspy.settings.configure(lm=dspy.LM(model="openai/gpt-4-turbo"))

# Create a PDF module
pdf_module = PDFRAGModule("datasheet.pdf")

# Ask questions
answer = pdf_module("What are the power requirements?")
print(answer.answer)
```

### With Custom Signatures

```python
from pdf_rag_module import PDFRAGModule
from signatures import PinoutExtractionSignature

# Use specialized extraction
pdf_module = PDFRAGModule(
    "datasheet.pdf",
    signature=PinoutExtractionSignature
)

result = pdf_module("What are the I2C pins?")
print(f"Pins: {result.pin_number}")
print(f"Function: {result.pin_function}")
```

### In DSPy Chains

```python
class TechnicalAnalysis(dspy.Module):
    def __init__(self, pdf_path):
        super().__init__()
        self.pdf_rag = PDFRAGModule(pdf_path)
        self.analyze = dspy.ChainOfThought("details -> analysis")
    
    def forward(self, component):
        details = self.pdf_rag(f"Describe the {component}")
        analysis = self.analyze(details=details.answer)
        return analysis
```

### CLI Demo

For a command-line demonstration:
```bash
uv run python main.py docs/your-datasheet.pdf
```

## How It Works

### 1. Document Processing
- PDFs are loaded and split into chunks using LlamaIndex
- Adaptive chunk sizing (1024 tokens with 200 token overlap)
- Handles tables and complex layouts

### 2. Embedding Generation
- Uses GTE-large embeddings via HuggingFace
- GPU acceleration with automatic batch size adjustment
- Embeddings stored in local ChromaDB for reuse

### 3. Retrieval & Extraction
- DSPy retrieval module finds relevant chunks
- Chain-of-thought reasoning extracts structured pinout data
- Returns pin number, name, function, and electrical specs

### 4. Optimization (Optional)
- MIPRO optimizer automatically tunes prompts
- Uses few-shot examples to improve accuracy
- Learns from your specific datasheet formats

## Embeddings Caching

The system automatically caches embeddings in ChromaDB:
- First run: Generates and stores embeddings (slower)
- Subsequent runs: Instantly loads from cache (fast)
- Each PDF gets its own collection named `datasheet_{filename}`

To clear the cache:
```bash
rm -rf chroma_db/
```

## Model Memory Management

The module automatically manages embedding models to prevent CUDA out-of-memory errors:

### Automatic Model Caching
When creating multiple `PDFRAGModule` instances, the embedding model is automatically cached and reused:

```python
# First module creates and caches the embedding model
pdf1 = PDFRAGModule("doc1.pdf")

# Second module reuses the cached model - no additional GPU memory used!
pdf2 = PDFRAGModule("doc2.pdf")

# Third module also reuses the same model
pdf3 = PDFRAGModule("doc3.pdf")
```

### Cache Management
For advanced users who need to manage GPU memory:

```python
from pdf_rag_module import clear_embed_cache, get_embed_cache_info

# Check what models are cached
info = get_embed_cache_info()
print(f"Cached models: {info['num_models']}")

# Clear the cache to free GPU memory
clear_embed_cache()
```

### Custom Embedding Models
You can still provide your own embedding model if needed:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Create a custom model
custom_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
    device="cuda"
)

# Use it in the module
pdf_module = PDFRAGModule("doc.pdf", embed_model=custom_model)
```

## Configuration

### Embedding Model
Default: `Alibaba-NLP/gte-large-en-v1.5`
- 1024-dimensional embeddings
- Excellent for technical content
- GPU batch size auto-adjusts based on document size

### LLM Model
The system supports any LLM provider through LiteLLM. Configure via environment variables:

```bash
# Default model for extraction
export LLM_MODEL="openai/gpt-4-turbo"

# Model for MIPRO optimization
export OPTIMIZE_MODEL="openai/gpt-4o-mini"
```

#### Supported Providers
- **OpenAI**: `openai/gpt-4-turbo`, `openai/gpt-4o-mini`
- **Anthropic**: `anthropic/claude-3-opus-20240229`, `anthropic/claude-3-sonnet-20240229`
- **Google**: `gemini/gemini-pro`, `gemini/gemini-1.5-pro`
- **Azure**: `azure/<your-deployment-name>`
- **Groq**: `groq/llama3-70b-8192`, `groq/mixtral-8x7b-32768`
- **Local**: `ollama/llama2`, `ollama/mistral`, `ollama/codellama`
- **And many more!** See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for full list

## Performance Tips

1. **GPU Memory**: For large PDFs (>500 pages), the system automatically reduces batch size
2. **Optimization**: First run with optimization takes longer but improves accuracy
3. **Caching**: Keep `chroma_db/` directory for fast subsequent runs

## Troubleshooting

### CUDA Out of Memory
- System automatically adjusts batch size for large documents
- If issues persist, edit `embed_batch_size` in `main.py`

### No Pinout Found
- Check if the PDF contains pinout tables/diagrams
- Some datasheets use different terminology (e.g., "terminal" instead of "pin")
- Try more specific queries

## Module Structure

### Core Module
- `pdf_rag_module.py` - The main `PDFRAGModule` class
- `signatures.py` - Pre-built signatures for common extraction tasks

### Signatures Available
- `PDFQASignature` - General question answering
- `PinoutExtractionSignature` - Extract pin specifications
- `SpecificationExtractionSignature` - Extract technical specs
- `TableExtractionSignature` - Extract structured table data
- `SummarySignature` - Generate summaries
- `ComparisonSignature` - Compare items in PDFs
- `TroubleshootingSignature` - Extract troubleshooting info
- `CodeExtractionSignature` - Extract code examples

### Examples
- `examples/basic_usage.py` - Simple usage patterns
- `examples/advanced_integration.py` - Complex DSPy chains and multi-PDF analysis

## Real-World Examples

Tested with various electronics datasheets:
- ESP32 (30 pages) - Fast processing, accurate results
- OMAP-L138 (291 pages) - Medium processing time
- SH7780 (1340 pages) - Handles complex pinouts with 1343 chunks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details