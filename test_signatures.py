#!/usr/bin/env python3
"""Test all signature types with the PDF RAG module."""

import dspy
from pdf_rag_module import PDFRAGModule
from signatures import (
    PinoutExtractionSignature,
    SpecificationExtractionSignature,
    TableExtractionSignature,
    SummarySignature,
    ComparisonSignature,
    TroubleshootingSignature,
    CodeExtractionSignature
)
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()

# Configure DSPy
dspy.settings.configure(lm=dspy.LM(model="openai/gpt-4o-mini"))

# Create a shared embedding model to avoid CUDA OOM
shared_embed_model = HuggingFaceEmbedding(
    model_name="Alibaba-NLP/gte-large-en-v1.5",
    embed_batch_size=4,
    trust_remote_code=True,
    device="cuda"
)

# Test PDF - using a smaller one for quick testing
test_pdf = "docs/ESP-32 Dev Kit C V2_EN.pdf"

print("Testing all signatures with ESP32 datasheet...\n")

# 1. Test PDFQASignature (default)
print("1. Testing PDFQASignature (default):")
qa_module = PDFRAGModule(test_pdf, embed_model=shared_embed_model)
result = qa_module("What is the operating voltage?")
print(f"Answer: {result.answer}\n")

# 2. Test PinoutExtractionSignature
print("2. Testing PinoutExtractionSignature:")
pinout_module = PDFRAGModule(test_pdf, signature=PinoutExtractionSignature, embed_model=shared_embed_model)
result = pinout_module("What are the I2C pins?")
print(f"Pin Number: {result.pin_number}")
print(f"Pin Name: {result.pin_name}")
print(f"Function: {result.pin_function}")
print(f"Electrical: {result.electrical_characteristics}\n")

# 3. Test SpecificationExtractionSignature
print("3. Testing SpecificationExtractionSignature:")
spec_module = PDFRAGModule(test_pdf, signature=SpecificationExtractionSignature, embed_model=shared_embed_model)
result = spec_module("What is the flash memory size?")
print(f"Parameter: {result.parameter}")
print(f"Value: {result.value}")
print(f"Conditions: {result.conditions}")
print(f"Range: {result.min_max_range}\n")

# 4. Test TableExtractionSignature
print("4. Testing TableExtractionSignature:")
table_module = PDFRAGModule(test_pdf, signature=TableExtractionSignature, embed_model=shared_embed_model)
result = table_module("Show me the pin mapping table")
print(f"Table Title: {result.table_title}")
print(f"Headers: {result.headers}")
print(f"Rows (first 2): {str(result.rows)[:100]}...")  # First 100 chars
print(f"Notes: {result.notes}\n")

# 5. Test SummarySignature
print("5. Testing SummarySignature:")
summary_module = PDFRAGModule(test_pdf, signature=SummarySignature, embed_model=shared_embed_model)
result = summary_module("Summarize the key features")
print(f"Summary: {result.summary}")
print(f"Key Points: {result.key_points}")
print(f"Technical Terms: {result.technical_terms}\n")

# 6. Test ComparisonSignature
print("6. Testing ComparisonSignature:")
compare_module = PDFRAGModule(test_pdf, signature=ComparisonSignature, embed_model=shared_embed_model)
result = compare_module("Compare GPIO and ADC capabilities")
print(f"Items Compared: {result.items_compared}")
print(f"Similarities: {result.similarities}")
print(f"Differences: {result.differences}")
print(f"Recommendation: {result.recommendation}\n")

# 7. Test TroubleshootingSignature
print("7. Testing TroubleshootingSignature:")
trouble_module = PDFRAGModule(test_pdf, signature=TroubleshootingSignature, embed_model=shared_embed_model)
result = trouble_module("Device not booting")
print(f"Problem: {result.problem_description}")
print(f"Possible Causes: {result.possible_causes}")
print(f"Solutions: {result.solutions}")
print(f"Preventive Measures: {result.preventive_measures}\n")

# 8. Test CodeExtractionSignature
print("8. Testing CodeExtractionSignature:")
code_module = PDFRAGModule(test_pdf, signature=CodeExtractionSignature, embed_model=shared_embed_model)
result = code_module("Show initialization code")
print(f"Code Snippet: {result.code_snippet}")
print(f"Language: {result.language}")
print(f"Description: {result.description}")
print(f"Dependencies: {result.dependencies}\n")

print("All signatures tested successfully!")