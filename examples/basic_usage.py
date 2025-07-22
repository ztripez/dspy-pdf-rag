"""
Basic usage examples for the PDF RAG Module.
"""

import dspy
from pdf_rag_module import PDFRAGModule
from signatures import (
    PinoutExtractionSignature, 
    SpecificationExtractionSignature,
    SummarySignature
)

# Configure DSPy with your preferred LLM
dspy.settings.configure(lm=dspy.LM(model="openai/gpt-4-turbo"))


def example_simple_qa():
    """Example 1: Simple question answering with default signature."""
    print("=== Example 1: Simple Q&A ===")
    
    # Initialize module with a PDF
    pdf_module = PDFRAGModule("docs/ESP-32 Dev Kit C V2_EN.pdf")
    
    # Ask questions
    result = pdf_module("What are the power supply requirements?")
    print(f"Question: What are the power supply requirements?")
    print(f"Answer: {result.answer}")
    print(f"Based on {result.num_chunks} retrieved chunks\n")


def example_pinout_extraction():
    """Example 2: Extract pinout information using custom signature."""
    print("=== Example 2: Pinout Extraction ===")
    
    # Initialize with pinout extraction signature
    pdf_module = PDFRAGModule(
        "docs/esp32-wroom-32e_esp32-wroom-32ue_datasheet_en.pdf",
        signature=PinoutExtractionSignature
    )
    
    # Extract pinout information
    result = pdf_module("What are the I2C pins?")
    print(f"Query: What are the I2C pins?")
    print(f"Pin Numbers: {result.pin_number}")
    print(f"Pin Names: {result.pin_name}")
    print(f"Function: {result.pin_function}")
    print(f"Electrical Specs: {result.electrical_characteristics}\n")


def example_specification_extraction():
    """Example 3: Extract technical specifications."""
    print("=== Example 3: Specification Extraction ===")
    
    pdf_module = PDFRAGModule(
        "docs/omap-l138.pdf",
        signature=SpecificationExtractionSignature,
        retrieve_k=3  # Retrieve fewer chunks for specific queries
    )
    
    result = pdf_module("What is the maximum operating frequency?")
    print(f"Parameter: {result.parameter}")
    print(f"Value: {result.value}")
    print(f"Conditions: {result.conditions}")
    print(f"Range: {result.min_max_range}\n")


def example_in_dspy_chain():
    """Example 4: Using PDF RAG Module in a DSPy chain."""
    print("=== Example 4: DSPy Chain Usage ===")
    
    class TechnicalReportGenerator(dspy.Module):
        """Generate technical reports by combining PDF analysis with summaries."""
        
        def __init__(self, pdf_path):
            super().__init__()
            # PDF module for detailed extraction
            self.pdf_qa = PDFRAGModule(pdf_path)
            # PDF module for summaries
            self.pdf_summary = PDFRAGModule(pdf_path, signature=SummarySignature)
            # Additional processing
            self.format_report = dspy.ChainOfThought("details, summary -> report")
        
        def forward(self, topic):
            # Get detailed information
            details = self.pdf_qa(f"Explain the {topic} in detail")
            
            # Get summary
            summary = self.pdf_summary(f"Summarize the {topic} section")
            
            # Generate report
            report = self.format_report(
                details=details.answer,
                summary=summary.summary
            )
            
            return report
    
    # Use the chain
    report_generator = TechnicalReportGenerator("docs/sh7780_series.pdf")
    report = report_generator("memory architecture")
    print(f"Technical Report on Memory Architecture:")
    print(f"{report.report}\n")


def example_multiple_pdfs():
    """Example 5: Working with multiple PDFs."""
    print("=== Example 5: Multiple PDFs ===")
    
    class MultiDatasheetAnalyzer(dspy.Module):
        """Analyze and compare information from multiple datasheets."""
        
        def __init__(self, pdf_paths):
            super().__init__()
            # Create a PDF module for each datasheet
            self.pdf_modules = {
                name: PDFRAGModule(path, collection_name=f"datasheet_{i}")
                for i, (name, path) in enumerate(pdf_paths.items())
            }
            self.synthesize = dspy.ChainOfThought("answers -> synthesis")
        
        def forward(self, query):
            # Query each PDF
            answers = {}
            for name, module in self.pdf_modules.items():
                result = module(query)
                answers[name] = result.answer
            
            # Synthesize answers
            synthesis = self.synthesize(
                answers=str(answers)
            )
            
            return synthesis
    
    # Analyze multiple datasheets
    analyzer = MultiDatasheetAnalyzer({
        "ESP32": "docs/esp32-wroom-32e_esp32-wroom-32ue_datasheet_en.pdf",
        "OMAP": "docs/omap-l138.pdf"
    })
    
    result = analyzer("What communication interfaces are supported?")
    print(f"Synthesis across datasheets:")
    print(f"{result.synthesis}\n")


def example_with_optimization():
    """Example 6: Using MIPRO optimization with the module."""
    print("=== Example 6: With MIPRO Optimization ===")
    
    from dspy.teleprompt import MIPROv2
    
    # Create module
    pdf_module = PDFRAGModule(
        "docs/ESP-32 Dev Kit C V2_EN.pdf",
        signature=PinoutExtractionSignature
    )
    
    # Create training examples
    trainset = [
        dspy.Example(
            query="What is pin 23 used for?",
            pin_number="23",
            pin_name="GPIO23",
            pin_function="General Purpose I/O",
            electrical_characteristics="3.3V"
        )
    ]
    
    # Define metric
    def metric(example, pred, trace=None):
        return all([
            pred.pin_number,
            pred.pin_name,
            pred.pin_function
        ])
    
    # Optimize
    optimizer = MIPROv2(
        metric=metric,
        prompt_model=dspy.LM(model="openai/gpt-4o-mini"),
        task_model=dspy.LM(model="openai/gpt-4o-mini"),
        auto='light'
    )
    
    optimized_module = optimizer.compile(
        pdf_module,
        trainset=trainset,
        requires_permission_to_run=False
    )
    
    # Use optimized module
    result = optimized_module("What are the SPI pins?")
    print(f"Optimized extraction result:")
    print(f"Pins: {result.pin_number}")
    print(f"Names: {result.pin_name}")


if __name__ == "__main__":
    # Run all examples
    example_simple_qa()
    example_pinout_extraction()
    example_specification_extraction()
    example_in_dspy_chain()
    example_multiple_pdfs()
    example_with_optimization()