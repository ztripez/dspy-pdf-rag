"""
Collection of DSPy signatures for different PDF extraction tasks.
"""

import dspy


class PDFQASignature(dspy.Signature):
    """General question answering from PDF content."""
    context = dspy.InputField(desc="relevant sections from the PDF")
    query = dspy.InputField(desc="user question about the PDF content")
    answer = dspy.OutputField(desc="comprehensive answer based on the PDF content")


class PinoutExtractionSignature(dspy.Signature):
    """Extract pinout information from technical documentation."""
    context = dspy.InputField(desc="relevant sections from the datasheet")
    query = dspy.InputField(desc="user query about pinout")
    
    pin_number = dspy.OutputField(desc="pin number or range")
    pin_name = dspy.OutputField(desc="pin name/label")
    pin_function = dspy.OutputField(desc="pin function description")
    electrical_characteristics = dspy.OutputField(desc="voltage, current, or other electrical specs")


class SpecificationExtractionSignature(dspy.Signature):
    """Extract technical specifications from documentation."""
    context = dspy.InputField(desc="relevant sections from the document")
    query = dspy.InputField(desc="specification query")
    
    parameter = dspy.OutputField(desc="specification parameter name")
    value = dspy.OutputField(desc="specification value with units")
    conditions = dspy.OutputField(desc="conditions or notes for this specification")
    min_max_range = dspy.OutputField(desc="minimum and maximum values if applicable")


class TableExtractionSignature(dspy.Signature):
    """Extract and structure table data from PDFs."""
    context = dspy.InputField(desc="relevant sections containing tables")
    query = dspy.InputField(desc="what table data to extract")
    
    table_title = dspy.OutputField(desc="title or description of the table")
    headers = dspy.OutputField(desc="column headers as a list")
    rows = dspy.OutputField(desc="table data as a list of rows")
    notes = dspy.OutputField(desc="any footnotes or additional information")


class SummarySignature(dspy.Signature):
    """Generate summaries of PDF sections."""
    context = dspy.InputField(desc="PDF content to summarize")
    query = dspy.InputField(desc="what aspect to focus the summary on")
    
    summary = dspy.OutputField(desc="concise summary of the relevant content")
    key_points = dspy.OutputField(desc="bullet points of main ideas")
    technical_terms = dspy.OutputField(desc="important technical terms mentioned")


class ComparisonSignature(dspy.Signature):
    """Compare different items or specifications in a PDF."""
    context = dspy.InputField(desc="relevant sections for comparison")
    query = dspy.InputField(desc="what items or features to compare")
    
    items_compared = dspy.OutputField(desc="list of items being compared")
    similarities = dspy.OutputField(desc="common features or specifications")
    differences = dspy.OutputField(desc="key differences between items")
    recommendation = dspy.OutputField(desc="recommendation based on comparison")


class TroubleshootingSignature(dspy.Signature):
    """Extract troubleshooting information from technical manuals."""
    context = dspy.InputField(desc="relevant troubleshooting sections")
    query = dspy.InputField(desc="problem or issue description")
    
    problem_description = dspy.OutputField(desc="clear description of the issue")
    possible_causes = dspy.OutputField(desc="list of potential causes")
    solutions = dspy.OutputField(desc="step-by-step solutions or fixes")
    preventive_measures = dspy.OutputField(desc="how to prevent this issue")


class CodeExtractionSignature(dspy.Signature):
    """Extract code examples or configurations from documentation."""
    context = dspy.InputField(desc="relevant sections containing code")
    query = dspy.InputField(desc="what code or configuration to extract")
    
    code_snippet = dspy.OutputField(desc="the relevant code example")
    language = dspy.OutputField(desc="programming language or format")
    description = dspy.OutputField(desc="what the code does")
    dependencies = dspy.OutputField(desc="required libraries or prerequisites")