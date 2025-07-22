"""
Advanced integration examples showing PDF RAG Module in complex DSPy pipelines.
"""

import dspy
from pdf_rag_module import PDFRAGModule
from signatures import (
    SpecificationExtractionSignature,
    ComparisonSignature,
    TroubleshootingSignature,
    CodeExtractionSignature
)

# Configure DSPy
dspy.settings.configure(lm=dspy.LM(model="openai/gpt-4-turbo"))


class DatasheetValidator(dspy.Module):
    """
    Validate component compatibility by checking specifications across datasheets.
    """
    
    def __init__(self, component_pdfs):
        super().__init__()
        # Create PDF modules for each component
        self.components = {
            name: PDFRAGModule(path, signature=SpecificationExtractionSignature)
            for name, path in component_pdfs.items()
        }
        
        # Comparison module
        self.compare = dspy.ChainOfThought(ComparisonSignature)
        
        # Validation logic
        self.validate = dspy.ChainOfThought(
            "component_specs, requirements -> validation_result, issues, recommendations"
        )
    
    def forward(self, spec_to_check, requirements):
        # Extract specifications from each component
        specs = {}
        for name, module in self.components.items():
            result = module(spec_to_check)
            specs[name] = {
                'parameter': result.parameter,
                'value': result.value,
                'range': result.min_max_range
            }
        
        # Validate against requirements
        validation = self.validate(
            component_specs=str(specs),
            requirements=requirements
        )
        
        return validation


class TroubleshootingAssistant(dspy.Module):
    """
    Intelligent troubleshooting assistant that combines PDF manuals with diagnostic reasoning.
    """
    
    def __init__(self, manual_path, knowledge_base_path=None):
        super().__init__()
        # Manual PDF module
        self.manual = PDFRAGModule(
            manual_path,
            signature=TroubleshootingSignature,
            retrieve_k=7  # Get more context for troubleshooting
        )
        
        # Optional knowledge base
        if knowledge_base_path:
            self.knowledge = PDFRAGModule(knowledge_base_path)
        else:
            self.knowledge = None
        
        # Diagnostic reasoning
        self.diagnose = dspy.ChainOfThought(
            "symptoms, manual_guidance, past_cases -> diagnosis, confidence, next_steps"
        )
        
        # Solution verification
        self.verify = dspy.ChainOfThought(
            "proposed_solution, constraints -> feasibility, risks, alternatives"
        )
    
    def forward(self, problem_description, constraints=None):
        # Get troubleshooting info from manual
        manual_result = self.manual(problem_description)
        
        # Get additional context from knowledge base if available
        past_cases = ""
        if self.knowledge:
            kb_result = self.knowledge(f"Similar issues to: {problem_description}")
            past_cases = kb_result.answer
        
        # Diagnose the problem
        diagnosis = self.diagnose(
            symptoms=problem_description,
            manual_guidance=f"Causes: {manual_result.possible_causes}\nSolutions: {manual_result.solutions}",
            past_cases=past_cases
        )
        
        # Verify solution feasibility
        if constraints:
            verification = self.verify(
                proposed_solution=diagnosis.next_steps,
                constraints=constraints
            )
            diagnosis.feasibility = verification.feasibility
            diagnosis.alternatives = verification.alternatives
        
        return diagnosis


class CodeGeneratorFromDocs(dspy.Module):
    """
    Generate code based on documentation examples and specifications.
    """
    
    def __init__(self, api_docs_path, examples_path=None):
        super().__init__()
        # API documentation
        self.api_docs = PDFRAGModule(
            api_docs_path,
            signature=CodeExtractionSignature
        )
        
        # Code examples if available
        if examples_path:
            self.examples = PDFRAGModule(
                examples_path,
                signature=CodeExtractionSignature
            )
        else:
            self.examples = None
        
        # Code generation
        self.generate = dspy.ChainOfThought(
            "task, api_reference, examples -> code, explanation, dependencies"
        )
        
        # Code optimization
        self.optimize = dspy.ChainOfThought(
            "code, requirements -> optimized_code, improvements"
        )
    
    def forward(self, task_description, language="python", optimize=True):
        # Get relevant API documentation
        api_result = self.api_docs(f"{task_description} API {language}")
        
        # Get examples if available
        example_code = ""
        if self.examples:
            example_result = self.examples(f"{task_description} example {language}")
            example_code = example_result.code_snippet
        
        # Generate code
        generated = self.generate(
            task=task_description,
            api_reference=api_result.code_snippet,
            examples=example_code
        )
        
        # Optimize if requested
        if optimize:
            optimized = self.optimize(
                code=generated.code,
                requirements=f"Language: {language}, Task: {task_description}"
            )
            generated.code = optimized.optimized_code
            generated.optimizations = optimized.improvements
        
        return generated


class ResearchAssistant(dspy.Module):
    """
    Research assistant that can work with multiple PDFs to answer complex questions.
    """
    
    def __init__(self, pdf_library):
        super().__init__()
        # Create PDF modules for each document
        self.library = {
            name: PDFRAGModule(path, collection_name=f"research_{i}")
            for i, (name, path) in enumerate(pdf_library.items())
        }
        
        # Research planning
        self.plan_research = dspy.ChainOfThought(
            "question -> search_strategy, key_terms, relevant_docs"
        )
        
        # Information synthesis
        self.synthesize = dspy.ChainOfThought(
            "findings, question -> comprehensive_answer, confidence, sources"
        )
        
        # Citation generation
        self.cite = dspy.ChainOfThought(
            "content, source -> formatted_citation"
        )
    
    def forward(self, research_question):
        # Plan research strategy
        plan = self.plan_research(question=research_question)
        
        # Search across relevant documents
        findings = {}
        sources = {}
        
        for doc_name, module in self.library.items():
            # Check if this document is relevant based on the plan
            if doc_name.lower() in plan.relevant_docs.lower() or "all" in plan.relevant_docs.lower():
                result = module(research_question)
                if result.answer and len(result.answer) > 50:  # Filter out empty or minimal responses
                    findings[doc_name] = result.answer
                    sources[doc_name] = result.retrieved_chunks[:2]  # Keep top 2 chunks for citation
        
        # Synthesize findings
        synthesis = self.synthesize(
            findings=str(findings),
            question=research_question
        )
        
        # Generate citations
        citations = []
        for doc_name, chunks in sources.items():
            for chunk in chunks:
                citation = self.cite(
                    content=chunk[:200],  # First 200 chars
                    source=doc_name
                )
                citations.append(citation.formatted_citation)
        
        synthesis.citations = citations
        synthesis.sources_used = list(findings.keys())
        
        return synthesis


# Example usage
if __name__ == "__main__":
    # Example 1: Component Validation
    print("=== Component Compatibility Validation ===")
    validator = DatasheetValidator({
        "MCU": "docs/esp32-wroom-32e_esp32-wroom-32ue_datasheet_en.pdf",
        "Processor": "docs/omap-l138.pdf"
    })
    
    result = validator(
        spec_to_check="operating voltage",
        requirements="All components must operate at 3.3V"
    )
    print(f"Validation: {result.validation_result}")
    print(f"Issues: {result.issues}\n")
    
    # Example 2: Troubleshooting
    print("=== Troubleshooting Assistant ===")
    troubleshooter = TroubleshootingAssistant("docs/sh7780_series.pdf")
    
    diagnosis = troubleshooter(
        problem_description="System fails to boot, no output on UART",
        constraints="Cannot modify hardware connections"
    )
    print(f"Diagnosis: {diagnosis.diagnosis}")
    print(f"Next Steps: {diagnosis.next_steps}\n")
    
    # Example 3: Research Assistant
    print("=== Research Assistant ===")
    researcher = ResearchAssistant({
        "ESP32": "docs/esp32-wroom-32e_esp32-wroom-32ue_datasheet_en.pdf",
        "OMAP": "docs/omap-l138.pdf",
        "SH7780": "docs/sh7780_series.pdf"
    })
    
    research = researcher(
        "Compare the interrupt handling capabilities across different processors"
    )
    print(f"Research Answer: {research.comprehensive_answer}")
    print(f"Sources Used: {research.sources_used}")
    print(f"Confidence: {research.confidence}")