from crewai import Task

class ResearchTasks:
    """Factory for creating tasks used in the research conversion process."""
    
    @staticmethod
    def create_research_task(researcher_agent, pdf_path=None):
        """Create a task for researching and extracting insights.
        
        Args:
            researcher_agent: The agent assigned to this task
            pdf_path: Optional path to PDF for context
            
        Returns:
            Configured Task instance
        """
        pdf_context = f" from the document at {pdf_path}" if pdf_path else ""
        
        return Task(
            description=f"""STRICTLY extract and organize content{pdf_context} exactly as it appears in the original document.

CRITICAL RULES:
1. DO NOT add any information that is not explicitly present in the source PDF
2. DO NOT make creative interpretations or expansions
3. DO NOT reorganize or restructure the content's original flow
4. Copy text verbatim where possible, maintaining exact wording
5. Preserve all numerical data, statistics, and figures exactly as they appear

EXTRACTION PROCESS:
1. First, use 'extract all' to get the complete document content
2. Then extract specific sections in order:
   - Title (search for title at start of document)
   - Authors (search for author information)
   - Abstract/Introduction (search for these sections)
   - Main Content (extract section by section)
   - Conclusions (search for conclusion section)
   - References (search for references section)

For each section:
- Use exact quotes and passages from the document
- Include page numbers when available
- Maintain original paragraph structure
- Keep all numerical values and data points unchanged
- Preserve technical terminology exactly as used
- Keep citations and references in their original format

Your task is to act as a precise extractor, not an interpreter or rewriter.
If any section is not found, explicitly state that it's not present in the document.""",
            agent=researcher_agent,
            expected_output="A faithful, verbatim reproduction of the source document's content, maintaining original structure, wording, and data, with page numbers preserved."
        )
    
    @staticmethod
    def create_content_creation_task(content_creator, research_task, brand_context=""):
        """Create a task for content creation.
        
        Args:
            content_creator: The agent assigned to this task
            research_task: The preceding research task (for context)
            brand_context: Optional branding guidelines
            
        Returns:
            Configured Task instance
        """
        return Task(
            description=f"""Using ONLY the extracted content from the research task, create a structured document.

CRITICAL REQUIREMENTS:
1. Use ONLY information present in the source material
2. DO NOT add any new information, interpretations, or expansions
3. DO NOT modify or paraphrase technical content
4. Maintain all original data, figures, and statistics exactly
5. Keep all technical terminology unchanged

Structure the content as follows:
1. Title: Use the exact original title
2. Authors: List all authors as shown in the source
3. Main Content: Follow the original document's structure
   - Keep original section headings
   - Maintain original paragraph organization
   - Use verbatim quotes for key findings
   - Keep all numerical data unchanged
4. Conclusions: Use the original conclusions
5. References: Include all original references in their exact format

Brand Context (if applicable):
{brand_context}

Note: Apply branding ONLY to visual formatting, never to modify the actual content.""",
            agent=content_creator,
            expected_output="A structured document that faithfully represents the original content without any creative additions or modifications.",
            context=[research_task]
        )
    
    @staticmethod
    def create_formatting_task(formatter_agent, creation_task):
        """Create a task for formatting content.
        
        Args:
            formatter_agent: The agent assigned to this task
            creation_task: The preceding content creation task (for context)
            
        Returns:
            Configured Task instance
        """
        return Task(
            description="""Format the document while preserving exact content.

CRITICAL RULES:
1. DO NOT modify any content
2. DO NOT add or remove information
3. DO NOT rewrite or paraphrase
4. Maintain all technical terms exactly

Apply only these formatting elements:
1. Basic Structure:
   - # for document title
   - ## for main sections
   - ### for subsections
   - Basic lists (when present in original)
   - Simple tables (when present in original)

2. Text Formatting:
   - Preserve original paragraph breaks
   - Maintain original list structures
   - Keep table layouts as in source
   - Retain original emphasis (bold/italic) if present

3. NO modifications to:
   - Technical terminology
   - Numerical values
   - Equations or formulas
   - Citations or references
   - Author names or affiliations""",
            agent=formatter_agent,
            expected_output="A cleanly formatted document with the exact same content as the source material.",
            context=[creation_task]
        ) 