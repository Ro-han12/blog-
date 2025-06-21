import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Import modular components
from whitepaper.tools import ResearchTools
from whitepaper.agents import ResearchAgents
from whitepaper.tasks import ResearchTasks
from whitepaper.crews import ResearchCrews
from whitepaper.exporters import ContentExporters

# Load environment variables
load_dotenv()

class ResearchConverter:
    def __init__(self, gemini_api_key=None, output_dir="exports"):
        """Initialize the research converter with API keys and configuration."""
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not provided. Set GOOGLE_API_KEY environment variable or pass it as an argument.")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the LLM
        self.llm = ResearchTools.create_gemini_llm(self.gemini_api_key)
    
    def process_pdf(self, pdf_path, brand_guidelines=None, output_format="both"):
        """
        Process a PDF document and convert it to branded content.
        
        Args:
            pdf_path: Path to the PDF file
            brand_guidelines: Optional path to brand guidelines document
            output_format: 'pdf', 'html', or 'both'
        
        Returns:
            Paths to the generated output files
        """
        # Initialize tools
        pdf_tool = ResearchTools.create_pdf_search_tool(pdf_path, self.gemini_api_key)
        
        # Load brand guidelines if provided
        brand_context = ""
        if brand_guidelines:
            brand_tool = ResearchTools.create_pdf_search_tool(brand_guidelines, self.gemini_api_key)
            brand_context = brand_tool.search("Extract key branding elements, tone, voice, and styling guidelines")
        
        # Create the agents
        researcher = ResearchAgents.create_researcher(self.llm, pdf_tool)
        content_creator = ResearchAgents.create_content_creator(self.llm)
        formatter = ResearchAgents.create_formatter(self.llm)
        
        # Create the tasks
        research_task = ResearchTasks.create_research_task(researcher, pdf_path)
        creation_task = ResearchTasks.create_content_creation_task(content_creator, research_task, brand_context)
        formatting_task = ResearchTasks.create_formatting_task(formatter, creation_task)
        
        # Create and run the crew
        crew = ResearchCrews.create_research_to_content_crew(
            agents=[researcher, content_creator, formatter],
            tasks=[research_task, creation_task, formatting_task]
        )
        
        result = crew.kickoff()
        
        # Generate output files
        outputs = []
        filename_base = Path(pdf_path).stem
        
        if output_format in ["pdf", "both"]:
            pdf_path = ContentExporters.export_as_pdf(result, filename_base, self.output_dir)
            if pdf_path:
                outputs.append(pdf_path)
            
        if output_format in ["html", "both"]:
            html_path = ContentExporters.export_as_html(result, filename_base, self.output_dir)
            outputs.append(html_path)
        
        return outputs


def main():
    parser = argparse.ArgumentParser(description="Convert research PDFs to branded content")
    parser.add_argument("pdf_path", help="Path to the PDF document to process")
    parser.add_argument("--brand", help="Optional path to brand guidelines document")
    parser.add_argument("--output", choices=["pdf", "html", "both"], default="both", 
                       help="Output format (pdf, html, or both)")
    parser.add_argument("--output-dir", default="exports", help="Directory to save outputs")
    parser.add_argument("--api-key", help="Gemini API key (if not set in environment)")
    
    args = parser.parse_args()
    
    converter = ResearchConverter(gemini_api_key=args.api_key, output_dir=args.output_dir)
    outputs = converter.process_pdf(args.pdf_path, args.brand, args.output)
    
    print(f"Generated outputs: {', '.join(str(path) for path in outputs if path)}")


if __name__ == "__main__":
    main()
