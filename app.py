# Fix sqlite3 version issue (required for chromadb)
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import streamlit as st
import os
import tempfile
import base64
from pathlib import Path
import traceback
import uuid
import shutil
import time
import chardet
import io

# ChromaDB configuration for Streamlit Cloud
os.makedirs("chroma_db", exist_ok=True)
os.environ["CHROMADB_IMPLEMENTATION"] = "duckdb"

def get_chroma_client():
    """Initialize and return ChromaDB client with proper settings."""
    from chromadb import Client
    from chromadb.config import Settings
    return Client(Settings(persist_directory="chroma_db"))

# Initialize ChromaDB client
try:
    chroma_client = get_chroma_client()
except Exception as e:
    st.error(f"ChromaDB initialization error: {e}")

from crewai import Crew, Task, Agent
from pypdf import PdfWriter, PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import google.generativeai as genai
from dotenv import load_dotenv
import re
from textwrap import dedent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')

# App version
APP_VERSION = "1.0.0"

# Available models
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash", 
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-1.5-pro"
]

# Inline BlogAgents class
class BlogAgents:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0.7
        )

    def create_research_agent(self):
        return Agent(
            role='Research Specialist',
            goal='Gather comprehensive information about the given topic while considering target audience, industry, and content goals',
            backstory=dedent("""
                You are an expert researcher with years of experience in gathering
                and analyzing information from various sources. Your expertise lies
                in finding accurate and relevant information quickly, while ensuring
                the content aligns with the target audience's knowledge level,
                industry context, and specific content goals. You excel at adapting
                research depth and focus based on whether the content needs to educate,
                convert, rank, or entertain.
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_nlp_agent(self):
        return Agent(
            role='NLP Specialist',
            goal='Process and analyze the gathered information using NLP techniques while maintaining the specified tone and style',
            backstory=dedent("""
                You are an NLP expert who specializes in text processing and analysis.
                You can identify key themes, extract important information, and structure
                content effectively. You excel at adapting the content's tone and style
                to match the specified requirements, whether it's conversational,
                storytelling, humorous, or professional. You ensure the content structure
                aligns with the chosen blog type (how-to, listicle, case study, etc.)
                while maintaining engagement and readability.
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_writer_agent(self):
        return Agent(
            role='Content Writer',
            goal='Create engaging and well-structured blog content that matches the specified audience, tone, and content goals',
            backstory=dedent("""
                You are a professional content writer with expertise in creating
                engaging and informative blog posts. You know how to structure
                content effectively and maintain reader interest. You excel at
                adapting your writing style to different audiences (students,
                engineers, PhD researchers, etc.) and can seamlessly switch between
                various tones (conversational, storytelling, humorous, etc.). You
                understand how to optimize content for different goals (educate,
                convert, rank, entertain) while maintaining the appropriate level
                of technical depth and engagement for the target audience.
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_plagiarism_checker_agent(self):
        return Agent(
            role='Plagiarism Checker',
            goal='Analyze content originality and provide a plagiarism score based on specific criteria while considering the content type and audience',
            backstory=dedent("""
                You are an expert in content verification and plagiarism detection.
                You analyze content based on the following specific criteria, each worth 20 points:

                1. Writing Style Originality (20 points):
                   - Unique sentence structures
                   - Personal voice and tone
                   - Creative expression
                   - Avoidance of clichés
                   - Adaptation to specified tone and audience

                2. Content Structure (20 points):
                   - Original organization
                   - Unique flow and transitions
                   - Creative section arrangement
                   - Innovative presentation
                   - Alignment with blog type and content goal

                3. Language and Vocabulary (20 points):
                   - Unique word choices
                   - Varied vocabulary
                   - Creative metaphors
                   - Original expressions
                   - Appropriate for target audience

                4. Idea Development (20 points):
                   - Original perspectives
                   - Unique insights
                   - Creative connections
                   - Innovative approaches
                   - Relevance to industry/domain

                5. Technical Elements (20 points):
                   - Original examples
                   - Unique data presentation
                   - Creative formatting
                   - Innovative use of technical terms
                   - Alignment with content goals

                For each criterion, provide:
                - Score (0-20)
                - Specific examples from the text
                - Areas for improvement
                - Recommendations

                Calculate the final score (0-100) by summing all criteria scores.
                A score of:
                - 90-100: Highly original
                - 70-89: Mostly original
                - 50-69: Moderately original
                - 30-49: Needs improvement
                - 0-29: Significant concerns
            """),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

# Inline ResearchConverter class
class ResearchConverter:
    def __init__(self, gemini_api_key=None, output_dir="exports", model_name="gemini-2.0-flash"):
        """Initialize the research converter with API keys and configuration."""
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not provided. Set GOOGLE_API_KEY environment variable or pass it as an argument.")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.gemini_api_key,
            temperature=0.7
        )

# Inline ResearchAgents class
class ResearchAgents:
    @staticmethod
    def create_researcher(llm, search_tool):
        return Agent(
            role='Research Analyst',
            goal='Extract and organize information from research documents',
            backstory=dedent("""
                You are an expert research analyst with deep experience in academic
                and technical document analysis. You excel at extracting key information,
                identifying main arguments, and organizing content logically while
                maintaining the original context and accuracy of the source material.
            """),
            llm=llm,
            tools=[search_tool],
            verbose=True,
            allow_delegation=False
        )

    @staticmethod
    def create_content_creator(llm):
        return Agent(
            role='Content Creator',
            goal='Transform research findings into well-structured, readable content',
            backstory=dedent("""
                You are a skilled content creator who specializes in transforming
                complex research into clear, engaging content. You maintain the
                technical accuracy while improving readability and flow.
            """),
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

    @staticmethod
    def create_formatter(llm):
        return Agent(
            role='Content Formatter',
            goal='Format content for various output formats while maintaining structure',
            backstory=dedent("""
                You are an expert in content formatting and presentation. You ensure
                that content is properly structured for different output formats
                while maintaining readability and professional appearance.
            """),
            llm=llm,
            verbose=True,
            allow_delegation=False
        )

# Inline ResearchTasks class
class ResearchTasks:
    @staticmethod
    def create_research_task(researcher, pdf_path):
        return Task(
            description=f"""Extract and organize the following content exactly as it appears in the document.
Use the search tool to extract content - it will provide either Hindi or English text depending on the translation setting.

CRITICAL RULES:
1. DO NOT add any information that is not in the document
2. DO NOT make creative interpretations or expansions
3. DO NOT reorganize or restructure the content's original flow
4. Copy text verbatim where possible, maintaining exact wording
5. Preserve all numerical data, statistics, and figures exactly as they appear

Extract and organize the following sections IN ORDER:
1. Title (from the beginning of the document)
2. Authors (if present)
3. Abstract/Introduction
4. Main Content (maintaining original structure)
5. Conclusions
6. References

For each section:
- Use exact quotes from the document
- Maintain original paragraph structure
- Keep all numerical values unchanged
- Preserve technical terminology exactly
- Keep citations in their original format""",
            agent=researcher,
            expected_output="A faithful, verbatim reproduction of the source document's content, maintaining original structure, wording, and data."
        )

    @staticmethod
    def create_content_creation_task(content_creator, research_task, brand_context):
        return Task(
            description=f"""Create well-structured content based on the research findings.
Maintain the original information while improving readability and flow.
Brand Context: {brand_context}""",
            agent=content_creator,
            expected_output="Well-structured content that maintains original information while improving readability."
        )

    @staticmethod
    def create_formatting_task(formatter, creation_task):
        return Task(
            description="Format the content for professional presentation while maintaining structure and readability.",
            agent=formatter,
            expected_output="Properly formatted content ready for export."
        )

# Inline ResearchCrews class
class ResearchCrews:
    @staticmethod
    def create_research_to_content_crew(agents, tasks):
        return Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        )

# Inline ContentExporters class
class ContentExporters:
    @staticmethod
    def clean_text(text):
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
        return text.strip()

    @staticmethod
    def export_as_pdf(content, filename_base, export_dir):
        """Export content as PDF."""
        try:
            pdf_path = export_dir / f"{filename_base}.pdf"
            
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=12
            )

            story = []
            
            # Split content into paragraphs
            paragraphs = content.split('\n\n')
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    if len(story) == 0:  # First paragraph as title
                        story.append(Paragraph(paragraph, title_style))
                    else:
                        story.append(Paragraph(paragraph, body_style))
                    story.append(Spacer(1, 12))

            doc.build(story)
            return pdf_path
        except Exception as e:
            st.error(f"Error creating PDF: {str(e)}")
            return None

    @staticmethod
    def export_as_html(content, filename_base, export_dir):
        """Export content as HTML."""
        try:
            html_path = export_dir / f"{filename_base}.html"
            
            # Clean content for HTML
            cleaned_content = ContentExporters.clean_text(content)
            
            # Create HTML template
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{filename_base}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                    p {{ margin-bottom: 15px; }}
                </style>
            </head>
            <body>
                <h1>{filename_base}</h1>
                <div>{cleaned_content.replace(chr(10), '<br>')}</div>
            </body>
            </html>
            """
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            return html_path
        except Exception as e:
            st.error(f"Error creating HTML: {str(e)}")
            return None

# Helper functions
def clean_markdown(text):
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'[#*_~`]', '', text)
    return text

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def display_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_html(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)

def check_dependencies():
    try:
        import crewai
        import crewai_tools
        import langchain
        import langchain_google_genai
        import fpdf
        import markdown
        return True
    except ImportError as e:
        st.error(f"Missing dependency: {str(e)}")
        st.info("Please install all dependencies with: pip install -r requirements.txt")
        return False

def ensure_export_dir():
    export_dir = Path("exports")
    if not export_dir.exists():
        export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir

def detect_encoding(text_bytes):
    """Detect the encoding of the given bytes."""
    result = chardet.detect(text_bytes)
    return result

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.encode('utf-8')  # Convert to bytes for encoding detection
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def show_encoding_details(text_bytes, encoding_result):
    """Show detailed information about the encoding detection and decoding process."""
    st.subheader("Encoding Detection Details")
    
    # Show byte patterns
    st.write("First 20 bytes in different formats:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Hexadecimal:")
        hex_bytes = ' '.join(f'{b:02x}' for b in text_bytes[:20])
        st.code(hex_bytes)
    
    with col2:
        st.write("Binary:")
        binary_bytes = ' '.join(f'{b:08b}' for b in text_bytes[:20])
        st.code(binary_bytes)
    
    with col3:
        st.write("ASCII (if printable):")
        ascii_bytes = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in text_bytes[:20])
        st.code(ascii_bytes)
    
    # Show encoding detection confidence breakdown
    st.write("Encoding Detection Confidence:")
    confidence = encoding_result['confidence']
    st.progress(confidence)
    
    # Show common encoding patterns
    st.write("Common Encoding Patterns Found:")
    patterns = {
        'UTF-8': any(b & 0x80 for b in text_bytes[:100]),  # Check for high bit set
        'ASCII': all(b < 128 for b in text_bytes[:100]),   # Check if all bytes < 128
        'UTF-16': any(b == 0 for b in text_bytes[:100]),   # Check for null bytes
    }
    
    for encoding, found in patterns.items():
        st.write(f"- {encoding}: {'✓' if found else '✗'}")
    
    return patterns

def blog_writer_page():
    st.title("AI Blog Writer")
    st.write("Generate high-quality blog posts using AI agents")

    # Initialize session state for blog
    if 'blog_content' not in st.session_state:
        st.session_state.blog_content = None
    if 'content_analysis' not in st.session_state:
        st.session_state.content_analysis = None
    if 'plagiarism_score' not in st.session_state:
        st.session_state.plagiarism_score = None

    # Input section
    topic = st.text_area("Enter your blog topic or search query:", height=100)

    # Add customization options
    st.subheader("Blog Customization")
    col1, col2 = st.columns(2)

    with col1:
        audience = st.selectbox(
            "Target Audience",
            ["Student", "Engineer", "PhD Researcher", "Business Professional", "General Public", "Technical Expert", "Academic"]
        )
        
        tone = st.selectbox(
            "Writing Tone",
            ["Conversational", "Storytelling", "Humorous", "Professional", "Academic", "Technical", "Casual"]
        )
        
        industry = st.selectbox(
            "Industry/Domain",
            ["EdTech", "Finance", "Legal", "Healthcare", "Technology", "Marketing", "Science", "Education", "Business", "Other"]
        )

    with col2:
        blog_type = st.selectbox(
            "Blog Type",
            ["How-to Guide", "Listicle", "Case Study", "Opinion Piece", "Technical Tutorial", "News Analysis", "Review", "Research Summary"]
        )
        
        content_goal = st.selectbox(
            "Content Goal",
            ["Educate", "Rank (SEO)", "Convert", "Explain", "Entertain", "Inform", "Persuade"]
        )
        
        word_limit = st.slider(
            "Word Limit",
            min_value=700,
            max_value=2000,
            value=1000,
            step=50,
            help="Set the desired word count for the generated blog post."
        )

    if st.button("Generate Blog"):
        if topic:
            with st.spinner("Generating your blog post..."):
                # Initialize agents with selected model
                blog_agents = BlogAgents(model_name=st.session_state.selected_model)
                research_agent = blog_agents.create_research_agent()
                nlp_agent = blog_agents.create_nlp_agent()
                writer_agent = blog_agents.create_writer_agent()

                # Create tasks
                research_task = Task(
                    description=f"""Research and gather information about: {topic}
                    Target Audience: {audience}
                    Industry/Domain: {industry}
                    Blog Type: {blog_type}
                    Content Goal: {content_goal}""",
                    agent=research_agent,
                    expected_output="A comprehensive research summary with key points, statistics, and relevant information about the topic."
                )

                nlp_task = Task(
                    description=f"""Process and analyze the gathered information using NLP techniques.
                    Consider the following parameters:
                    - Target Audience: {audience}
                    - Writing Tone: {tone}
                    - Industry/Domain: {industry}
                    - Blog Type: {blog_type}
                    - Content Goal: {content_goal}""",
                    agent=nlp_agent,
                    expected_output="An analyzed and structured outline with key points organized for blog writing, incorporating NLP insights."
                )

                writing_task = Task(
                    description=f"""Write an engaging blog post based on the processed information.
                    Follow these guidelines:
                    - Target Audience: {audience}
                    - Writing Tone: {tone}
                    - Industry/Domain: {industry}
                    - Blog Type: {blog_type}
                    - Content Goal: {content_goal}
                    - Word Limit: {word_limit} words (strictly adhere to this range)
                    Ensure the content is well-structured and meets the specified requirements.""",
                    agent=writer_agent,
                    expected_output="A complete, well-structured blog post that meets all specified requirements and guidelines."
                )

                # Create and run the crew
                crew = Crew(
                    agents=[research_agent, nlp_agent, writer_agent],
                    tasks=[research_task, nlp_task, writing_task],
                    verbose=True
                )

                result = crew.kickoff()
                st.session_state.blog_content = result

        else:
            st.error("Please enter a topic to generate a blog post.")

    # Display results
    if st.session_state.blog_content:
        st.subheader("Generated Blog Post")
        st.write(st.session_state.blog_content)

        # Clean markdown from content
        cleaned_content = clean_markdown(st.session_state.blog_content)

        # Generate HTML with minimal styling
        html_template = '<html><head></head><body><h1>{0}</h1><div>{1}</div></body></html>'
        
        html_content = html_template.format(
            topic,
            cleaned_content.replace('\n', '<br>')
        )

        # Create HTML and PDF outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save files
            html_path = os.path.join(temp_dir, "blog_post.html")
            with open(html_path, 'w') as f:
                f.write(html_content)

            pdf_path = os.path.join(temp_dir, "blog_post.pdf")
            
            # Generate PDF
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=12
            )

            content = []
            content.append(Paragraph(topic, title_style))
            content.append(Spacer(1, 12))

            for paragraph in cleaned_content.split('\n\n'):
                if paragraph.strip():
                    content.append(Paragraph(paragraph, body_style))
                    content.append(Spacer(1, 12))

            doc.build(content)

            # Display download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(get_binary_file_downloader_html(html_path, "HTML"), unsafe_allow_html=True)
            with col2:
                st.markdown(get_binary_file_downloader_html(pdf_path, "PDF"), unsafe_allow_html=True)

            # Add content analysis and plagiarism check buttons
            st.subheader("Content Analysis Tools")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Analyze Content"):
                    with st.spinner("Analyzing content..."):
                        prompt = f"""Analyze the following blog content and provide a detailed report including:
                        1. Content Quality Assessment
                        2. Key Points and Main Arguments
                        3. Writing Style Analysis
                        4. Potential Improvements
                        5. Originality Assessment (based on common patterns and structures)

                        Blog Content:
                        {st.session_state.blog_content}
                        """
                        
                        response = model.generate_content(prompt)
                        st.session_state.content_analysis = response.text

            with col2:
                if st.button("Check Plagiarism"):
                    with st.spinner("Checking for plagiarism..."):
                        # Initialize plagiarism checker agent
                        blog_agents = BlogAgents(model_name=st.session_state.selected_model)
                        plagiarism_agent = blog_agents.create_plagiarism_checker_agent()
                        
                        # Create plagiarism check task
                        plagiarism_task = Task(
                            description=f"""Analyze the following content and provide:
                            1. A plagiarism score (0-100, where 100 is completely original)
                            2. Detailed analysis of writing patterns
                            3. Specific areas that might need improvement
                            4. Recommendations for enhancing originality

                            Content to analyze:
                            {st.session_state.blog_content}
                            """,
                            agent=plagiarism_agent,
                            expected_output="A detailed plagiarism analysis report with score and recommendations."
                        )
                        
                        # Run the plagiarism check
                        result = plagiarism_task.execute()
                        st.session_state.plagiarism_score = result

            # Display content analysis if available
            if st.session_state.content_analysis:
                st.subheader("Content Analysis Report")
                st.write(st.session_state.content_analysis)

            # Display plagiarism score if available
            if st.session_state.plagiarism_score:
                st.subheader("Plagiarism Analysis")
                
                # Create tabs for different aspects of the analysis
                tab1, tab2 = st.tabs(["Score Summary", "Detailed Analysis"])
                
                with tab1:
                    # Extract and display the overall score
                    score_text = st.session_state.plagiarism_score
                    if "Score:" in score_text:
                        score = score_text.split("Score:")[1].split("\n")[0].strip()
                        st.metric("Originality Score", score)
                    
                    # Display score interpretation
                    st.write("Score Interpretation:")
                    st.write("""
                    - 90-100: Highly original
                    - 70-89: Mostly original
                    - 50-69: Moderately original
                    - 30-49: Needs improvement
                    - 0-29: Significant concerns
                    """)
                
                with tab2:
                    # Display the detailed analysis
                    st.write(st.session_state.plagiarism_score)

def decode_indic_text(text):
    """Try multiple encodings to decode text, with special handling for Indic scripts."""
    # Common patterns to detect and fix
    replacements = {
        # Common Devanagari patterns
        '(R)': 'र',
        '+/-': 'ल',
        'É': 'ा',
        'è': 'ै',
        'ú': 'र',
        'ù': 'द',
        'þ': 'ह',
        'ò': 'क',
        'ä': 'े',
        'æ': 'ो',
        'Î': 'ि',
        'õ': 'ट',
        '¨': 'म',
        '½': 'ह',
        'Ê': 'ि',
        'º': 'स',
        'Æ': 'ं',
        'Ç': 'च',
        'ª': 'य',
        '¦': 'भ',
        'P': 'श',
        'ÿ': 'ह',
        'Ò': 'ी',
        'Ó': 'ी',
        'Ô': 'ी',
        'Õ': 'ी',
        'Ö': 'ी',
        '×': 'ी',
        'Ø': 'ी',
        'Ù': 'ी',
        'Ú': 'ी',
        'Û': 'ी',
        'Ü': 'ी',
        'Ý': 'ी',
        'Þ': 'ी',
        'ß': 'ी',
        'à': 'ी',
        'á': 'ी',
        'â': 'ी',
        'ã': 'ी',
        'ä': 'े',
        'å': 'े',
        'æ': 'ो',
        'ç': 'ो',
        'è': 'ै',
        'é': 'ै',
        'ê': 'ै',
        'ë': 'ै',
        'ì': 'ै',
        'í': 'ै',
        'î': 'ै',
        'ï': 'ै',
        'ð': 'ै',
        'ñ': 'ै',
        'ò': 'क',
        'ó': 'क',
        'ô': 'क',
        'õ': 'ट',
        'ö': 'ट',
        '÷': 'ट',
        'ø': 'ट',
        'ù': 'द',
        'ú': 'र',
        'û': 'र',
        'ü': 'र',
        'ý': 'र',
        'þ': 'ह',
        'ÿ': 'ह'
    }
    
    # First try to detect if it's Indic text by checking for common patterns
    indic_markers = ['É', 'ú', 'ù', 'þ', 'ò', 'ä', 'æ', 'Î', 'õ']
    is_indic = any(marker in text for marker in indic_markers)
    
    if is_indic:
        # Apply character replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Try to decode as ISCII if it still looks encoded
        try:
            # Convert to bytes if it's a string
            if isinstance(text, str):
                text = text.encode('utf-8', errors='ignore')
            
            # Try ISCII decoding
            decoded = text.decode('iscii', errors='replace')
            if sum(c.isprintable() for c in decoded) / len(decoded) > 0.8:
                return decoded
        except:
            pass
    
    # If not Indic or ISCII decoding failed, try other encodings
    encodings_to_try = [
        'utf-8',
        'utf-16',
        'utf-32',
        'iso-8859-1',
        'cp1252',
        'ascii',
        'latin1',
        'latin_1',
        'iso-8859-15',
        'windows-1252',
        'mac_roman'
    ]
    
    # First try chardet
    if isinstance(text, str):
        text = text.encode('utf-8', errors='ignore')
    
    detected = chardet.detect(text)
    if detected['confidence'] > 0.8:
        try:
            return text.decode(detected['encoding'])
        except:
            pass
    
    # Try each encoding
    for encoding in encodings_to_try:
        try:
            decoded = text.decode(encoding) if isinstance(text, bytes) else text.encode('utf-8').decode(encoding)
            if sum(c.isprintable() for c in decoded) / len(decoded) > 0.8:
                return decoded
        except:
            continue
    
    # If all fails, use utf-8 with replace
    return text.decode('utf-8', errors='replace') if isinstance(text, bytes) else text

def normalize_text(text):
    """Normalize text by removing control characters and normalizing whitespace."""
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common encoding artifacts
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    text = text.replace('Â', '')
    
    # Fix common Indic text artifacts
    text = text.replace('degü', 'deg')
    text = text.replace('(c)', '©')
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize multiple newlines
    
    return text.strip()

def translate_to_english(text, model):
    """Translate Hindi text to English using Gemini."""
    try:
        prompt = f"""Translate the following Hindi/Devanagari text to English. 
Keep technical terms as is, and maintain any numerical values or measurements exactly.
Preserve formatting and structure of the text.

Text to translate:
{text}

Please provide a clear and accurate translation while keeping technical terminology intact."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def create_search_tool(text):
    """Create a search tool that works with decoded/translated text."""
    def search_tool(query):
        """Custom search tool that searches through our processed text."""
        try:
            # If query is a dict with "extract all", return full text
            if isinstance(query, dict) and query.get("extract all"):
                return text
            
            # If it's a string query, do semantic search
            # For now, return full text as we don't have semantic search implemented
            return text
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return text
    
    return search_tool

def research_converter_page():
    st.title("Research PDF Converter")
    st.markdown("Upload a research PDF to convert it into structured, plain content")

    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.current_outputs = None
        st.session_state.processed_text = None
        st.session_state.translated_text = None
        st.session_state.search_tool = None

    # Ensure the exports directory exists
    ensure_export_dir()

    # Simple file uploaders
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Brand guidelines uploader (optional)
    include_brand = st.checkbox("Include brand guidelines")
    uploaded_brand = None
    if include_brand:
        uploaded_brand = st.file_uploader("Choose brand guidelines PDF (optional)", type="pdf")
    
    output_format = st.radio("Output format", ["pdf", "html", "both"], index=2)
    translate = st.checkbox("Translate to English", value=True)
    
    if uploaded_file is not None:
        # Process PDF with encoding detection
        with st.spinner("Analyzing document encoding..."):
            pdf_bytes = uploaded_file.getvalue()
            
            # Create temp directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                temp_pdf_path = temp_dir_path / uploaded_file.name
                
                # Save PDF for processing
                with open(temp_pdf_path, "wb") as f:
                    f.write(pdf_bytes)

                try:
                    # Extract text using PyPDF2
                    pdf_reader = PdfReader(temp_pdf_path)
                    processed_text = ""
                    raw_text = ""
                    
                    for i, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            # Store raw text for debugging
                            raw_text += f"\n=== Page {i} Raw ===\n{page_text}\n"
                            
                            # First try to decode any special encodings
                            decoded_text = decode_indic_text(page_text)
                            
                            # Normalize the decoded text
                            normalized_text = normalize_text(decoded_text)
                            
                            # Finally clean the text for display
                            cleaned_text = ContentExporters.clean_text(normalized_text)
                            processed_text += f"\n=== Page {i} ===\n{cleaned_text}\n"
                    
                    # Store both raw and processed text in session state
                    st.session_state.raw_text = raw_text
                    st.session_state.processed_text = processed_text
                    
                    # Translate if requested
                    if translate:
                        with st.spinner("Translating text..."):
                            # Create a temporary model instance for translation
                            temp_model = genai.GenerativeModel(st.session_state.selected_model)
                            translated_text = translate_to_english(processed_text, temp_model)
                            st.session_state.translated_text = translated_text
                            # Create search tool with translated text
                            st.session_state.search_tool = create_search_tool(translated_text)
                    else:
                        # Create search tool with processed text
                        st.session_state.search_tool = create_search_tool(processed_text)
                    
                    # Show document analysis
                    with st.expander("View Document Analysis"):
                        if translate:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.subheader("Raw Text")
                                st.text_area("Raw Content Preview", 
                                           st.session_state.raw_text[:1000] + "...", 
                                           height=200)
                            with col2:
                                st.subheader("Processed Hindi Text")
                                st.text_area("Processed Content Preview", 
                                           st.session_state.processed_text[:1000] + "...", 
                                           height=200)
                            with col3:
                                st.subheader("Translated Text")
                                st.text_area("English Translation", 
                                           st.session_state.translated_text[:1000] + "...", 
                                           height=200)
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Raw Text")
                                st.text_area("Raw Content Preview", 
                                           st.session_state.raw_text[:1000] + "...", 
                                           height=200)
                            with col2:
                                st.subheader("Processed Text")
                                st.text_area("Processed Content Preview", 
                                           st.session_state.processed_text[:1000] + "...", 
                                           height=200)
                        
                        # Show encoding analysis
                        st.subheader("Encoding Analysis")
                        raw_bytes = raw_text.encode('utf-8', errors='ignore')
                        processed_bytes = processed_text.encode('utf-8', errors='ignore')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Raw Text Encoding:")
                            st.json(chardet.detect(raw_bytes))
                        with col2:
                            st.write("Processed Text Encoding:")
                            st.json(chardet.detect(processed_bytes))
                        
                        show_encoding_details(processed_bytes, chardet.detect(processed_bytes))

                    if st.button("Process PDF"):
                        try:
                            # Initialize ResearchConverter with selected model
                            converter = ResearchConverter(gemini_api_key=GOOGLE_API_KEY, 
                                                       output_dir=temp_dir_path,
                                                       model_name=st.session_state.selected_model)
                            
                            # Use our custom search tool instead of the default one
                            researcher = ResearchAgents.create_researcher(converter.llm, st.session_state.search_tool)
                            content_creator = ResearchAgents.create_content_creator(converter.llm)
                            formatter = ResearchAgents.create_formatter(converter.llm)
                            
                            # Create tasks with explicit content passing
                            research_task = Task(
                                description=f"""Extract and organize the following content exactly as it appears in the document.
Use the search tool to extract content - it will provide either Hindi or English text depending on the translation setting.

CRITICAL RULES:
1. DO NOT add any information that is not in the document
2. DO NOT make creative interpretations or expansions
3. DO NOT reorganize or restructure the content's original flow
4. Copy text verbatim where possible, maintaining exact wording
5. Preserve all numerical data, statistics, and figures exactly as they appear

Extract and organize the following sections IN ORDER:
1. Title (from the beginning of the document)
2. Authors (if present)
3. Abstract/Introduction
4. Main Content (maintaining original structure)
5. Conclusions
6. References

For each section:
- Use exact quotes from the document
- Maintain original paragraph structure
- Keep all numerical values unchanged
- Preserve technical terminology exactly
- Keep citations in their original format""",
                                agent=researcher,
                                expected_output="A faithful, verbatim reproduction of the source document's content, maintaining original structure, wording, and data."
                            )
                            
                            creation_task = ResearchTasks.create_content_creation_task(
                                content_creator, research_task, ""
                            )
                            formatting_task = ResearchTasks.create_formatting_task(
                                formatter, creation_task
                            )
                            
                            # Create and run crew
                            crew = ResearchCrews.create_research_to_content_crew(
                                agents=[researcher, content_creator, formatter],
                                tasks=[research_task, creation_task, formatting_task]
                            )
                            
                            with st.spinner("Processing document..."):
                                result = crew.kickoff()
                                
                                if result:
                                    # Use ContentExporters for file generation
                                    outputs = []
                                    filename_base = Path(uploaded_file.name).stem
                                    if translate:
                                        filename_base += "_english"
                                    unique_id = f"{int(time.time())}"
                                    export_dir = ensure_export_dir() / unique_id
                                    export_dir.mkdir(exist_ok=True)
                                    
                                    if output_format in ["pdf", "both"]:
                                        pdf_path = ContentExporters.export_as_pdf(
                                            result, filename_base, export_dir
                                        )
                                        if pdf_path:
                                            outputs.append(pdf_path)
                                    
                                    if output_format in ["html", "both"]:
                                        html_path = ContentExporters.export_as_html(
                                            result, filename_base, export_dir
                                        )
                                        if html_path:
                                            outputs.append(html_path)
                                    
                                    if outputs:
                                        st.session_state.current_outputs = outputs
                                        st.success(f"Successfully processed: {uploaded_file.name}")
                                    else:
                                        st.error("No outputs were generated during conversion.")
                                else:
                                    st.error("Failed to process the document.")
                                    
                        except Exception as e:
                            st.error(f"Error during processing: {str(e)}")
                            st.error(traceback.format_exc())
                            
                except Exception as e:
                    st.error(f"Error reading PDF: {str(e)}")
                    st.error(traceback.format_exc())
    
    # Display outputs if they exist
    if st.session_state.get('current_outputs'):
        tabs = st.tabs(["PDF Output", "HTML Output"])
        
        with tabs[0]:
            pdf_output = [p for p in st.session_state.current_outputs if str(p).endswith('.pdf')]
            if pdf_output:
                st.subheader("PDF Preview")
                display_pdf(pdf_output[0])
                filename = uploaded_file.name if uploaded_file else "output"
                st.markdown(get_binary_file_downloader_html(pdf_output[0], f"PDF - {filename}"), unsafe_allow_html=True)
            else:
                st.warning("PDF output not available.")
        
        with tabs[1]:
            html_output = [p for p in st.session_state.current_outputs if str(p).endswith('.html')]
            if html_output:
                st.subheader("HTML Preview")
                display_html(html_output[0])
                filename = uploaded_file.name if uploaded_file else "output"
                st.markdown(get_binary_file_downloader_html(html_output[0], f"HTML - {filename}"), unsafe_allow_html=True)
            else:
                st.warning("HTML output not available.")

def main():
    st.set_page_config(
        page_title="AI Content Tools",
        page_icon="📚",
        layout="wide"
    )

    # Check for API key
    if not GOOGLE_API_KEY:
        st.warning("Google API key not found! Please make sure it's set in .env file or environment variables.")
        api_key = st.text_input("Enter your Google API key:", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
        else:
            return

    # Check dependencies
    if not check_dependencies():
        return

    # Initialize session state for model selection
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gemini-2.0-flash"

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a tool:", ["AI Blog Writer", "Research PDF Converter"])
    
    # Model selection in sidebar
    st.sidebar.subheader("Model Settings")
    selected_model = st.sidebar.selectbox(
        "Choose AI Model:",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.selected_model),
        help="Select the Gemini model to use for content generation. Different models may have varying capabilities and response times."
    )
    
    # Model information
    model_info = {
        "gemini-2.0-flash": "Fast, efficient model for quick responses",
        "gemini-1.5-flash": "Balanced performance and speed",
        "gemini-2.5-pro": "Most capable model, best for complex tasks",
        "gemini-2.5-flash": "Fast version of 2.5, good balance",
        "gemini-1.5-pro": "Pro version with enhanced capabilities"
    }
    
    st.sidebar.info(f"**Current Model:** {selected_model}\n\n{model_info.get(selected_model, 'Model information not available')}")
    
    # Update session state if model changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.blog_content = None
        st.session_state.content_analysis = None
        st.session_state.plagiarism_score = None
        st.session_state.current_outputs = None
        st.rerun()
    
    # Reset button in sidebar
    if st.sidebar.button("Reset App"):
        st.session_state.clear()
        st.rerun()

    # Display selected page
    if page == "AI Blog Writer":
        blog_writer_page()
    else:
        research_converter_page()

if __name__ == "__main__":
    main() 