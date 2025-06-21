import streamlit as st
from crewai import Crew, Task
from agents import BlogAgents
from PyPDF2 import PdfWriter
import tempfile
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import google.generativeai as genai
from dotenv import load_dotenv
import re

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

def clean_markdown(text):
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove any remaining markdown symbols
    text = re.sub(r'[#*_~`]', '', text)
    return text

st.set_page_config(page_title="AI Blog Writer", page_icon="✍️", layout="wide")

st.title("AI Blog Writer")
st.write("Generate high-quality blog posts using AI agents")

# Initialize session state
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

# Create columns for better layout
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
            # Initialize agents
            blog_agents = BlogAgents()
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
                expected_output="A detailed research report containing key information, facts, and data points about the topic."
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
                expected_output="An analyzed and processed version of the research data, optimized for the target audience and content goals."
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
            
            # Store results in session state
            st.session_state.blog_content = result
            st.session_state.content_analysis = None

    else:
        st.error("Please enter a topic to generate a blog post.")

# Display results
if st.session_state.blog_content:
    st.subheader("Generated Blog Post")
    st.write(st.session_state.blog_content)

    # Clean markdown from content
    cleaned_content = clean_markdown(st.session_state.blog_content)

    # Create HTML preview
    html_template = """
    <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }}
                h1 {{
                    color: #2c3e50;
                    font-size: 24px;
                    margin-bottom: 20px;
                }}
                p {{
                    margin-bottom: 15px;
                }}
                .section {{
                    margin-bottom: 25px;
                }}
            </style>
        </head>
        <body>
            <h1>{0}</h1>
            <div class="content">
                {1}
            </div>
        </body>
    </html>
    """
    
    html_content = html_template.format(
        topic,
        cleaned_content.replace('\n', '<br>')
    )

    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        f.write(html_content)
        html_path = f.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
        pdf_path = f.name

    # Generate PDF using reportlab
    try:
        # Create the PDF document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Create styles
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

        # Create content
        content = []
        content.append(Paragraph(topic, title_style))
        content.append(Spacer(1, 12))

        # Add the blog content
        for paragraph in cleaned_content.split('\n\n'):
            if paragraph.strip():
                content.append(Paragraph(paragraph, body_style))
                content.append(Spacer(1, 12))

        # Build the PDF
        doc.build(content)
        
        # Display preview and download options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("HTML Preview")
            st.components.v1.html(html_content, height=600, scrolling=True)
            
        with col2:
            st.subheader("Download Options")
            with open(html_path, 'rb') as f:
                st.download_button(
                    label="Download HTML",
                    data=f,
                    file_name=f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
            
            with open(pdf_path, 'rb') as f:
                st.download_button(
                    label="Download PDF",
                    data=f,
                    file_name=f"blog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
    
    finally:
        # Clean up temporary files
        os.unlink(html_path)
        os.unlink(pdf_path)

    # Add content analysis and plagiarism check buttons
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
                blog_agents = BlogAgents()
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
                    agent=plagiarism_agent
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