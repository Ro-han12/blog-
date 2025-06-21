from crewai import Agent
from textwrap import dedent
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class BlogAgents:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
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