from crewai import Agent
from whitepaper.tools import ResearchTools
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

class ResearchAgents:
    """Factory for creating agents used in the research conversion process."""
    
    @staticmethod
    def create_researcher(llm, pdf_tool):
        """Create an agent for researching and extracting insights.
        
        Args:
            llm: Language model to use
            pdf_tool: Tool for searching PDFs
            
        Returns:
            Configured Agent instance
        """
        tools = [
            Tool(
                name="SearchPDF",
                func=pdf_tool,
                description="""Search and extract information from the PDF document. 
                The document is pre-processed with encoding detection and contains page markers.
                Use 'extract all' to get the complete document content."""
            )
        ]
        
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        return Agent(
            role='Research Analyst',
            goal='Extract and analyze key information from research documents',
            backstory="""You are an expert research analyst with years of experience in 
            analyzing academic and technical documents. Your strength lies in identifying 
            key findings, methodologies, and insights from complex research materials.
            You understand the importance of handling different text encodings and can
            work with documents that may contain special characters or technical notation.""",
            llm=llm,
            tools=tools,
            memory=memory,
            allow_delegation=False,
            verbose=True
        )
    
    @staticmethod
    def create_content_creator(llm):
        """Create an agent for transforming research into content.
        
        Args:
            llm: Language model to use
            
        Returns:
            Configured Agent instance
        """
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        return Agent(
            role='Content Creator',
            goal='Transform research insights into engaging, accessible content',
            backstory="""You are a skilled content creator specializing in making complex 
            research accessible to broader audiences. You excel at maintaining scientific 
            accuracy while making content engaging and understandable.""",
            llm=llm,
            memory=memory,
            allow_delegation=False,
            verbose=True
        )
    
    @staticmethod
    def create_formatter(llm):
        """Create an agent for formatting content.
        
        Args:
            llm: Language model to use
            
        Returns:
            Configured Agent instance
        """
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        return Agent(
            role='Content Formatter',
            goal='Format content for clean, professional presentation',
            backstory="""You are a detail-oriented content formatter with expertise in 
            creating clean, well-structured documents. You ensure content is organized 
            logically and presented professionally.""",
            llm=llm,
            memory=memory,
            allow_delegation=False,
            verbose=True
        ) 