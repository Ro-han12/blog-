from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ResearchTools:
    """Collection of tools for research and content creation."""
    
    @staticmethod
    def create_pdf_search_tool(pdf_path=None, gemini_api_key=None):
        """Create a PDF search tool using simple text splitting and LLM.
        
        Args:
            pdf_path: Path to the PDF file
            gemini_api_key: Google API key for Gemini
            
        Returns:
            A callable tool for searching PDF content
        """
        if not pdf_path:
            raise ValueError("PDF path is required")

        # Load and split the PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Store the full text and page numbers
        full_text = ""
        page_texts = {}
        
        for page in pages:
            page_number = page.metadata.get('page', 0) + 1  # 1-based page numbers
            text = page.page_content.strip()
            if text:  # Only add non-empty pages
                page_texts[page_number] = text
                full_text += f"\n\n=== Page {page_number} ===\n{text}"
        
        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key,
            temperature=0.1  # Lower temperature for more precise extraction
        )

        # Create search prompt
        search_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            You are extracting content EXACTLY as it appears in a research document. 
            Your task is to be a precise copier, not an interpreter.

            Context from document:
            {context}

            Query: {query}

            CRITICAL RULES:
            1. ONLY return text that appears verbatim in the context
            2. DO NOT add any information not present in the context
            3. DO NOT modify or paraphrase any content
            4. DO NOT reorganize or restructure the content
            5. Maintain exact quotes, numbers, and technical terms
            6. If information is not in the context, say "NOT FOUND IN THIS SECTION"

            Copy the relevant content exactly as it appears, maintaining original:
            - Wording and phrasing
            - Numbers and statistics
            - Technical terminology
            - Citations and references
            - Section structure and order
            """
        )

        # Create chain
        chain = LLMChain(llm=llm, prompt=search_prompt)

        def search(query):
            try:
                # For general queries about the whole document
                if query.lower().startswith(("extract all", "get all", "find all")):
                    return full_text
                
                # For specific section queries
                result = chain.run(query=query, context=full_text)
                if result and not result.lower().startswith(("not found", "i cannot", "i am unable")):
                    return result
                
                # If no results, try searching page by page
                results = []
                for page_num, page_text in page_texts.items():
                    page_result = chain.run(
                        query=f"{query} (on page {page_num})", 
                        context=page_text
                    )
                    if page_result and not page_result.lower().startswith(("not found", "i cannot", "i am unable")):
                        results.append(f"[Page {page_num}]: {page_result}")
                
                if results:
                    return "\n\n".join(results)
                else:
                    return "The requested information was not found in the document."
                    
            except Exception as e:
                return f"Error processing PDF: {str(e)}"

        return search
    
    @staticmethod
    def create_gemini_llm(api_key, temperature=0.3):
        """Create a Gemini LLM instance.
        
        Args:
            api_key: Google API key
            temperature: Temperature setting for generation
            
        Returns:
            ChatGoogleGenerativeAI instance
        """
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=temperature
        ) 