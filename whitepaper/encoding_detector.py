import streamlit as st
import chardet
import PyPDF2
import io

def detect_encoding(text_bytes):
    """Detect the encoding of the given bytes."""
    result = chardet.detect(text_bytes)
    return result

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file with encoding detection."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_chunks = []
        encodings = []
        
        for page in pdf_reader.pages:
            # Extract text from each page
            page_text = page.extract_text()
            if page_text:
                # Convert to bytes for encoding detection
                text_bytes = page_text.encode('utf-8', errors='ignore')
                # Detect encoding for this chunk
                encoding_result = detect_encoding(text_bytes)
                
                text_chunks.append({
                    'text': page_text,
                    'bytes': text_bytes,
                    'encoding': encoding_result
                })
                encodings.append(encoding_result['encoding'] if encoding_result else 'utf-8')
        
        return text_chunks, encodings
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None, None

def process_for_research(text_chunks, encodings):
    """Process text chunks for research agents."""
    processed_text = ""
    
    for i, chunk in enumerate(text_chunks):
        try:
            # Try to decode with detected encoding, fallback to utf-8
            encoding = encodings[i] if encodings[i] else 'utf-8'
            decoded_text = chunk['bytes'].decode(encoding, errors='replace')
            
            # Add page markers for better context
            processed_text += f"\n=== Page {i+1} ===\n{decoded_text}\n"
            
        except Exception as e:
            st.warning(f"Warning: Could not process page {i+1} with encoding {encoding}. Using replacement characters.")
            # Use a safe fallback
            processed_text += f"\n=== Page {i+1} ===\n{chunk['text']}\n"
    
    return processed_text

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

def main():
    st.title("PDF Encoding Detector and Processor")
    st.write("Upload a PDF file to detect its text encoding and process for research")

    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        # Read the PDF file
        pdf_bytes = uploaded_file.read()
        
        # Extract text from PDF with encoding detection
        text_chunks, encodings = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        
        if text_chunks and encodings:
            # Process for research
            processed_text = process_for_research(text_chunks, encodings)
            
            # Display encoding information for each chunk
            st.subheader("Encoding Detection Results")
            for i, chunk in enumerate(text_chunks):
                with st.expander(f"Page {i+1} Encoding Details"):
                    st.json(chunk['encoding'])
                    patterns = show_encoding_details(chunk['bytes'], chunk['encoding'])
            
            # Preview processed text
            st.subheader("Processed Text Preview")
            st.text_area("First 500 characters", processed_text[:500], height=200)
                
            # Store processed text in session state for research agents
            if 'processed_pdf_text' not in st.session_state:
                st.session_state.processed_pdf_text = {}
            
            st.session_state.processed_pdf_text[uploaded_file.name] = processed_text
            
            # Add research processing button
            if st.button("Process with Research Agents"):
                try:
                    from whitepaper.tools import ResearchTools
                    from whitepaper.agents import ResearchAgents
                    import os
                    
                    # Get API key from environment
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        st.warning("Google API key not found. Please set GOOGLE_API_KEY in your environment.")
                        return
                    
                    # Initialize LLM and tools
                    llm = ResearchTools.create_gemini_llm(api_key)
                    
                    # Create a custom search function for the processed text
                    def pdf_search(query):
                        """Search function for the processed text."""
                        return processed_text
                    
                    # Create research agent
                    researcher = ResearchAgents.create_researcher(llm, pdf_search)
                    
                    with st.spinner("Analyzing document..."):
                        # Execute research task
                        analysis = researcher.execute_task(
                            "Analyze this document and provide a detailed summary of its content, "
                            "structure, and key findings. Include any technical terms or important "
                            "data points exactly as they appear in the text."
                        )
                        
                        st.subheader("Document Analysis")
                        st.write(analysis)
                        
                except Exception as e:
                    st.error(f"Error during research processing: {str(e)}")
                    st.error("Please check your API key and try again.")

if __name__ == "__main__":
    main() 