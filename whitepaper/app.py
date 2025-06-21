import os
import streamlit as st
import tempfile
import base64
from pathlib import Path
import traceback
import uuid
import shutil
import time

from main import ResearchConverter

st.set_page_config(
    page_title="Research PDF Converter",
    page_icon="📚",
    layout="wide"
)

# Simple version marker to help track app state
APP_VERSION = "1.0.1"

def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate HTML code for file download link"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def display_pdf(pdf_path):
    """Display a PDF file in Streamlit"""
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_html(html_path):
    """Display HTML content in Streamlit"""
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)

def check_dependencies():
    """Check if all required packages are installed"""
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
    """Safely ensure the exports directory exists"""
    export_dir = Path("exports")
    if not export_dir.exists():
        export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir

def main():
    st.title("Research PDF Converter")
    st.markdown("Upload a research PDF to convert it into structured, plain content")
    
    # Simple reset button
    if st.sidebar.button("Reset App"):
        st.session_state.clear()
        st.experimental_rerun()
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.current_outputs = None
    
    # Ensure the exports directory exists
    ensure_export_dir()
    
    if not check_dependencies():
        return
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("Google API key not found! Please make sure it's set in .env file or environment variables.")
        api_key = st.text_input("Enter your Google API key:", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            return
    
    # Simple file uploaders
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Brand guidelines uploader (optional)
    include_brand = st.checkbox("Include brand guidelines")
    uploaded_brand = None
    if include_brand:
        uploaded_brand = st.file_uploader("Choose brand guidelines PDF (optional)", type="pdf")
    
    output_format = st.radio("Output format", ["pdf", "html", "both"], index=2)
    
    # Only show the process button if a file is uploaded
    if uploaded_file is not None:
        process_button = st.button("Process PDF")
        
        if process_button:
            try:
                # Create temp directory for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir_path = Path(temp_dir)
                    
                    # Save uploaded file to temp directory
                    original_filename = uploaded_file.name
                    temp_pdf_path = temp_dir_path / original_filename
                    with open(temp_pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Save brand guidelines if uploaded
                    temp_brand_path = None
                    if include_brand and uploaded_brand is not None:
                        brand_filename = uploaded_brand.name
                        temp_brand_path = temp_dir_path / brand_filename
                        with open(temp_brand_path, "wb") as f:
                            f.write(uploaded_brand.getvalue())
                    
                    # Create output directory
                    unique_id = f"{int(time.time())}"
                    output_dir = temp_dir_path / f"output_{unique_id}"
                    output_dir.mkdir(exist_ok=True)
                    
                    with st.spinner(f"Processing '{original_filename}'... This may take a few minutes."):
                        # Process PDF
                        converter = ResearchConverter(gemini_api_key=api_key, output_dir=str(output_dir))
                        outputs = converter.process_pdf(
                            pdf_path=str(temp_pdf_path),
                            brand_guidelines=str(temp_brand_path) if temp_brand_path else None,
                            output_format=output_format
                        )
                        
                        if outputs:
                            # Copy outputs to exports folder
                            export_dir = ensure_export_dir() / unique_id
                            export_dir.mkdir(exist_ok=True)
                            
                            permanent_outputs = []
                            for output_path in outputs:
                                new_path = export_dir / Path(output_path).name
                                shutil.copy2(output_path, new_path)
                                permanent_outputs.append(new_path)
                            
                            # Store in session state
                            st.session_state.current_outputs = permanent_outputs
                            st.success(f"Successfully processed: {original_filename}")
                        else:
                            st.error("No outputs were generated during conversion.")
            
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.error(traceback.format_exc())
    
    # Display outputs if they exist in session state
    if st.session_state.get('current_outputs'):
        # Display outputs in tabs
        tabs = st.tabs(["PDF Output", "HTML Output"])
        
        # PDF tab
        with tabs[0]:
            pdf_output = [p for p in st.session_state.current_outputs if str(p).endswith('.pdf')]
            if pdf_output:
                st.subheader("PDF Preview")
                display_pdf(pdf_output[0])
                filename = uploaded_file.name if uploaded_file else "output"
                st.markdown(get_binary_file_downloader_html(pdf_output[0], f"PDF - {filename}"), unsafe_allow_html=True)
            else:
                st.warning("PDF output not available.")
        
        # HTML tab
        with tabs[1]:
            html_output = [p for p in st.session_state.current_outputs if str(p).endswith('.html')]
            if html_output:
                st.subheader("HTML Preview")
                display_html(html_output[0])
                filename = uploaded_file.name if uploaded_file else "output"
                st.markdown(get_binary_file_downloader_html(html_output[0], f"HTML - {filename}"), unsafe_allow_html=True)
            else:
                st.warning("HTML output not available.")

if __name__ == "__main__":
    main() 