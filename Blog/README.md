# AI Blog Writer

An intelligent blog writing application that uses CrewAI, Streamlit, and Google's Gemini API to generate high-quality blog posts.

## Features

- Research and gather information about any topic
- Process and analyze content using NLP techniques
- Generate well-structured blog posts
- Check for plagiarism
- Preview blog posts in HTML format
- Download blog posts as HTML or PDF

## Prerequisites

- Python 3.8 or higher
- Google API key for Gemini
- wkhtmltopdf (for PDF generation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install wkhtmltopdf:
- For macOS:
```bash
brew install wkhtmltopdf
```


4. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter your blog topic or search query in the text area

4. Click "Generate Blog" to start the blog generation process

5. Once generated, you can:
   - Preview the blog post in HTML format
   - Download the blog post as HTML or PDF
   - View the plagiarism report

## Project Structure

- `app.py`: Main Streamlit application
- `agents.py`: CrewAI agents definitions
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (create this file)

 