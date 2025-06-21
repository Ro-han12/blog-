# AI Content Tools

A Streamlit application that provides AI-powered content creation tools including blog writing and research PDF conversion.

## Features

- **AI Blog Writer**: Generate high-quality blog posts using AI agents
- **Research PDF Converter**: Convert research PDFs into structured, readable content

## Deployment to Streamlit Cloud

### Prerequisites

1. A Google Gemini API key
2. A GitHub repository with your code

### Setup Instructions

1. **Fork or clone this repository**

2. **Set up your Google Gemini API key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the API key

3. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to `app.py`
   - Add your Google API key as a secret:
     - Go to "Secrets" in your app settings
     - Add: `GOOGLE_API_KEY = "your-api-key-here"`

4. **Deploy!**

### Environment Variables

The app requires the following environment variable:
- `GOOGLE_API_KEY`: Your Google Gemini API key

### File Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── exports/              # Generated output files
└── README.md            # This file
```

### Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API key: `GOOGLE_API_KEY=your-key-here`
4. Run the app: `streamlit run app.py`

## Usage

### AI Blog Writer
1. Enter your blog topic
2. Customize audience, tone, industry, and other parameters
3. Click "Generate Blog" to create content
4. Download as HTML or PDF

### Research PDF Converter
1. Upload a research PDF
2. Choose output format (PDF, HTML, or both)
3. Optionally enable translation to English
4. Process the document
5. Download the converted content

## Troubleshooting

- **Import errors**: All dependencies are now inline in `app.py` to avoid import issues on Streamlit Cloud
- **API key issues**: Make sure your Google API key is properly set in Streamlit Cloud secrets
- **File upload issues**: The app supports PDF files up to 200MB

## License

This project is open source and available under the MIT License. 