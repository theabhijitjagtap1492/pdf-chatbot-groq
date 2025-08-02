# PDF Chatbot with Groq AI

A smart PDF assistant that allows you to upload PDF documents and ask questions about their content using Groq AI.

## Features

- 📤 Upload multiple PDF files
- 🤖 AI-powered document processing
- 💬 Interactive Q&A interface
- 🔍 Smart search through document content
- 📱 Responsive design

## Tech Stack

- **Frontend**: Streamlit
- **AI**: Groq API
- **Vector Database**: FAISS
- **PDF Processing**: PyMuPDF
- **Text Processing**: LangChain

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Groq API key in `config.py`

3. Run the application:
```bash
streamlit run app_groq.py
```

## Deployment

This app is deployed on Streamlit Cloud. Visit the live app at: [Your Streamlit Cloud URL]

## Environment Variables

- `GROQ_API_KEY`: Your Groq API key (set in Streamlit Cloud secrets)

## Usage

1. Upload one or more PDF files
2. Click "Process Documents" to analyze the content
3. Ask questions about your documents
4. Get AI-powered answers based on the document content 