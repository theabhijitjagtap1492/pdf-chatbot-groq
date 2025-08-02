import fitz  # PyMuPDF
import streamlit as st
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile


def extract_text_from_pdfs(uploaded_files: List[UploadedFile]) -> str:
    """
    Extract text from multiple PDF files using PyMuPDF with enhanced processing.
    
    Args:
        uploaded_files: List of uploaded PDF files from Streamlit
        
    Returns:
        str: Combined text from all PDF files
    """
    combined_text = ""
    
    for uploaded_file in uploaded_files:
        try:
            # Read the PDF file
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            
            # Extract text from each page with better formatting
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Get text with better formatting
                text = page.get_text("text")
                
                # Clean up the text
                text = text.replace('\n\n\n', '\n\n')  # Remove excessive newlines
                text = text.replace('  ', ' ')  # Remove double spaces
                
                # Add page number for reference
                combined_text += f"--- Page {page_num + 1} ---\n{text}\n\n"
            
            # Close the document
            pdf_document.close()
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    return combined_text.strip()


def validate_pdf_files(uploaded_files: List[UploadedFile]) -> bool:
    """
    Validate that uploaded files are PDFs.
    
    Args:
        uploaded_files: List of uploaded files
        
    Returns:
        bool: True if all files are valid PDFs, False otherwise
    """
    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith('.pdf'):
            st.error(f"{uploaded_file.name} is not a PDF file.")
            return False
    return True 