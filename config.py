"""
Configuration file for the PDF Chatbot
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API Key - get from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")