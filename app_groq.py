import streamlit as st
from pdf_utils import extract_text_from_pdfs, validate_pdf_files
from groq_utils import split_text, create_vector_store, get_qa_chain, answer_question, test_groq_connection
from config import GROQ_API_KEY

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot (Groq)",
    page_icon="📚",
    layout="wide"
)

# Large font styling for better user experience
st.markdown("""
<style>
    /* Increase font sizes for better readability */
    .stMarkdown, .stText {
        font-size: 18px !important;
    }
    
    .stMarkdown h1 {
        font-size: 2.5rem !important;
        font-weight: bold !important;
    }
    
    .stMarkdown h2, .stMarkdown h3 {
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    
    .stMarkdown p {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Larger buttons */
    .stButton > button {
        font-size: 1.2rem !important;
        padding: 0.8rem 1.5rem !important;
        height: 50px !important;
    }
    
    /* Larger text inputs */
    .stTextInput > div > div > input {
        font-size: 1.1rem !important;
        padding: 0.8rem !important;
        height: 50px !important;
    }
    
    /* Larger file uploader */
    .uploadedFile {
        font-size: 1.1rem !important;
        padding: 1rem !important;
    }
    
    /* Larger expander */
    .streamlit-expanderHeader {
        font-size: 1.3rem !important;
        font-weight: bold !important;
    }
    
    /* Larger success/error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        font-size: 1.1rem !important;
        padding: 1rem !important;
    }
    
    /* Larger captions */
    .stMarkdown small {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Simple header without any custom styling
    st.title("📚 Smart PDF Assistant")
    st.caption("Powered by Groq AI")
    
    # Quick Start Guide - collapsed by default
    with st.expander("📖 Quick Start Guide", expanded=False):
        st.markdown("""
        **Three Steps:** 1. Upload PDFs 2. Process 3. Ask Questions
        **Tips:** Upload multiple PDFs, ask specific questions
        """)
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload section
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✨ {len(uploaded_files)} file(s) ready!")
            
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing..."):
                    try:
                        combined_text = extract_text_from_pdfs(uploaded_files)
                        if combined_text.strip():
                            documents = split_text(combined_text)
                            vector_store = create_vector_store(documents, GROQ_API_KEY)
                            st.session_state.qa_chain = get_qa_chain(vector_store, GROQ_API_KEY)
                            st.session_state.pdfs_processed = True
                            st.success("✅ Ready to ask questions!")
                        else:
                            st.error("❌ Could not extract text from PDFs")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        # Instructions
        st.subheader("💡 How It Works")
        st.markdown("""
        1. **Upload** PDF files
        2. **Process** with AI
        3. **Ask** questions
        """)
    
    # Chat section - only show if files are processed
    if uploaded_files and st.session_state.get('pdfs_processed', False):
        st.subheader("💬 Ask Questions")
        
        # Question input and button in same row
        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "What would you like to know?",
                placeholder="Ask about your documents..."
            )
        with col2:
            ask_button = st.button("🔍 Ask", type="primary")
        
        if question and ask_button:
            with st.spinner("Searching..."):
                try:
                    answer, is_found_in_docs = answer_question(st.session_state.qa_chain, question)
                    
                    if is_found_in_docs:
                        st.markdown("**📝 Answer:**")
                        st.write(answer)
                        st.success("✅ Found in documents")
                    else:
                        st.warning("⚠️ Not found in documents")
                        st.info("Try rephrasing your question")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Simple footer
    st.divider()
    st.caption("Built with Streamlit, LangChain, and Groq AI")

if __name__ == "__main__":
    main()