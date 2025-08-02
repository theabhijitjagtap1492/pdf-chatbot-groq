import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
import streamlit as st
import re


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """
    Split a long text into smaller chunks using RecursiveCharacterTextSplitter for better accuracy.
    
    Args:
        text: The text to split
        chunk_size: Size of each chunk (smaller for better precision)
        chunk_overlap: Overlap between chunks (increased for better context)
        
    Returns:
        List[Document]: List of text chunks as Document objects
    """
    # Clean the text first
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk.strip()) for chunk in chunks if chunk.strip()]
    
    return documents


def create_vector_store(documents: List[Document], api_key: str):
    """
    Create an improved vector store from document chunks using enhanced TF-IDF.
    
    Args:
        documents: List of Document objects
        api_key: Groq API key (not used for embeddings)
        
    Returns:
        SimpleVectorStore: Enhanced vector store for similarity search
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    # Enhanced TF-IDF with better parameters
    texts = [doc.page_content for doc in documents]
    vectorizer = TfidfVectorizer(
        max_features=2000,  # Increased features
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=1,
        max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Create a simple FAISS index
    import faiss
    dimension = tfidf_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Convert to dense matrix and normalize for cosine similarity
    dense_matrix = tfidf_matrix.toarray()
    # Normalize for cosine similarity
    norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
    normalized_matrix = dense_matrix / np.where(norms > 0, norms, 1)
    
    index.add(normalized_matrix.astype('float32'))
    
    # Create an enhanced vector store
    class EnhancedVectorStore:
        def __init__(self, index, documents, vectorizer):
            self.index = index
            self.documents = documents
            self.vectorizer = vectorizer
        
        def similarity_search(self, query, k=5):  # Increased k for better retrieval
            # Clean and preprocess query
            query = re.sub(r'\s+', ' ', query.strip())
            
            # Vectorize the query
            query_vector = self.vectorizer.transform([query]).toarray()
            query_norm = np.linalg.norm(query_vector)
            if query_norm > 0:
                query_vector = query_vector / query_norm
            
            # Search with more results
            D, I = self.index.search(query_vector.astype('float32'), k)
            
            # Return documents with scores
            results = []
            for i, (score, idx) in enumerate(zip(D[0], I[0])):
                if idx < len(self.documents) and score > 0.1:  # Filter by similarity threshold
                    results.append(self.documents[idx])
            
            return results[:3]  # Return top 3 most relevant
    
    return EnhancedVectorStore(index, documents, vectorizer)


def get_qa_chain(vector_store, api_key: str):
    """
    Create an enhanced QA chain using ChatGroq with better prompting.
    
    Args:
        vector_store: Vector store
        api_key: Groq API key
        
    Returns:
        EnhancedQAChat: Improved QA chain for answering questions
    """
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1  # Slight temperature for better creativity while maintaining accuracy
    )
    
    # Create an enhanced retriever
    class EnhancedRetriever:
        def __init__(self, vector_store):
            self.vector_store = vector_store
        
        def get_relevant_documents(self, query):
            return self.vector_store.similarity_search(query, k=5)
    
    retriever = EnhancedRetriever(vector_store)
    
    # Create an enhanced QA chain
    class EnhancedQAChat:
        def __init__(self, llm, retriever):
            self.llm = llm
            self.retriever = retriever
        
        def __call__(self, inputs):
            query = inputs["query"]
            
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return {
                    "result": "I cannot find any relevant information in the provided documents to answer your question.",
                    "source_documents": []
                }
            
            # Create enhanced context
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"Document {i}:\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Enhanced prompt for better accuracy
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided documents. 

IMPORTANT INSTRUCTIONS:
1. Only answer based on the information provided in the documents
2. If the answer is not clearly found in the documents, say "I cannot find a specific answer to this question in the provided documents."
3. Be precise and accurate in your response
4. If you're unsure about any part of the answer, mention it
5. Use the exact information from the documents when possible

DOCUMENTS:
{context}

QUESTION: {query}

Please provide a clear and accurate answer based only on the information in the documents above:"""
            
            # Get response from Groq
            response = self.llm.invoke(prompt)
            
            return {
                "result": response.content,
                "source_documents": docs
            }
    
    return EnhancedQAChat(llm, retriever)


def answer_question(qa_chain, question: str) -> tuple:
    """
    Answer a question using the enhanced QA chain and check if answer is found in documents.
    
    Args:
        qa_chain: QA chain
        question: User's question
        
    Returns:
        tuple: (answer, is_found_in_docs)
    """
    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        source_documents = result["source_documents"]
        
        # Enhanced check for relevant source documents
        is_found_in_docs = (
            len(source_documents) > 0 and 
            any(doc.page_content.strip() for doc in source_documents) and
            "cannot find" not in answer.lower() and
            "not found" not in answer.lower()
        )
        
        return answer, is_found_in_docs
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return "An error occurred while processing your question.", False


def test_groq_connection(api_key: str) -> bool:
    """
    Test if Groq API key is valid and working.
    
    Args:
        api_key: Groq API key
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant"
        )
        response = llm.invoke("Hello, this is a test.")
        return True
    except Exception as e:
        st.error(f"Groq connection test failed: {str(e)}")
        return False 