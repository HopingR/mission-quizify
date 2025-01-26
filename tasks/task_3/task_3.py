# pdf_processing.py

# Necessary imports
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import uuid

class DocumentProcessor:
    """
    This class processes uploaded PDF documents using Streamlit
    and Langchain's PyPDFLoader. It extracts pages from PDFs and
    displays the total number of pages processed.
    """
    def __init__(self):
        self.pages = []  # List to keep track of pages from all documents
    
    def ingest_documents(self):
        """
        Renders a file uploader widget in a Streamlit app, processes uploaded PDF files,
        extracts their pages using PyPDFLoader, and updates the self.pages list.
        """
        # Step 1: Render a file uploader widget to accept multiple PDFs
        uploaded_files = st.file_uploader(
            "Upload your PDF files",
            type='pdf',
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Generate a unique identifier for the temporary file
                unique_id = uuid.uuid4().hex
                original_name, file_extension = os.path.splitext(uploaded_file.name)
                temp_file_name = f"{original_name}_{unique_id}{file_extension}"
                temp_file_path = os.path.join(tempfile.gettempdir(), temp_file_name)

                # Write the uploaded file to the temporary location
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                try:
                    # Use PyPDFLoader to load and extract pages from the PDF
                    loader = PyPDFLoader(temp_file_path)
                    document_pages = loader.load()
                    self.pages.extend(document_pages)
                finally:
                    # Delete the temporary file after processing
                    os.unlink(temp_file_path)
            
            # Display the total number of pages processed
            st.write(f"Total pages processed: {len(self.pages)}")

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.ingest_documents()
