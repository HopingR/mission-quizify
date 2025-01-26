import sys
import os
import streamlit as st

"""
Task: Build a Quiz Builder with Streamlit and LangChain

Overview:
In this task, you will leverage your skills acquired from previous tasks to create a "Quiz Builder" application utilizing Streamlit. This interactive application enables users to upload documents, designate a quiz topic, select a number of questions, and subsequently generate a quiz based on the uploaded document contents.

Components to Integrate:
- DocumentProcessor: A class developed in Task 3 for processing uploaded PDF documents.
- EmbeddingClient: A class from Task 4 dedicated to embedding queries.
- ChromaCollectionCreator: A class from Task 5 responsible for creating a Chroma collection derived from the processed documents.

Step-by-Step Instructions:
1. Begin by initializing an instance of the `DocumentProcessor` and invoke the `ingest_documents()` method to process the uploaded PDF documents.

2. Configure and initialize the `EmbeddingClient` with the specified model, project, and location details as provided in the `embed_config`.

3. Instantiate the `ChromaCollectionCreator` using the previously initialized `DocumentProcessor` and `EmbeddingClient`.

4. Utilize Streamlit to construct a form. This form should prompt users to input the quiz's topic and select the desired number of questions via a slider component.

5. Following the form submission, employ the `ChromaCollectionCreator` to forge a Chroma collection from the documents processed earlier.

6. Enable users to input a query pertinent to the quiz topic. Utilize the generated Chroma collection to extract relevant information corresponding to the query, which aids in quiz question generation.

Implementation Guidance:
- Deploy Streamlit's widgets such as `st.header`, `st.subheader`, `st.text_input`, and `st.slider` to craft an engaging form. This form should accurately capture the user's inputs for both the quiz topic and the number of questions desired.

- Post form submission, verify that the documents have been processed and that a Chroma collection has been successfully created. The build-in methods will communicate the outcome of these operations to the user through appropriate feedback.

- Lastly, introduce a query input field post-Chroma collection creation. This field will gather user queries for generating quiz questions, showcasing the utility of the Chroma collection in sourcing information relevant to the quiz topic.
"""

# Add the parent directory to the system path
sys.path.append(os.path.abspath('../../'))

# Import necessary classes from previous tasks
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator

if __name__ == "__main__":
    st.header("Quizzify: Build a Quiz from Documents")

    # Configuration for EmbeddingClient
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "geminiquizify-422815",
        "location": "us-central1"
    }

    # Initialize components
    document_processor = DocumentProcessor()
    embedding_client = EmbeddingClient(
        model_name=embed_config["model_name"],
        project=embed_config["project"],
        location=embed_config["location"]
    )
    chroma_creator = ChromaCollectionCreator(document_processor, embedding_client)

    # Create a section for document ingestion and quiz setup
    with st.form("Load Data to Chroma"):
        st.subheader("Quiz Builder")
        st.write("Upload PDFs, specify a quiz topic, and generate a quiz!")

        # File uploader for documents
        uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

        # Input for quiz topic
        quiz_topic = st.text_input("Enter Quiz Topic")

        # Slider for the number of questions
        num_questions = st.slider("Select Number of Questions", min_value=1, max_value=20, value=5)

        # Submit button
        submitted = st.form_submit_button("Generate a Quiz!")

        # Process and generate Chroma collection upon submission
        if submitted:
            if uploaded_files and quiz_topic:
                try:
                    # Ingest uploaded documents
                    st.write("Processing documents...")
                    document_processor.ingest_documents(uploaded_files)  # Pass uploaded files

                    # Create Chroma collection
                    st.write("Creating Chroma collection...")
                    chroma_creator.create_chroma_collection()

                    # Provide feedback to the user
                    st.success("Chroma collection created successfully! You can now query the collection.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload documents and specify a quiz topic.")

    # Section for querying Chroma collection
    with st.container():
        st.subheader("Query Chroma Collection")
        query = st.text_input("Enter a Query Related to the Quiz Topic")

        if query:
            try:
                # Query the Chroma collection
                st.write("Fetching relevant information...")
                document = chroma_creator.query_chroma_collection(query)
                st.write("Top Document Results:")
                st.write(document)
            except Exception as e:
                st.error(f"An error occurred while querying: {e}")
