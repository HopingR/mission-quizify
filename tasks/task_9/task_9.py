import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator
from tasks.task_8.task_8 import QuizGenerator

class QuizManager:
    ##########################################################
    def __init__(self, questions: list):
        """
        Initialize the QuizManager class with a list of quiz questions.
        """
        self.questions = questions
        self.total_questions = len(questions)
    ##########################################################

    def get_question_at_index(self, index: int):
        """
        Retrieves the quiz question object at the specified index. If the index is out of bounds,
        it restarts from the beginning index.
        """
        valid_index = index % self.total_questions
        return self.questions[valid_index]

    ##########################################################
    def next_question_index(self, direction=1):
        """
        Adjust the current quiz question index based on the specified direction.
        """
        if "question_index" not in st.session_state:
            st.session_state["question_index"] = 0
        
        current_index = st.session_state["question_index"]
        new_index = (current_index + direction) % self.total_questions
        st.session_state["question_index"] = new_index
    ##########################################################

# Test Generating the Quiz
if __name__ == "__main__":
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "geminiquizify-422815",
        "location": "us-central1"
    }
    
    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()
    
        embed_client = EmbeddingClient(**embed_config) 
    
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
    
        question = None
        question_bank = None
    
        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                
                st.write(topic_input)
                
                # Test the Quiz Generator
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question_bank = generator.generate_quiz()

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            
            # Task 9
            ##########################################################
            quiz_manager = QuizManager(question_bank)
            
            # Get the first question using the get_question_at_index method
            index_question = quiz_manager.get_question_at_index(0)
            
            with st.form("Multiple Choice Question"):
                # Unpack choices for radio
                choices = []
                for key, value in index_question['choices'].items():
                    choices.append(f"{key}) {value}")
                
                # Display the question onto streamlit
                st.write(f"**{index_question['question']}**")
                
                answer = st.radio( 
                    'Choose the correct answer',
                    choices
                )
                submitted = st.form_submit_button("Submit")
                
                if submitted: 
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
            ##########################################################
