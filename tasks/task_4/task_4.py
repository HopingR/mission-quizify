from langchain_google_vertexai import VertexAIEmbeddings

#from vertexai.language_models import TextEmbeddingModel

#model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003") 


class EmbeddingClient:
    """
    Initialize the EmbeddingClient class to connect to Google Cloud's VertexAI for text embeddings.
    """

    def __init__(self, model_name, project, location):
        """
        Initialize the EmbeddingClient with the specified model, project, and location.
        """
        print(f"Initializing VertexAIEmbeddings with model: {model_name}, project: {project}, location: {location}")
        
        try:
            self.client = VertexAIEmbeddings(
                model=model_name,
                project=project,
                location=location
            )
            print("Embedding client initialized successfully.")
        except Exception as e:
            print(f"Error initializing embedding client: {e}")
            self.client = None
        
    def embed_query(self, query):
        """
        Uses the embedding client to retrieve embeddings for the given query.
        """
        if self.client:
            try:
                vectors = self.client.embed_query(query)
                return vectors
            except Exception as e:
                print(f"Error embedding query: {e}")
                return None
        else:
            print("Client is not initialized.")
            return None
    
    def embed_documents(self, documents):
        """
        Retrieve embeddings for multiple documents.
        """
        if self.client:
            try:
                return self.client.embed_documents(documents)
            except AttributeError:
                print("Method embed_documents not defined for the client.")
                return None
        else:
            print("Client is not initialized.")
            return None

if __name__ == "__main__":
    model_name = "textembedding-gecko@003"
    project = "geminiquizify-422815"
    location = "us-central1"

    # Initialize the embedding client
    embedding_client = EmbeddingClient(model_name, project, location)
    
    # Check if client is initialized and use it
    if embedding_client.client:
        vectors = embedding_client.embed_query("Hello World!")
        if vectors:
            print(vectors)
            print("Successfully used the embedding client!")
    else:
        print("Failed to initialize the embedding client.")
