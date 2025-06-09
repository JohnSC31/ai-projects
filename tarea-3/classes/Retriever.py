from openai import OpenAI
from dotenv import load_dotenv
import faiss
import pickle
import numpy as np
import os
import sys




class Retriever:
    def __init__(self, faiss_index_path, documents_path, openai_api_key=None):
        if openai_api_key:
            self.openai = OpenAI(api_key=openai_api_key)
        self.faiss_index = faiss.read_index(faiss_index_path)
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)

    def __call__(self, query, k=6):
        # Generate embeddings for the query
        query_embedding = self.openai.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
        query_embedding = np.array(query_embedding).astype('float32')

        # Search the FAISS index
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)

        # Retrieve documents based on indices
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs, distances[0]

if __name__ == "__main__":
    sys.path.append("./")
    from classes.Document import Document
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Create a Retriever instance
    retriever = Retriever('faiss_index_inner_product.index', 'documents.pkl', openai_api_key=openai_api_key)
    query = input("Enter your query: ")
    retrieved_docs, distances = retriever(query)
    print("Retrieved Documents:")
    for doc, dist in zip(retrieved_docs, distances):
        print(f"Document: {doc.filename}, Distance: {dist:.4f}")
        print(f"Text: {doc.get_text()}\n")