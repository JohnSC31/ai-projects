# tools.py
import pickle
import numpy as np
import faiss
from openai import OpenAI
from classes.Document import Document

class RAGTool:
    def __init__(self, faiss_index_path="faiss_index_inner_product.index", documents_path="documents.pkl"):
        self.index = faiss.read_index(faiss_index_path)
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        self.client = OpenAI() 

    def __call__(self, query: str, k: int = 3):
        # Generar embedding de la consulta
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Buscar en FAISS
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        # Recuperar documentos relevantes
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc = self.documents[idx]
            results.append({
                "text": doc.get_text(),
                "metadata": doc.get_metadata(),
                "distance": float(dist)
            })
        return results

# Instancia global para usar en el agente
rag_tool_function = RAGTool()