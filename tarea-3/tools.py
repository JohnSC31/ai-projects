import faiss
import pickle
import numpy as np
from openai import OpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper

# --- RAG Tool (Inyección explícita) ---
def create_rag_tool(openai_api_key: str):
    """Factory para la herramienta RAG con clave inyectada."""
    class RAGTool:
        def __init__(self):
            self.client = OpenAI(api_key=openai_api_key)
            self.index = faiss.read_index("faiss_index_inner_product.index")
            with open("documents.pkl", "rb") as f:
                self.documents = pickle.load(f)

        def __call__(self, query: str, k: int = 3):
            # Generar embedding
            embedding = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            ).data[0].embedding
            
            # Buscar en FAISS
            distances, indices = self.index.search(
                np.array([embedding], dtype=np.float32), k
            )
            
            # Formatear resultados
            return [
                {
                    "text": self.documents[i].get_text(),
                    "metadata": self.documents[i].get_metadata(),
                    "distance": float(distances[0][j])
                }
                for j, i in enumerate(indices[0])
            ]

    return RAGTool()  # Devuelve instancia configurada

# --- Web Search Tool (Inyección explícita) ---
def create_web_search_tool(google_api_key: str, google_cse_id: str):
    """Factory para búsqueda web con credenciales inyectadas."""
    class WebSearchTool:
        def __init__(self):
            self.search = GoogleSearchAPIWrapper(
                google_api_key=google_api_key,
                google_cse_id=google_cse_id
            )

        def __call__(self, query: str, k: int = 3):
            results = self.search.results(query, k)
            return [{
                "content": f"{res['snippet']}\nFuente: {res['link']}",
                "metadata": {"source": res["link"]}
            } for res in results]

    return WebSearchTool() # Devuelve instancia configurada