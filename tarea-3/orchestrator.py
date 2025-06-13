from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from tools import create_rag_tool, create_web_search_tool
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
google_key = os.getenv("GOOGLE_API_KEY")
google_cse = os.getenv("GOOGLE_CSE_ID")


rag_tool = create_rag_tool(openai_api_key=openai_api_key)
web_search_tool = create_web_search_tool(
    google_api_key=google_key,
    google_cse_id=google_cse
)

tools = [
    Tool(
        name="RAGTool",
        func=rag_tool,
        description="Usa esta herramienta para responder preguntas basadas en los apuntes del curso. Úsala por defecto."
    ),
    Tool(
        name="WebSearchTool",
        func=web_search_tool,
        description="Usa esta herramienta si el usuario explícitamente pide buscar en internet o menciona fuentes externas."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_prompt = """
Eres un asistente conversacional experto en Inteligencia Artificial. 
Tu trabajo es ayudar a los estudiantes a responder preguntas basadas en sus apuntes del curso del primer semestre 2025. 

- Usa la herramienta RAG para buscar en los apuntes.
- Usa la herramienta de búsqueda en internet **solo si el usuario lo solicita explícitamente** con frases como "busca en internet", "verifica en la web", etc.
- Mantén el contexto de las preguntas anteriores para responder con coherencia.
- Sé claro, conciso y evita responder "no sé" si hay información en los apuntes.

Comienza ahora.
"""

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

agent_response = ""

st.title("Agente de preguntas - Notas IA 2025")
user_query = st.text_input("Ingresa la pregunta:")

if "internet" in user_query.lower():
    agent_response = agent.run("Usa WebSearchTool: " + user_query)
else:
    agent_response = agent.run(user_query)

st.write("Respuesta:")
st.write(agent_response)