import streamlit as st
import os
import sqlite3
import hashlib
import tempfile
from typing import List, Tuple

# ---------- LLM & ADVANCED ORCHESTRATION ----------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Runtime API Key Enforcement
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---------- PAGE STATE CONFIGURATION ----------
st.set_page_config(
    page_title="🧬 Enterprise AI Research Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- THREAD-SAFE DATABASE INTERFACE ----------
DB_PATH = "research_core_v4.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            role TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            username TEXT PRIMARY KEY,
            name TEXT,
            role_title TEXT,
            institution TEXT,
            biography TEXT,
            research_interests TEXT,
            technical_skills TEXT,
            publications_projects TEXT
        )
        """)
        conn.commit()

init_db()

# ---------- SECURITY ENGINE ----------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def signup(u: str, p: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, hash_password(p)))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False

def login(u: str, p: str) -> Tuple:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_password(p)))
        return cursor.fetchone()

# ---------- SESSION STATE INITIALIZATION ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "vector" not in st.session_state:
    st.session_state.vector = None

# ---------- IDENTITY VERIFICATION GATEWAY ----------
if st.session_state.user is None:
    st.title("🔐 Core Research Gateway Access")
    tab1, tab2 = st.tabs(["🔒 Secure Authentication", "📝 System Registration"])

    with tab1:
        u = st.text_input("Credential Identifier (Username)", key="auth_u")
        p = st.text_input("Access Token (Password)", type="password", key="auth_p")
        if st.button("Initialize Session", use_container_width=True):
            res = login(u, p)
            if res:
                st.session_state.user = u
                st.success("Authorization granted. Mounting workspace...")
                st.rerun()
            else:
                st.error("Access Denied: Invalid credentials.")

    with tab2:
        nu = st.text_input("Request New Identifier", key="reg_u")
        np = st.text_input("Configure Secure Token", type="password", key="reg_p")
        if st.button("Provision Account", use_container_width=True):
            if signup(nu, np):
                st.success("Provisioning successful! Proceed to authentication.")
            else:
                st.error("Registration Conflict: Identifier already allocated.")
    st.stop()

# ---------- CACHED AI RESOURCES ----------
@st.cache_resource
def instantiate_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.15,  
        max_output_tokens=2048
    )

@st.cache_resource
def instantiate_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

llm = instantiate_llm()
embeddings = instantiate_embeddings()

def run_inference(prompt_payload: str) -> str:
    try:
        response = llm.invoke(prompt_payload)
        return response.content
    except Exception as e:
        return f"🚨 Runtime Core Inference Exception: {str(e)}"

# ---------- ADVANCED VECTOR CONTEXT RETRIEVAL (HyDE) ----------
def process_pdf(uploaded_file) -> FAISS:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            length_function=len
        )
        split_chunks = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(split_chunks, embeddings)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return vector_store

def advanced_retrieval(vector_db: FAISS, user_query: str) -> str:
    hyde_generation_prompt = f"""
    You are a principal computational biologist. Generate a single highly technical, ideal paragraph answering the following request. 
    Use specialized jargon, chemical formulas, or analytical patterns appropriate for the topic. Do not include introductory notes.
    
    Target Request: {user_query}
    """
    hypothetical_answer = run_inference(hyde_generation_prompt)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})
    matched_docs = retriever.invoke(hypothetical_answer)
    return "\n\n".join([doc.page_content for doc in matched_docs])

# ---------- USER PROFILE DATA MATRIX ----------
def load_user_profile() -> dict:
    fallback_profile = {
        "name": "Mohan K",
        "role_title": "Biotechnology & AI Systems Researcher",
        "institution": "Department of Biotechnology",
        "biography": "Biotechnology student with a strong foundation in molecular biology, microbiology, and applied biosciences. Experienced in scientific literature review, research analysis, and AI-powered biotechnology applications.",
        "research_interests": "Molecular Biology, Microbiology, Bioinformatics, Computational Biology, Drug Discovery, Artificial Intelligence in Biotechnology",
        "technical_skills": "Python, Streamlit, LangChain, Google Gemini AI, FAISS, Hugging Face Embeddings, SQLite",
        "publications_projects": "AI Research Assistant System Engine (v2.0) - Lead Developer"
    }
    
    if not st.session_state.get("user"):
        return fallback_profile

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM profile WHERE username=?", (st.session_state.user,))
        res = cursor.fetchone()
        if res:
            return dict(res)
        return fallback_profile

def save_user_profile(profile_data: dict):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO profile 
        (username, name, role_title, institution, biography, research_interests, technical_skills, publications_projects) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            st.session_state.user,
            profile_data["name"],
            profile_data["role_title"],
            profile_data["institution"],
            profile_data["biography"],
            profile_data["research_interests"],
            profile_data["technical_skills"],
            profile_data["publications_projects"]
        ))
        conn.commit()

# ---------- ISOLATED CHAT STORAGE ----------
def archive_chat_interaction(role: str, message: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chats (username, role, message) VALUES (?, ?, ?)", (st.session_state.user, role, message))
        conn.commit()

def retrieve_chat_history() ->
