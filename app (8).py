import streamlit as st
import os
import sqlite3
import hashlib
import tempfile

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

# ---------- DATABASE INTERFACE ----------
DB_PATH = "research_final.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT, message TEXT)")
    cursor.execute("CREATE TABLE IF NOT EXISTS profile (username TEXT PRIMARY KEY, name TEXT, role_title TEXT, institution TEXT, biography TEXT, research_interests TEXT, technical_skills TEXT, publications_projects TEXT)")
    conn.commit()
    conn.close()

init_db()

# ---------- CORE FUNCTIONS ----------
def hash_pass(p): return hashlib.sha256(p.encode()).hexdigest()

def login(u, p):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_pass(p)))
    res = cursor.fetchone()
    conn.close()
    return res

def signup(u, p):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, hash_pass(p)))
        conn.commit()
        conn.close()
        return True
    except: return False

def save_user_profile(u, data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO profile VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
                   (u, data['name'], data['role_title'], data['institution'], data['biography'], data['research_interests'], data['technical_skills'], data['publications_projects']))
    conn.commit()
    conn.close()

def load_user_profile(u):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM profile WHERE username=?", (u,))
    res = cursor.fetchone()
    conn.close()
    return dict(res) if res else None

def archive_chat(u, role, msg):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (username, role, message) VALUES (?, ?, ?)", (u, role, msg))
    conn.commit()
    conn.close()

def retrieve_chat_history(u):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM chats WHERE username=? ORDER BY id ASC", (u,))
    res = cursor.fetchall()
    conn.close()
    return res

# ---------- SESSION & UI ----------
if "user" not in st.session_state: st.session_state.user = None

if not st.session_state.user:
    st.title("🔐 Access Gateway")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(u, p):
            st.session_state.user = u
            st.rerun()
    if st.button("Sign Up"):
        if signup(u, p): st.success("Account created!")
    st.stop()

st.sidebar.button("Logout", on_click=lambda: st.session_state.update(user=None))
view = st.sidebar.radio("Navigation", ["💬 Chat", "👤 Profile"])

if view == "💬 Chat":
    st.title("💬 Research Workspace")
    for r, m in retrieve_chat_history(st.session_state.user):
        with st.chat_message(r): st.markdown(m)
    
    if q := st.chat_input("Ask a research question..."):
        archive_chat(st.session_state.user, "user", q)
        with st.chat_message("user"): st.markdown(q)
        # Simplified inference call
        response = "System active. Awaiting vector index."
        archive_chat(st.session_state.user, "assistant", response)
        st.rerun()

elif view == "👤 Profile":
    st.title("👤 Profile")
    p = load_user_profile(st.session_state.user) or {"name": "", "role_title": "", "institution": "", "biography": "", "research_interests": "", "technical_skills": "", "publications_projects": ""}
    with st.form("p"):
        data = {k: st.text_input(k.replace('_', ' ').title(), value=p.get(k, "")) for k in p.keys()}
        if st.form_submit_button("Save"):
            save_user_profile(st.session_state.user, data)
            st.success("Updated!")
