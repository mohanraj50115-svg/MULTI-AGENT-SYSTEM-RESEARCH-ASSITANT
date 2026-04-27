import streamlit as st
import os
import sqlite3
import hashlib
import re

# ---------- LLM ----------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- PAGE ----------
st.set_page_config(page_title="🧬 AI Research System", layout="wide")

# ---------- DATABASE ----------
conn = sqlite3.connect("app.db", check_same_thread=False)
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
    message TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS profile (
    username TEXT,
    name TEXT
)
""")

conn.commit()

# ---------- SECURITY ----------
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def signup(u, p):
    try:
        cursor.execute("INSERT INTO users VALUES (?, ?)", (u, hash_password(p)))
        conn.commit()
        return True
    except:
        return False

def login(u, p):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_password(p)))
    return cursor.fetchone()

# ---------- SESSION ----------
if "user" not in st.session_state:
    st.session_state.user = None

# ---------- LOGIN UI ----------
if st.session_state.user is None:
    st.title("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(u, p):
                st.session_state.user = u
                st.success("Logged in")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if signup(u, p):
                st.success("Account created")
            else:
                st.error("Username exists")

    st.stop()

# ---------- MODEL ----------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0.3
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- CLEAN ----------
def clean_response(res):
    text = res.content if hasattr(res, "content") else str(res)
    if isinstance(text, list):
        text = " ".join([str(i) for i in text])
    text = str(text)
    text = re.sub(r'signature.*', '', text, flags=re.I)
    return text.strip()

def run_llm(prompt):
    return clean_response(llm.invoke(prompt))

# ---------- PROFILE ----------
def get_name():
    cursor.execute("SELECT name FROM profile WHERE username=?", (st.session_state.user,))
    r = cursor.fetchone()
    return r[0] if r else st.session_state.user

def save_name(name):
    cursor.execute("DELETE FROM profile WHERE username=?", (st.session_state.user,))
    cursor.execute("INSERT INTO profile VALUES (?, ?)", (st.session_state.user, name))
    conn.commit()

# ---------- CHAT MEMORY ----------
def save_chat(role, msg):
    cursor.execute("INSERT INTO chats (username, role, message) VALUES (?, ?, ?)",
                   (st.session_state.user, role, msg))
    conn.commit()

def load_chat():
    cursor.execute("SELECT role, message FROM chats WHERE username=?", (st.session_state.user,))
    return cursor.fetchall()

def clear_chat():
    cursor.execute("DELETE FROM chats WHERE username=?", (st.session_state.user,))
    conn.commit()

# ---------- PDF ----------
def process_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    loader = PyMuPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    return FAISS.from_documents(chunks, embeddings)

def retrieval(vector, q):
    docs = vector.as_retriever().invoke(q)
    return "\n\n".join([d.page_content for d in docs])

# ---------- AGENT ----------
def mode(q):
    q = q.lower()
    if "experiment" in q:
        return "experiment"
    elif "proposal" in q:
        return "proposal"
    elif "gap" in q:
        return "research"
    elif "paper" in q:
        return "paper"
    elif "pdf" in q:
        return "rag"
    else:
        return "explain"

# ---------- SIDEBAR ----------
st.sidebar.write(f"👤 {get_name()}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

page = st.sidebar.radio("Menu", ["Chat", "Paper Analyzer", "Profile", "Help", "About"])

pdf = st.sidebar.file_uploader("Upload PDF")

if pdf:
    st.session_state.vector = process_pdf(pdf)
    st.sidebar.success("PDF ready")

if st.sidebar.button("Clear Chat"):
    clear_chat()
    st.rerun()

# ---------- CHAT ----------
if page == "Chat":
    st.title(f"💬 Welcome {get_name()}")

    chat = load_chat()
    for r, m in chat:
        with st.chat_message(r):
            st.write(m)

    q = st.chat_input("Ask...")
    if q:
        save_chat("user", q)
        with st.chat_message("user"):
            st.write(q)

        context = ""
        if "vector" in st.session_state:
            context = retrieval(st.session_state.vector, q)

        history = "\n".join([f"{r}:{m}" for r, m in chat[-5:]])

        prompt = f"""
User: {get_name()}
History:
{history}
Context:
{context}
Question:
{q}
Answer in paragraph form.
"""

        ans = run_llm(prompt)

        save_chat("assistant", ans)

        with st.chat_message("assistant"):
            st.write(ans)

# ---------- PAPER ----------
elif page == "Paper Analyzer":
    st.title("📄 Paper Analyzer")
    if "vector" in st.session_state:
        ctx = retrieval(st.session_state.vector, "summarize paper")
        ans = run_llm(f"Explain paper clearly:\n{ctx}")
        st.write(ans)
    else:
        st.warning("Upload PDF")


# ---------- HELP ----------
elif page == "Help":
    st.write("""
Upload PDF → Ask questions
Use chat or analyzer
Supports research + experiments
""")

# ---------- ABOUT ----------
elif page == "About":
    st.title("👤 About")

    st.markdown("""
Mr. Mohan K is a researcher in Biotechnology, currently pursuing a PhD at the Vellore Institute of Technology.
His work focuses on AI-driven drug discovery, computational biology, and intelligent research systems. 
He is building advanced platforms that combine artificial intelligence with biotechnology to accelerate scientific innovation.

### 📞 Contact
Phone: 9361245583  
Email: mohanraj50115@gmail.com
""")

AI Research Assistant System
Supports RAG, multi-agent reasoning
Future: drug discovery AI
""")
