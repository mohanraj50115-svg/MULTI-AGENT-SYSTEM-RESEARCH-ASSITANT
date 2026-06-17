import streamlit as st
import os
import sqlite3
import hashlib
import re

# ---------- LLM & LANGCHAIN ----------
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Must be configured before using the Gemini API
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---------- PAGE ----------
st.set_page_config(page_title="🧬 AI Research System", layout="wide")

# ---------- DATABASE SETUP ----------
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
    username TEXT PRIMARY KEY,
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
    except sqlite3.IntegrityError:
        return False

def login(u, p):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u, hash_password(p)))
    return cursor.fetchone()

# ---------- SESSION STATE ----------
if "user" not in st.session_state:
    st.session_state.user = None

# ---------- LOGIN / SIGNUP UI ----------
if st.session_state.user is None:
    st.title("🔐 Login / Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login(u, p):
                st.session_state.user = u
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        u = st.text_input("New Username", key="signup_user")
        p = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Create Account"):
            if signup(u, p):
                st.success("Account created! Please log in.")
            else:
                st.error("Username already exists")
    st.stop()

# ---------- MODEL INITIALIZATION ----------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Upgraded to stable generation architecture 
        temperature=0.3
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = load_llm()
embeddings = load_embeddings()

# Helper function to invoke LLM safely
def run_llm(prompt_text):
    try:
        response = llm.invoke(prompt_text)
        return response.content
    except Exception as e:
        return f"An error occurred while generating a response: {str(e)}"

# ---------- PROFILE MANAGEMENT ----------
def get_name():
    cursor.execute("SELECT name FROM profile WHERE username=?", (st.session_state.user,))
    r = cursor.fetchone()
    return r[0] if r else st.session_state.user

def save_name(name):
    cursor.execute("INSERT OR REPLACE INTO profile (username, name) VALUES (?, ?)", (st.session_state.user, name))
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

# ---------- PDF & RAG ENGINE ----------
def process_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    loader = PyMuPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Clean up temp file
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")

    return FAISS.from_documents(chunks, embeddings)

def retrieval(vector_store, query):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.write(f"劈 {get_name()}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

page = st.sidebar.radio("Menu", ["Chat", "Paper Analyzer", "Profile", "Help", "About"])

pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if pdf:
    with st.sidebar.spinner("Parsing document chunks..."):
        st.session_state.vector = process_pdf(pdf)
    st.sidebar.success("PDF knowledge index ready!")

if st.sidebar.button("Clear Chat History"):
    clear_chat()
    st.rerun()

# ---------- APP PAGES ----------

# 1. CHAT UI
if page == "Chat":
    st.title(f"💬 Welcome, {get_name()}")

    chat = load_chat()
    for role, message in chat:
        with st.chat_message(role):
            st.write(message)

    q = st.chat_input("Ask a question about biology, AI, or your document...")
    if q:
        save_chat("user", q)
        with st.chat_message("user"):
            st.write(q)

        context = ""
        if "vector" in st.session_state:
            context = retrieval(st.session_state.vector, q)

        # Get recent 5 pairs of exchanges for context memory window
        history = "\n".join([f"{r}: {m}" for r, m in chat[-10:]])

        prompt = f"""
You are an advanced AI Research Assistant tailored for Biotechnology and Computational Biology.
User: {get_name()}

Conversation History:
{history}

Document Reference Context:
{context}

Question:
{q}

Provide a well-structured academic response. Use markdown formatting elements like bolding, tables, or bulleted items when appropriate to make complex metrics readable.
"""
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                ans = run_llm(prompt)
                st.write(ans)
        
        save_chat("assistant", ans)

# 2. PAPER ANALYZER UI
elif page == "Paper Analyzer":
    st.title("📄 Research Paper Automated Executive Summary")

    if "vector" in st.session_state:
        with st.spinner("Extracting parameters and processing structural context..."):
            ctx = retrieval(
                st.session_state.vector,
                "research objective methodology findings conclusion limitations future work"
            )

            prompt = f"""
Analyze this research data extract and organize a detailed report incorporating these sections exactly:
Use clean, professional Markdown syntax.

## Research Objective
*Provide clear bullet items outlining objectives*

## Methodology
*Analyze design patterns, architectures or assays used*

## Key Findings
*Outline experimental results*

## Conclusion
*Summarize primary takeaways*

## Future Scope
*Where can this work expand next?*

## Simple Explanation
*A high-level plain English summary mapping what this means for non-technical peers*

Paper Context:
{ctx}
"""
            ans = run_llm(prompt)
        st.markdown(ans)
    else:
        st.warning("⚠️ Please upload a research paper PDF in the sidebar to initialize analytics.")

# 3. PROFILE UI
elif page == "Profile":
    st.title("👤 Research Profile Settings")
    current_name = get_name()

    name = st.text_input("Full Name", value=current_name)
    if st.button("Save Profile"):
        save_name(name)
        st.success("Profile updated successfully!")
        st.rerun()

    st.markdown("---")
    st.markdown(f"""
    ### User Information
    * **Display Name:** {current_name}
    * **System Identifier:** {st.session_state.user}
    
    ### About Me
    Biotechnology student with a strong foundation in molecular biology, microbiology, and applied biosciences. Experienced in scientific literature review, research analysis, and AI-powered biotechnology applications.

    ### Research Interests
    * Molecular Biology & Genetics
    * Bioinformatics & Computational Modeling
    * Automated High-Throughput Screening & Drug Discovery
    * Machine Learning Application inside Applied Biology

    ### Core Framework Competencies
    * **Languages:** Python, SQL
    * **AI Engineering:** LangChain, Google Gemini API, FAISS, Hugging Face Tokenizers
    * **UI/Data Deployments:** Streamlit, SQLite
    """)

# 4. HELP GUIDE
elif page == "Help":
    st.title("❓ Help & Documentation Engine")
    st.markdown("""
    ### 🚀 Getting Started Workflow

    1. **Upload Reference Media:** Drag and drop any multi-page scientific study manuscript (PDF format) into the left-hand sidebar workspace tool. 
    2. **Context Engine Build:** The platform automatically splits data arrays using recursive text splitters to embed data blocks via vector modeling directly into a running local storage database instance.
    3. **Query Engine Processing:** Open the **Chat** interface to run complex contextual lookups or use **Paper Analyzer** to produce custom executive summaries instantaneously.
    
    ### 💡 Pro-Tips for Optimal Prompt Performance
    * Keep questions direct and contextually aligned to what exists inside your uploaded document schema.
    * Use the **Clear Chat History** option whenever starting a brand new topic framework to prevent token overflow bottlenecks.
    """)

# 5. ABOUT PAGE
elif page == "About":
    st.title("👨‍🔬 System Developer Blueprint")
    st.markdown("""
    ### Mohan K
    **Biotechnology Scholar & Applied AI Developer**
    
    This interface bridges the gap between processing raw scientific publications and surfacing instant intelligence insights using Retrieval-Augmented Generation (RAG).

    * **Contact Interface:** mohanraj50115@gmail.com
    * **Professional Networks:** [LinkedIn Workspace Profile](http://www.linkedin.com/in/mohan-k-307749308)
    """)
