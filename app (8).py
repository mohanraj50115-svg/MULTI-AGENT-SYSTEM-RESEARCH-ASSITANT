import streamlit as st
import os
import sqlite3
import hashlib
import re
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

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

def clean_response(res):
    try:
        text = res.content

        if isinstance(text, list):
            cleaned = []
            for item in text:
                if hasattr(item, "text"):
                    cleaned.append(item.text)
                else:
                    cleaned.append(str(item))

            text = "\n".join(cleaned)

        text = str(text)

        text = text.replace("\\n", "\n")
        text = text.replace("###", "")
        text = text.replace("**", "")

        return text.strip()

    except Exception:
        return str(res)

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

# ---------- RETRIEVAL ----------
def retrieval(vector, q):
    retriever = vector.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(q)

    return "\n\n".join(
        [d.page_content for d in docs]
    )
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
User:prompt = f"""
You are a scientific research assistant.

Read the research paper content below and provide:

1. Title of the study
2. Research objective
3. Methodology
4. Key findings
5. Applications
6. Limitations
7. Simple summary for students

Paper:
{ctx}

Write in clean markdown format.
"""

ans = run_llm(prompt)

        save_chat("assistant", ans)

        with st.chat_message("assistant"):
            st.write(ans)

elif page == "Paper Analyzer":
    st.title("📄 Research Paper Analysis")

    if "vector" in st.session_state:

        with st.spinner("Analyzing paper..."):

            ctx = retrieval(
                st.session_state.vector,
                "research objective methodology findings conclusion"
            )

            prompt = f"""
            Analyze this research paper and provide:

            ## Research Objective

            ## Methodology

            ## Key Findings

            ## Conclusion

            ## Future Scope

            ## Simple Explanation

            Paper:
            {ctx}
            """

            ans = run_llm(prompt)

        st.markdown(ans)

    else:
        st.warning("Please upload a research paper PDF first.")

# ---------- PROFILE ----------
elif page == "Profile":
    st.title("👤 My Profile")

    current_name = get_name()

    name = st.text_input(
        "Full Name",
        value=current_name
    )

    if st.button("Save Profile"):
        save_name(name)
        st.success("Profile updated successfully!")

    st.markdown(f"""
    ### User Information

    **Name:** {current_name}

    **Username:** {st.session_state.user}
    """)

    st.markdown("""
    ### About Me

    Biotechnology student with a strong foundation in molecular biology, microbiology, and applied biosciences. Experienced in scientific literature review, research analysis, and AI-powered biotechnology applications.

    ### Research Interests
    - Molecular Biology
    - Microbiology
    - Bioinformatics
    - Computational Biology
    - Drug Discovery
    - Artificial Intelligence in Biotechnology

    ### Technical Skills
    - Python
    - Streamlit
    - LangChain
    - Google Gemini AI
    - FAISS
    - Hugging Face Embeddings
    - SQLite
    """)

# ---------- HELP ----------
elif page == "Help":
    st.title("❓ Help & User Guide")

    st.markdown("""
    ## Welcome to AI Research Assistant System

    This platform helps researchers and students analyze scientific literature using Artificial Intelligence and Retrieval-Augmented Generation (RAG).

    ### 🚀 Getting Started

    #### Step 1: Upload a Research Paper
    - Use the PDF uploader in the sidebar.
    - Upload a scientific article or research paper in PDF format.
    - The system will process the document and create a searchable knowledge base.

    #### Step 2: Ask Questions
    - Go to the **Chat** section.
    - Ask questions related to your uploaded paper.
    - The AI will retrieve relevant information and provide contextual answers.

    #### Step 3: Analyze the Paper
    - Open **Paper Analyzer**.
    - Get an AI-generated summary of the uploaded research article.

    ### 🔬 Supported Research Tasks
    - Research Paper Summarization
    - Literature Review Support
    - Scientific Question Answering
    - Research Gap Exploration
    - Experimental Design Assistance
    - Knowledge Retrieval from PDFs

    ### 🧠 Technologies Used
    - Google Gemini AI
    - LangChain
    - FAISS Vector Database
    - Hugging Face Embeddings
    - Streamlit
    - SQLite

    ### 💡 Tips
    - Upload clear and readable PDFs.
    - Ask specific scientific questions for better responses.
    - Use the Paper Analyzer for quick literature insights.
    - Clear chat history from the sidebar when starting a new project.

    Happy Researching! 🧬
    """)

  # ---------- ABOUT ----------
elif page == "About":
    st.title("👨‍🔬 About the Developer")

    st.markdown("""
    ## Mohan K

    Biotechnology Student | AI Research Enthusiast

    I am a Biotechnology student with a strong foundation in molecular biology, microbiology, and applied biosciences. My interests lie at the intersection of biotechnology and artificial intelligence, where I explore innovative solutions for research automation, scientific knowledge discovery, and drug development.

    ### Academic Interests
    - Molecular Biology
    - Microbiology
    - Bioinformatics
    - Computational Biology
    - Artificial Intelligence in Biotechnology
    - Drug Discovery

    ### Technical Skills
    - Python
    - Streamlit
    - LangChain
    - Google Gemini AI
    - FAISS
    - Hugging Face Embeddings
    - SQLite

    ### About This Project
    The AI Research Assistant System helps researchers and students analyze scientific literature, summarize research papers, and retrieve knowledge using AI-powered Retrieval-Augmented Generation (RAG).

    ### Career Objective
    Seeking opportunities to develop research, analytical, and biotechnology skills through hands-on projects and collaborative research.

    📧 Email: mohanraj50115@gmail.com

    🔗 LinkedIn:
    www.linkedin.com/in/mohan-k-307749308

    🚀 Passionate about combining AI and Biotechnology to solve real-world challenges.
    """)
""")
