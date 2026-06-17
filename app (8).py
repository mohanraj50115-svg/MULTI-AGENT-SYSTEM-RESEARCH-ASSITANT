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
DB_PATH = "app.db"

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

def retrieve_chat_history() -> List[Tuple[str, str]]:
    if not st.session_state.get("user"):
        return []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = None 
        cursor = conn.cursor()
        cursor.execute("SELECT role, message FROM chats WHERE username=? ORDER BY timestamp ASC", (st.session_state.user,))
        return cursor.fetchall()

def purge_user_chat_history():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chats WHERE username=?", (st.session_state.user,))
        conn.commit()

# ---------- SIDEBAR NAVIGATION & IO CONTROL ----------
profile_state = load_user_profile()
st.sidebar.markdown(f"### 🧬 Operator: `{profile_state['name']}`")
st.sidebar.caption("🎯 Context Mode: Active")

if st.sidebar.button("Terminate Session", use_container_width=True):
    st.session_state.user = None
    st.session_state.vector = None
    st.rerun()

st.sidebar.markdown("---")
view_selection = st.sidebar.radio(
    "Control Panel Subsystems", 
    ["💬 Analytical Chat Workspace", "📄 Structural Document Analyzer", "👤 Researcher Profile Matrix", "💡 Core Documentation Workspace"]
)
st.sidebar.markdown("---")

uploaded_pdf = st.sidebar.file_uploader("Ingest Reference Manuscript (PDF)", type=["pdf"])
if uploaded_pdf:
    if st.session_state.vector is None:
        with st.sidebar.spinner("Compiling contextual database index vectors..."):
            st.session_state.vector = process_pdf(uploaded_pdf)
        st.sidebar.success("✅ Context Engine Armed.")

if st.sidebar.button("Purge Conversation Cache", type="secondary", use_container_width=True):
    purge_user_chat_history()
    st.toast("Internal chat memory tracks cleared.", icon="🗑️")
    st.rerun()

# ---------- ACTIVE APPLICATIONS INTERFACES ----------

# 1. ANALYTICAL CHAT WORKSPACE
if view_selection == "💬 Analytical Chat Workspace":
    st.title("💬 Active Cognitive Workspace")
    st.caption("Enhanced with multi-modal research persona directives for maximizing downstream outcomes.")
    
    research_mode = st.selectbox(
        "Select Pipeline AI Core Persona Optimization Mode:",
        [
            "🔬 Default Comprehensive Research Scientist",
            "🧪 Methodological Assay & Experimental Architect",
            "📊 Data Signal, Statistics & Quant Extraction Matrix",
            "🔎 Lit-Review Meta-Analysis & Gap Identifier"
        ]
    )

    st.markdown("---")
    
    historical_logs = retrieve_chat_history()
    for role_type, message_payload in historical_logs:
        with st.chat_message(role_type):
            st.markdown(message_payload)

    user_raw_input = st.chat_input("Enter target research prompt, scientific inquiry, or dataset request...")
    if user_raw_input:
        archive_chat_interaction("user", user_raw_input)
        with st.chat_message("user"):
            st.markdown(user_raw_input)

        context_payload = ""
        if st.session_state.vector is not None:
            with st.spinner("Executing HyDE similarity mapping on knowledge base..."):
                context_payload = advanced_retrieval(st.session_state.vector, user_raw_input)

        bounded_history_slice = "\n".join([f"{r.upper()}: {m}" for r, m in historical_logs[-8:]])

        # Syntactically escaped        """)
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

# ---------- USER PROFILE DATA MATRIX (BUG PROTECTED) ----------
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

def retrieve_chat_history() -> List[Tuple[str, str]]:
    if not st.session_state.get("user"):
        return []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = None 
        cursor = conn.cursor()
        cursor.execute("SELECT role, message FROM chats WHERE username=? ORDER BY timestamp ASC", (st.session_state.user,))
        return cursor.fetchall()

def purge_user_chat_history():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chats WHERE username=?", (st.session_state.user,))
        conn.commit()

# ---------- SIDEBAR NAVIGATION & IO CONTROL ----------
profile_state = load_user_profile()
st.sidebar.markdown(f"### 🧬 Operator: `{profile_state['name']}`")
st.sidebar.caption(f"🎯 Context Mode: Active")

if st.sidebar.button("Terminate Session", use_container_width=True):
    st.session_state.user = None
    st.session_state.vector = None
    st.rerun()

st.sidebar.markdown("---")
view_selection = st.sidebar.radio(
    "Control Panel Subsystems", 
    ["💬 Analytical Chat Workspace", "📄 Structural Document Analyzer", "👤 Researcher Profile Matrix", "💡 Core Documentation Workspace"]
)
st.sidebar.markdown("---")

uploaded_pdf = st.sidebar.file_uploader("Ingest Reference Manuscript (PDF)", type=["pdf"])
if uploaded_pdf:
    if st.session_state.vector is None:
        with st.sidebar.spinner("Compiling contextual database index vectors..."):
            st.session_state.vector = process_pdf(uploaded_pdf)
        st.sidebar.success("✅ Context Engine Armed.")

if st.sidebar.button("Purge Conversation Cache", type="secondary", use_container_width=True):
    purge_user_chat_history()
    st.toast("Internal chat memory tracks cleared.", icon="🗑️")
    st.rerun()

# ---------- ACTIVE APPLICATIONS INTERFACES ----------

# 1. ANALYTICAL CHAT WORKSPACE
if view_selection == "💬 Analytical Chat Workspace":
    st.title(f"💬 Active Cognitive Workspace")
    st.caption("Enhanced with multi-modal research persona directives for maximizing downstream outcomes.")
    
    research_mode = st.selectbox(
        "Select Pipeline AI Core Persona Optimization Mode:",
        [
            "🔬 Default Comprehensive Research Scientist",
            "🧪 Methodological Assay & Experimental Architect",
            "📊 Data Signal, Statistics & Quant Extraction Matrix",
            "🔎 Lit-Review Meta-Analysis & Gap Identifier"
        ]
    )

    st.markdown("---")
    
    historical_logs = retrieve_chat_history()
    for role_type, message_payload in historical_logs:
        with st.chat_message(role_type):
            st.markdown(message_payload)

    user_raw_input = st.chat_input("Enter target research prompt, scientific inquiry, or dataset request...")
    if user_raw_input:
        archive_chat_interaction("user", user_raw_input)
        with st.chat_message("user"):
            st.markdown(user_raw_input)

        context_payload = ""
        if st.session_state.vector is not None:
            with st.spinner("Executing HyDE similarity mapping on knowledge base..."):
                context_payload = advanced_retrieval(st.session_state.vector, user_raw_input)

        bounded_history_slice = "\n".join([f"{r.upper()}: {m}" for r, m in historical_logs[-8:]])

        structured_system_prompt = f"""
        [ROLE & ROLE CONTEXT]
        You are an elite Senior AI Computational Biologist optimized via this directive: **{research_mode}**.
        Your goal is to address the user's true intent with insightful, yet clear and concise responses.
        Balance deep empathy with intellectual candor: validate structural scientific challenges but directly and professionally correct flaws in logic or user assumptions.

        [HISTORICAL CHAT MEMORY CONTEXT]
        {bounded_history_slice}

        [VERIFIED DOCUMENTARY KNOWLEDGE BASES]
        {context_payload if context_payload else "No reference document uploaded. Rely strictly on verified peer-reviewed scientific consensus."}

        [EXECUTION INSTRUCTIONS]
        - Answer the primary request directly with a rigorous, high-density scientific response.
        - Prioritize numerical parameters, controls, cell models, values, and explicit data lines over generic baseline information.
        - Use advanced markdown structures like clean nested list patterns, comparison tables, or clear LaTeX blocks ($inline$ or $$display$$) for math modeling if required.
        - Do not use meta-phrases like "Based on the text...". Output the raw scientific synthesis cleanly.

        USER SEARCH REQUEST: {user_raw_input}
        EXPERIMENTAL ADVANCED RESPONSE:
        """

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing context arrays..."):
                generated_insight = run_inference(structured_system_prompt)
                st.markdown(generated_insight)
        
        archive_chat_interaction("assistant", generated_insight)

# 2. STRUCTURAL MANUSCRIPT ANALYZER
elif view_selection == "📄 Structural Document Analyzer":
    st.title("📄 Structural Document Processing & Extraction Matrix")

    if st.session_state.vector is not None:
        with st.spinner("Compiling structural framework analysis..."):
            targeted_extraction_criteria = "experimental methodology data metrics benchmarks research gaps hypotheses results"
            structural_context = advanced_retrieval(st.session_state.vector, targeted_extraction_criteria)

            analytical_meta_prompt = f"""
            Execute a systematic meta-analysis of the provided research data stream. Assemble an executive report breaking down the details according to these precise criteria:

            # 🧬 Meta-Analysis Executive Dashboard

            ## 🎯 Primary Research Objective
            *Construct an analytical statement of the paper's core hypothesis and baseline aims.*

            ## 🧪 Methodological Framework & Assay Architectures
            *Deconstruct the exact tools, data analysis models, code setups, or cell lines used here.*

            ## 📊 Verified Key Findings & Data Signals
            *Isolate quantifiable metrics, outcomes, comparisons, and performance data points.*

            ## 🔎 Identified Research Gaps & Structural Limitations
            *What did the researchers fail to control for? Highlight exactly what remains unresolved or flawed.*

            ## 🔮 Future Translational Scope
            *Detail concrete future work vectors built logically from these findings.*

            ## 💡 Non-Technical Executive Translation
            *Synthesize the complex bio-computational takeaways into plain English tailored for project coordinators.*

            DOCUMENTATION BODY TARGET FILE:
            {structural_context}
            """
            comprehensive_analysis_report = run_inference(analytical_meta_prompt)
            st.markdown(comprehensive_analysis_report)
    else:
        st.warning("⚠️ Context engine offline. Please upload a scientific manuscript PDF via the sidebar control panel to initialize analyzer.")

# 3. RESEARCHER PROFILE MATRIX
elif view_selection == "👤 Researcher Profile Matrix":
    st.title("👤 Research Intelligence Identity Matrix")
    st.caption("Manage operational parameters, biographies, and academic domain specializations below.")
    
    col1, col2 = st.tabs(["📋 View Academic Portfolio CV", "⚙️ Edit Matrix Parameters"])
    
    with col1:
        st.markdown(f"""
        # {profile_state['name']}
        ### *{profile_state['role_title']}* — **{profile_state['institution']}**
        
        ---
        
        ### 🔬 Executive Academic Summary
        {profile_state['biography']}
        
        ### 🧬 Primary Domain Specializations
        """)
        
        for interest in profile_state['research_interests'].split(','):
            if interest.strip():
                st.markdown(f"- 🧪 `{interest.strip()}`")
            
        st.markdown("### 🛠️ Core Technology Stacks & Frameworks")
        for skill in profile_state['technical_skills'].split(','):
            if skill.strip():
                st.markdown(f"- 💻 **{skill.strip()}**")
            
        st.markdown(f"""
        ### 📂 Publications, Core Tracked Projects & Credentials
        {profile_state['publications_projects']}
        
        ---
        **Cryptographic Identity Verification Key:** `{st.session_state.user}`
        """)
        
    with col2:
        st.markdown("### Update Core Portfolio Metadata")
        with st.form("matrix_profile_form"):
            form_name = st.text_input("Operator Full Name", value=profile_state['name'])
            form_role = st.text_input("Professional Target Designation", value=profile_state['role_title'])
            form_inst = st.text_input("Affiliated Research Institution", value=profile_state['institution'])
            form_bio = st.text_area("Executive Summary / Professional Bio", value=profile_state['biography'], height=120)
            form_interests = st.text_area("Research Interests (Comma Separated Items)", value=profile_state['research_interests'])
            form_skills = st.text_area("Technical Framework Skills (Comma Separated Items)", value=profile_state['technical_skills'])
            form_pub = st.text_area("Project Tracking & Bibliography Records", value=profile_state['publications_projects'], height=100)
            
            if st.form_submit_button("Commit Global Workspace Changes", use_container_width=True):
                updated_matrix = {
                    "name": form_name,
                    "role_title": form_role,
                    "institution": form_inst,
                    "biography": form_bio,
                    "research_interests": form_interests,
                    "technical_skills": form_skills,
                    "publications_projects": form_pub
                }
                save_user_profile(updated_matrix)
                st.success("🎉 Matrix metadata updated successfully! Swapping view panels...")
                st.rerun()

# 4. WORKSPACE DOCUMENTATION
elif view_selection == "💡 Core Documentation Workspace":
    st.title("💡 Advanced Platform Architecture Blueprint")
    st.markdown("""
    ### 🛡️ Cognitive Framework Optimization Guide
    
    This interface runs a zero-trust memory state engine mapped across a thread-isolated database layer, processing scientific materials using **RAG (Retrieval-Augmented Generation)** loops.

    #### 💡 Maximize Vector Search Yields
    * **Semantic Precision queries:** Instead of entering generic keyword combinations like `"CRISPR data"`, use direct relational queries like `"Identify exact p-values showing the extraction variance between wild-type and modified sequences."`
    * **State Memory Recycled Cleanup:** Use the Sidebar *Purge* option periodically during multi-manuscript sessions to clean out the system memory window. This stops context interference from earlier documents.
    """)        """)
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

# Force-initialize schemas before any application view code executes
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
    # UPDATED: Replaced deprecated gemini-1.5-flash with stable gemini-2.5-flash endpoint
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
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row  # Clean key-value conversion dictionary format
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM profile WHERE username=?", (st.session_state.user,))
        res = cursor.fetchone()
        if res:
            return dict(res)
        return {
            "name": "Mohan K",
            "role_title": "Biotechnology & AI Systems Researcher",
            "institution": "Department of Biotechnology",
            "biography": "Biotechnology student with a strong foundation in molecular biology, microbiology, and applied biosciences. Experienced in scientific literature review, research analysis, and AI-powered biotechnology applications.",
            "research_interests": "Molecular Biology, Microbiology, Bioinformatics, Computational Biology, Drug Discovery, Artificial Intelligence in Biotechnology",
            "technical_skills": "Python, Streamlit, LangChain, Google Gemini AI, FAISS, Hugging Face Embeddings, SQLite",
            "publications_projects": "AI Research Assistant System Engine (v2.0) - Lead Developer"
        }

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

# ---------- ISOLATED CHAT STORAGE (STRICTLY SAFE MANIPULATION) ----------
def archive_chat_interaction(role: str, message: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chats (username, role, message) VALUES (?, ?, ?)", (st.session_state.user, role, message))
        conn.commit()

def retrieve_chat_history() -> List[Tuple[str, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        # Explicit connection-level initialization ensures row isolation from profile row factories
        conn.row_factory = None 
        cursor = conn.cursor()
        cursor.execute("SELECT role, message FROM chats WHERE username=? ORDER BY timestamp ASC", (st.session_state.user,))
        return cursor.fetchall()

def purge_user_chat_history():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chats WHERE username=?", (st.session_state.user,))
        conn.commit()

# ---------- SIDEBAR NAVIGATION & IO CONTROL ----------
profile_state = load_user_profile()
st.sidebar.markdown(f"### 🧬 Operator: `{profile_state['name']}`")
st.sidebar.caption(f"🎯 Context Mode: Active")

if st.sidebar.button("Terminate Session", use_container_width=True):
    st.session_state.user = None
    st.session_state.vector = None
    st.rerun()

st.sidebar.markdown("---")
view_selection = st.sidebar.radio(
    "Control Panel Subsystems", 
    ["💬 Analytical Chat Workspace", "📄 Structural Document Analyzer", "👤 Researcher Profile Matrix", "💡 Core Documentation Workspace"]
)
st.sidebar.markdown("---")

uploaded_pdf = st.sidebar.file_uploader("Ingest Reference Manuscript (PDF)", type=["pdf"])
if uploaded_pdf:
    if st.session_state.vector is None:
        with st.sidebar.spinner("Compiling contextual database index vectors..."):
            st.session_state.vector = process_pdf(uploaded_pdf)
        st.sidebar.success("✅ Context Engine Armed.")

if st.sidebar.button("Purge Conversation Cache", type="secondary", use_container_width=True):
    purge_user_chat_history()
    st.toast("Internal chat memory tracks cleared.", icon="🗑️")
    st.rerun()

# ---------- ACTIVE APPLICATIONS INTERFACES ----------

# 1. ANALYTICAL CHAT WORKSPACE
if view_selection == "💬 Analytical Chat Workspace":
    st.title(f"💬 Active Cognitive Workspace")
    st.caption("Enhanced with multi-modal research persona directives for maximizing downstream outcomes.")
    
    research_mode = st.selectbox(
        "Select Pipeline AI Core Persona Optimization Mode:",
        [
            "🔬 Default Comprehensive Research Scientist",
            "🧪 Methodological Assay & Experimental Architect",
            "📊 Data Signal, Statistics & Quant Extraction Matrix",
            "🔎 Lit-Review Meta-Analysis & Gap Identifier"
        ]
    )

    st.markdown("---")
    
    historical_logs = retrieve_chat_history()
    for role_type, message_payload in historical_logs:
        with st.chat_message(role_type):
            st.markdown(message_payload)

    user_raw_input = st.chat_input("Enter target research prompt, scientific inquiry, or dataset request...")
    if user_raw_input:
        archive_chat_interaction("user", user_raw_input)
        with st.chat_message("user"):
            st.markdown(user_raw_input)

        context_payload = ""
        if st.session_state.vector is not None:
            with st.spinner("Executing HyDE similarity mapping on knowledge base..."):
                context_payload = advanced_retrieval(st.session_state.vector, user_raw_input)

        bounded_history_slice = "\n".join([f"{r.upper()}: {m}" for r, m in historical_logs[-8:]])

        structured_system_prompt = f"""
        [ROLE & ROLE CONTEXT]
        You are an elite Senior AI Computational Biologist optimized via this directive: **{research_mode}**.
        Your goal is to address the user's true intent with insightful, yet clear and concise responses.
        Balance deep empathy with intellectual candor: validate structural scientific challenges but directly and professionally correct flaws in logic or user assumptions.

        [HISTORICAL CHAT MEMORY CONTEXT]
        {bounded_history_slice}

        [VERIFIED DOCUMENTARY KNOWLEDGE BASES]
        {context_payload if context_payload else "No reference document uploaded. Rely strictly on verified peer-reviewed scientific consensus."}

        [EXECUTION INSTRUCTIONS]
        - Answer the primary request directly with a rigorous, high-density scientific response.
        - Prioritize numerical parameters, controls, cell models, values, and explicit data lines over generic baseline information.
        - Use advanced markdown structures like clean nested list patterns, comparison tables, or clear LaTeX blocks ($inline$ or $$display$$) for math modeling if required.
        - Do not use meta-phrases like "Based on the text...". Output the raw scientific synthesis cleanly.

        USER SEARCH REQUEST: {user_raw_input}
        EXPERIMENTAL ADVANCED RESPONSE:
        """

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing context arrays..."):
                generated_insight = run_inference(structured_system_prompt)
                st.markdown(generated_insight)
        
        archive_chat_interaction("assistant", generated_insight)

# 2. STRUCTURAL MANUSCRIPT ANALYZER
elif view_selection == "📄 Structural Document Analyzer":
    st.title("📄 Structural Document Processing & Extraction Matrix")

    if st.session_state.vector is not None:
        with st.spinner("Compiling structural framework analysis..."):
            targeted_extraction_criteria = "experimental methodology data metrics benchmarks research gaps hypotheses results"
            structural_context = advanced_retrieval(st.session_state.vector, targeted_extraction_criteria)

            analytical_meta_prompt = f"""
            Execute a systematic meta-analysis of the provided research data stream. Assemble an executive report breaking down the details according to these precise criteria:

            # 🧬 Meta-Analysis Executive Dashboard

            ## 🎯 Primary Research Objective
            *Construct an analytical statement of the paper's core hypothesis and baseline aims.*

            ## 🧪 Methodological Framework & Assay Architectures
            *Deconstruct the exact tools, data analysis models, code setups, or cell lines used here.*

            ## 📊 Verified Key Findings & Data Signals
            *Isolate quantifiable metrics, outcomes, comparisons, and performance data points.*

            ## 🔎 Identified Research Gaps & Structural Limitations
            *What did the researchers fail to control for? Highlight exactly what remains unresolved or flawed.*

            ## 🔮 Future Translational Scope
            *Detail concrete future work vectors built logically from these findings.*

            ## 💡 Non-Technical Executive Translation
            *Synthesize the complex bio-computational takeaways into plain English tailored for project coordinators.*

            DOCUMENTATION BODY TARGET FILE:
            {structural_context}
            """
            comprehensive_analysis_report = run_inference(analytical_meta_prompt)
            st.markdown(comprehensive_analysis_report)
    else:
        st.warning("⚠️ Context engine offline. Please upload a scientific manuscript PDF via the sidebar control panel to initialize analyzer.")

# 3. RESEARCHER PROFILE MATRIX
elif view_selection == "👤 Researcher Profile Matrix":
    st.title("👤 Research Intelligence Identity Matrix")
    st.caption("Manage operational parameters, biographies, and academic domain specializations below.")
    
    col1, col2 = st.tabs(["📋 View Academic Portfolio CV", "⚙️ Edit Matrix Parameters"])
    
    with col1:
        st.markdown(f"""
        # {profile_state['name']}
        ### *{profile_state['role_title']}* — **{profile_state['institution']}**
        
        ---
        
        ### 🔬 Executive Academic Summary
        {profile_state['biography']}
        
        ### 🧬 Primary Domain Specializations
        """)
        
        for interest in profile_state['research_interests'].split(','):
            if interest.strip():
                st.markdown(f"- 🧪 `{interest.strip()}`")
            
        st.markdown("### 🛠️ Core Technology Stacks & Frameworks")
        for skill in profile_state['technical_skills'].split(','):
            if skill.strip():
                st.markdown(f"- 💻 **{skill.strip()}**")
            
        st.markdown(f"""
        ### 📂 Publications, Core Tracked Projects & Credentials
        {profile_state['publications_projects']}
        
        ---
        **Cryptographic Identity Verification Key:** `{st.session_state.user}`
        """)
        
    with col2:
        st.markdown("### Update Core Portfolio Metadata")
        with st.form("matrix_profile_form"):
            form_name = st.text_input("Operator Full Name", value=profile_state['name'])
            form_role = st.text_input("Professional Target Designation", value=profile_state['role_title'])
            form_inst = st.text_input("Affiliated Research Institution", value=profile_state['institution'])
            form_bio = st.text_area("Executive Summary / Professional Bio", value=profile_state['biography'], height=120)
            form_interests = st.text_area("Research Interests (Comma Separated Items)", value=profile_state['research_interests'])
            form_skills = st.text_area("Technical Framework Skills (Comma Separated Items)", value=profile_state['technical_skills'])
            form_pub = st.text_area("Project Tracking & Bibliography Records", value=profile_state['publications_projects'], height=100)
            
            if st.form_submit_button("Commit Global Workspace Changes", use_container_width=True):
                updated_matrix = {
                    "name": form_name,
                    "role_title": form_role,
                    "institution": form_inst,
                    "biography": form_bio,
                    "research_interests": form_interests,
                    "technical_skills": form_skills,
                    "publications_projects": form_pub
                }
                save_user_profile(updated_matrix)
                st.success("🎉 Matrix metadata updated successfully! Swapping view panels...")
                st.rerun()

# 4. WORKSPACE DOCUMENTATION
elif view_selection == "💡 Core Documentation Workspace":
    st.title("💡 Advanced Platform Architecture Blueprint")
    st.markdown("""
    ### 🛡️ Cognitive Framework Optimization Guide
    
    This interface runs a zero-trust memory state engine mapped across a thread-isolated database layer, processing scientific materials using **RAG (Retrieval-Augmented Generation)** loops.

    #### 💡 Maximize Vector Search Yields
    * **Semantic Precision queries:** Instead of entering generic keyword combinations like `"CRISPR data"`, use direct relational queries like `"Identify exact p-values showing the extraction variance between wild-type and modified sequences."`
    * **State Memory Recycled Cleanup:** Use the Sidebar *Purge* option periodically during multi-manuscript sessions to clean out the system memory window. This stops context interference from earlier documents.
    """)
