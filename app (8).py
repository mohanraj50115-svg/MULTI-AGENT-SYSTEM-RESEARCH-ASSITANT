import streamlit as st
from ai_core import ResearchEngine
from db_manager import DatabaseManager

# Initialize the modules
# Use session state to keep instances alive
if "engine" not in st.session_state:
    st.session_state.engine = ResearchEngine()
if "db" not in st.session_state:
    st.session_state.db = DatabaseManager()

st.set_page_config(page_title="🧬 Enterprise Research Engine", layout="wide")

# --- UI LOGIC ---
if st.session_state.get("user") is None:
    st.title("🔐 Secure Access Gateway")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if st.session_state.db.verify_user(u, p):
            st.session_state.user = u
            st.rerun()
    st.stop()

# --- MAIN DASHBOARD ---
st.sidebar.title(f"Operator: {st.session_state.user}")
page = st.sidebar.radio("Navigation", ["💬 Chat Workspace", "📄 Analyzer"])

if page == "💬 Chat Workspace":
    st.title("💬 Analytical Chat")
    
    # Render history
    history = st.session_state.db.get_history(st.session_state.user)
    for role, msg in history:
        with st.chat_message(role): st.markdown(msg)
    
    # Process input
    if q := st.chat_input("Scientific Inquiry..."):
        with st.chat_message("user"): st.markdown(q)
        st.session_state.db.save_message(st.session_state.user, "user", q)
        
        # Integrate AI Logic
        resp = st.session_state.engine.query_engine(
            st.session_state.get("vector"), q
        )
        with st.chat_message("assistant"): st.markdown(resp)
        st.session_state.db.save_message(st.session_state.user, "assistant", resp)

elif page == "📄 Analyzer":
    st.title("📄 Manuscript Analyzer")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf:
        with st.spinner("Indexing vector space..."):
            st.session_state.vector = st.session_state.engine.process_pdf(pdf.getvalue())
            st.success("Context indexed successfully.")
