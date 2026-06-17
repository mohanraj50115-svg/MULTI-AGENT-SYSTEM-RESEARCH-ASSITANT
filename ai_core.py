import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class ResearchEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    def process_pdf(self, file_bytes):
        """Creates a FAISS index from uploaded PDF bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            loader = PyMuPDFLoader(tmp.name)
            docs = loader.load()
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        os.remove(tmp.name)
        return vector_store

    def query_engine(self, vector_store, query, context_history=""):
        """Retrieves context and generates a response."""
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""
        Role: Senior Computational Biologist.
        Context: {context}
        History: {context_history}
        Query: {query}
        Instruction: Provide a rigorous, evidence-based academic response.
        """
        return self.llm.invoke(prompt).content
