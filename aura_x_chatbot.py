import os
import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Configuration ---
PDF_FILE_PATH = "IndianSpacePolicy2023.pdf"
VECTOR_STORE_DIRECTORY = "policy_vector_store_hf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "gpt2"  # Using text-generation model

@st.cache_resource
def load_pdf_and_split(file_path, chunk_size=1000, chunk_overlap=200):
    """Loads a PDF and splits it into chunks."""
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found.")
        st.stop()
    
    reader = PdfReader(file_path)
    documents = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        # Split text into chunks
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                documents.append({
                    "content": chunk,
                    "page": page_num + 1
                })
    
    return documents

@st.cache_resource
def get_embedding_model():
    """Load the embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_qa_model():
    """Load the text generation model."""
    return pipeline("text-generation", model=QA_MODEL, max_new_tokens=150)

@st.cache_resource
def create_vector_store(documents):
    """Create ChromaDB vector store from documents."""
    client = chromadb.Client()
    
    # Create or get collection
    collection = client.get_or_create_collection(name="indian_space_policy")
    
    # Get embeddings model
    embeddings_model = get_embedding_model()
    
    # Add documents to the collection
    doc_ids = []
    doc_embeddings = []
    doc_contents = []
    doc_metadata = []
    
    for idx, doc in enumerate(documents):
        doc_id = f"doc_{idx}"
        embedding = embeddings_model.encode(doc["content"]).tolist()
        
        doc_ids.append(doc_id)
        doc_embeddings.append(embedding)
        doc_contents.append(doc["content"])
        doc_metadata.append({"page": doc["page"]})
    
    collection.add(
        ids=doc_ids,
        embeddings=doc_embeddings,
        documents=doc_contents,
        metadatas=doc_metadata
    )
    
    return collection

def retrieve_context(collection, query, num_results=3):
    """Retrieve the most relevant documents for a query."""
    embeddings_model = get_embedding_model()
    query_embedding = embeddings_model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results
    )
    
    context_docs = []
    if results["documents"]:
        for docs, ids, metadatas in zip(results["documents"], results["ids"], results["metadatas"]):
            for doc, doc_id, metadata in zip(docs, ids, metadatas):
                context_docs.append({
                    "content": doc,
                    "page": metadata["page"],
                    "id": doc_id
                })
    
    return context_docs

def generate_answer(query, context_docs):
    """Generate an answer using the text generation model."""
    qa_model = get_qa_model()
    
    # Combine context documents
    context_text = "\n\n".join([doc["content"] for doc in context_docs])
    
    # If no context is found
    if not context_text.strip():
        return "I do not know the answer to your question as it is not covered in the Indian Space Policy 2023.", []
    
    try:
        # Create a concise prompt for text generation
        prompt = f"""Context: {context_text[:800]}

Question: {query}

Answer based on the context above:"""
        
        # Generate answer
        result = qa_model(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=50256  # GPT-2 EOS token to suppress warnings
        )
        
        generated_text = result[0]['generated_text']
        
        # Extract the answer part after the prompt
        if "Answer based on the context above:" in generated_text:
            answer = generated_text.split("Answer based on the context above:")[-1].strip()
        else:
            answer = generated_text[len(prompt):].strip()
        
        # Clean up the answer
        if not answer or len(answer) < 5:
            # Fallback: return the most relevant context directly
            answer = f"Based on the Indian Space Policy 2023: {context_text[:400]}..."
        
        # Limit answer length
        if len(answer) > 600:
            answer = answer[:600] + "..."
        
        return answer, context_docs
    except Exception as e:
        # Fallback: return context directly instead of "I don't know"
        fallback = f"Based on the Indian Space Policy 2023: {context_text[:400]}..."
        return fallback, context_docs

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Aura-X Chatbot",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # --- Custom CSS for a polished look ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }
    
    .answer-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2a2a3d 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        color: #e2e8f0 !important;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .source-card {
        background: #1e1e2e;
        border: 1px solid #3a3a5c;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin: 0.75rem 0;
        transition: box-shadow 0.2s ease;
        color: #e2e8f0 !important;
    }
    
    .source-card:hover {
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .status-badge {
        display: inline-block;
        background: linear-gradient(135deg, #34d399, #059669);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    div[data-testid="stTextInput"] input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: border-color 0.2s ease;
    }
    
    div[data-testid="stTextInput"] input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
    }
    
    .stSpinner > div {
        border-color: #667eea !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ About")
        st.markdown("""
        **AURA-X** is an Explainable AI framework 
        for Indian Space Policy compliance.
        
        **Tech Stack:**
        - 🔍 ChromaDB vector search
        - 🧠 sentence-transformers embeddings
        - 📝 GPT-2 text generation
        - 📄 Indian Space Policy 2023
        """)
        st.divider()
        st.markdown("### 💡 Sample Questions")
        st.markdown("""
        - What is the role of IN-SPACe?
        - What are the objectives of the space policy?
        - How does the policy address private sector participation?
        - What is ISRO's role under this policy?
        """)
        st.divider()
        st.caption("Built with ❤️ using Streamlit")
    
    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Aura-X Chatbot</h1>
        <p>Explainable AI for Indian Space Policy 2023 Compliance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and process PDF
    with st.spinner("🔄 Loading knowledge base — this may take a minute on first run..."):
        documents = load_pdf_and_split(PDF_FILE_PATH)
        collection = create_vector_store(documents)
    
    st.markdown('<span class="status-badge">✓ Knowledge Base Ready</span>', unsafe_allow_html=True)
    st.write("")
    
    # Input query
    query = st.text_input(
        "Ask a question about the Indian Space Policy 2023:",
        placeholder="e.g., What is the role of IN-SPACe?",
        label_visibility="visible"
    )
    
    if query:
        with st.spinner("🔍 Searching and generating answer..."):
            context_docs = retrieve_context(collection, query, num_results=3)
            answer, source_docs = generate_answer(query, context_docs)
        
        # Display answer
        st.markdown("### 💬 Answer")
        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)
        
        # Display source documents
        with st.expander("📄 View Source Documents", expanded=False):
            if source_docs:
                for idx, doc in enumerate(source_docs, 1):
                    content_preview = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
                    st.markdown(f"""
                    <div class="source-card">
                        <strong style="color: #a5b4fc;">📌 Source {idx} — Page {doc['page']}</strong><br>
                        <span style="color: #cbd5e1; font-size: 0.9rem;">{content_preview}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No source documents found.")

if __name__ == "__main__":
    main()
