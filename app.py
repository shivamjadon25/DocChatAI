import streamlit as st
import os
import hashlib
from google import genai
from google.genai import types

from rag_engine import parse_document, chunk_text, InMemoryVectorStore

# Page Config
st.set_page_config(
    page_title="Document RAG Chatbot",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper for API Key retrieval
def get_api_key():
    if "api_key_input" in st.session_state and st.session_state.api_key_input:
        return st.session_state.api_key_input.strip()
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    return ""

def get_shared_client(api_key):
    """
    Returns a shared, persistent Client instance.
    This prevents 'client has been closed' errors due to garbage collection between runs.
    """
    if "client" not in st.session_state:
        st.session_state.client = None
    if "client_api_key" not in st.session_state:
        st.session_state.client_api_key = ""
        
    if st.session_state.client is None or st.session_state.client_api_key != api_key:
        st.session_state.client_api_key = api_key
        st.session_state.client = genai.Client(api_key=api_key)
        # Force fresh chat initialization if client changes
        st.session_state.chat = None
        st.session_state.chat_config_hash = ""
        
    return st.session_state.client

# LLM Helper Functions
def condense_query(client, model_name, query, chat_history):
    """
    Rewrites a follow-up query to be standalone using conversation history.
    This resolves stateless retrieval in stateful chats.
    """
    if not chat_history:
        return query
        
    history_str = ""
    for msg in chat_history[-5:]: # Look back up to 5 turns
        history_str += f"{msg['role']}: {msg['content']}\n"
        
    prompt = f"""Given the following conversation history and a follow-up user question, rewrite the follow-up question into a standalone question that contains all necessary context (resolving pronouns, acronyms, and implicit references).

If the user question is already standalone and does not need any context from the history (e.g., standard greetings, chitchat, or complete self-contained questions), return it exactly as-is. Do not add any introductory text, explanation, markdown formatting, or punctuation.

Conversation History:
{history_str}
Follow-up Question: "{query}"

Standalone Question:"""
    try:
        chat = client.chats.create(
            model=model_name,
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.9
            )
        )
        response = chat.send_message(prompt)
        condensed = response.text.strip()
        return condensed if condensed else query
    except Exception:
        return query

def get_persistent_chat(client, model_name, system_prompt, temperature, top_p):
    """
    Retrieves or initializes the persistent chat session, rebuilding it if config changes.
    """
    config_hash = hashlib.md5(f"{model_name}-{system_prompt}-{temperature}-{top_p}".encode()).hexdigest()
    
    if st.session_state.chat is None or st.session_state.chat_config_hash != config_hash:
        st.session_state.chat_config_hash = config_hash
        
        history_list = []
        if st.session_state.chat is not None:
            try:
                history_list = st.session_state.chat.get_history()
            except Exception:
                pass
                
        unified_static_instructions = f"""{system_prompt}

You are a precise, stateful document QA chatbot. The user has uploaded a document, and you can only answer questions related to it.

Rules:
1. GREETING/SMALLTALK: If the user says hello, hi, how are you, or asks who you are, respond with a friendly greeting and welcome them to ask questions about the document.
2. OFF-TOPIC: If the user asks a question completely unrelated to the document (e.g. cooking, coding, math), politely state that you are restricted to answering questions about the uploaded document.
3. AMBIGUOUS/PROBING: If the user's query is vague, incomplete, or lacks context in relation to previous messages (e.g. "why?", "tell me more", "explain that"), generate a polite, clear, and specific probing question based on the conversation history to clarify what they want to find.
4. DOCUMENT QUERY: If the user asks about the document, answer it using only the context provided in their latest message. Cite sources (e.g., Page X) where facts are found. If the information is not in the context, politely state that you cannot find it in the document.

You must remember the conversation history to handle follow-up questions (e.g. resolving pronouns like "she", "it", "they" based on previous turns) naturally.
"""
        st.session_state.chat = client.chats.create(
            model=model_name,
            history=history_list,
            config=types.GenerateContentConfig(
                system_instruction=unified_static_instructions,
                temperature=temperature,
                top_p=top_p
            )
        )
    return st.session_state.chat

def stream_chat_response(chat, query, context_chunks):
    """
    Sends the user query along with RAG context to the persistent chat and yields stream chunks.
    """
    context_str = ""
    for res in context_chunks:
        chunk = res["chunk"]
        context_str += f"[Source: {chunk['page_label']}]\n{chunk['text']}\n\n"
        
    formatted_message = f"""Context:
---
{context_str}
---

User Question: {query}
"""
    try:
        response_stream = chat.send_message_stream(formatted_message)
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error generating response: {e}"


# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "indexing_success" not in st.session_state:
    st.session_state.indexing_success = False
if "chat" not in st.session_state:
    st.session_state.chat = None
if "chat_config_hash" not in st.session_state:
    st.session_state.chat_config_hash = ""

# Layout Structure
st.title("📂 Document RAG Chatbot")
st.markdown("""
This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer questions based **only** on the document you upload.
It features automatic **intent classification** to filter off-topic queries and **probing** to clarify ambiguous questions.
""")

# Sidebar
st.sidebar.title("Configuration & Upload")

# API Configuration
st.sidebar.subheader("API Keys")
st.sidebar.text_input(
    "Gemini API Key", 
    value=os.environ.get("GEMINI_API_KEY", ""), 
    type="password",
    key="api_key_input",
    help="Enter your Gemini API key. You can get one for free from Google AI Studio."
)

api_key = get_api_key()

# Model Config
st.sidebar.subheader("Model Settings")
model_name = st.sidebar.selectbox(
    "Model Selection",
    options=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    index=0,
    help="Select the Gemini model to use for generation. 'gemini-2.5-flash' is recommended for speed and capability."
)

system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are a helpful and precise assistant. You answer queries using only the provided document.",
    help="Define the role and instructions for the QA model.",
    height=120
)

st.sidebar.subheader("Generation Settings")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1, help="Controls randomness: lower values are more deterministic, higher values are more creative.")
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05, help="Controls diversity via nucleus sampling.")

# RAG Config
st.sidebar.subheader("RAG Index Settings")
chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=5000, value=1000, step=100, help="Number of characters in each chunk.")
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=1000, value=200, step=50, help="Overlap between consecutive chunks.")
top_k = st.sidebar.slider("Top K Chunks to Retrieve", min_value=1, max_value=15, value=5, step=1, help="Number of matching context chunks to retrieve.")

# Document Upload
st.sidebar.subheader("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file (PDF, TXT, or DOCX)", 
    type=["pdf", "txt", "docx"],
    help="Upload the document you want to query. File content is indexed automatically."
)

# Process Uploaded File
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    current_hash = hashlib.md5(file_bytes).hexdigest()
    # If chunking parameters or file changed, re-index
    config_hash = hashlib.md5(f"{chunk_size}-{chunk_overlap}".encode()).hexdigest()
    total_hash = f"{current_hash}-{config_hash}"
    
    if (st.session_state.file_hash != total_hash) or (st.session_state.vector_store is None):
        if not api_key:
            st.sidebar.warning("⚠️ Enter a Gemini API Key to start indexing!")
        else:
            with st.sidebar.status("Processing and indexing document...", expanded=True) as status:
                try:
                    status.update(label="Reading document content...", state="running")
                    pages = parse_document(file_bytes, uploaded_file.name)
                    
                    status.update(label="Chunking text segments...", state="running")
                    chunks = chunk_text(pages, chunk_size, chunk_overlap)
                    st.session_state.chunks = chunks
                    
                    status.update(label="Generating embeddings and indexing...", state="running")
                    client = get_shared_client(api_key)
                    vector_store = InMemoryVectorStore()
                    vector_store.add_chunks(chunks, client)
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.file_hash = total_hash
                    st.session_state.indexing_success = True
                    status.update(label="Document indexed successfully!", state="complete")
                except Exception as e:
                    st.session_state.indexing_success = False
                    status.update(label=f"Failed indexing: {e}", state="error")
                    st.sidebar.error(f"Error details: {e}")
else:
    # Reset when file is removed
    st.session_state.vector_store = None
    st.session_state.chunks = []
    st.session_state.file_hash = None
    st.session_state.file_name = None
    st.session_state.indexing_success = False

# Sidebar Footer Options
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.session_state.chat = None
    st.session_state.chat_config_hash = ""
    st.rerun()

# Document Status Display
if st.session_state.indexing_success:
    st.sidebar.success(f"✓ **Indexed:** {st.session_state.file_name} ({len(st.session_state.chunks)} chunks)")
else:
    st.sidebar.info("ℹ️ Upload a document to begin chatting.")

# Chat Interface
# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("Show Sources"):
                for cit in message["citations"]:
                    st.markdown(f"**Source:** {cit['page_label']} (Score: {cit['score']:.2f})")
                    st.caption(cit['text'])

# New Query Handler
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Pre-checks
    if not api_key:
        st.error("⚠️ Please enter your Gemini API Key in the sidebar to interact with the chatbot.")
    elif not st.session_state.indexing_success or st.session_state.vector_store is None:
        st.error("⚠️ Please upload and index a document in the sidebar before asking questions.")
    else:
        # Display user input
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Response Container
        with st.chat_message("assistant"):
            client = get_shared_client(api_key)
            
            # Show a spinner during initial search and model warm up
            with st.spinner("Searching document & connecting..."):
                # Rewrite query if it's a follow-up turn (resolves pronouns & context)
                search_query = condense_query(
                    client=client,
                    model_name=model_name,
                    query=prompt,
                    chat_history=st.session_state.messages[:-1]
                )
                
                # Proactively search document context using the rewritten standalone query
                search_results = st.session_state.vector_store.search(
                    query_text=search_query,
                    client=client,
                    k=top_k
                )
                
                # Retrieve or initialize the persistent chat session
                chat_session = get_persistent_chat(
                    client=client,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Initialize unified streaming response generator
                response_generator = stream_chat_response(
                    chat=chat_session,
                    query=prompt,
                    context_chunks=search_results
                )
                
                # Fetch first token to close spinner once stream is active
                try:
                    first_chunk = next(response_generator)
                except StopIteration:
                    first_chunk = ""
            
            # Wrapper generator to yield the first chunk and then the rest
            def final_stream_generator():
                if first_chunk:
                    yield first_chunk
                for chunk in response_generator:
                    yield chunk
            
            # Stream response in real-time with typewriter effect
            response_text = st.write_stream(final_stream_generator())
            
            # Extract and show citations if search results exist and response is not off-topic
            citations = []
            if search_results and "restricted to answering" not in response_text.lower():
                for res in search_results:
                    citations.append({
                        "page_label": res["chunk"]["page_label"],
                        "text": res["chunk"]["text"],
                        "score": res["score"]
                    })
                
                with st.expander("Show Sources"):
                    for cit in citations:
                        st.markdown(f"**Source:** {cit['page_label']} (Score: {cit['score']:.2f})")
                        st.caption(cit['text'])
                        
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "citations": citations
            })
