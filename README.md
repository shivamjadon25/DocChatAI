# 📂 Document RAG Chatbot with Intent Routing & Probing

This is a complete, feature-rich **Retrieval-Augmented Generation (RAG) Chatbot** designed to answer user questions using only the content of an uploaded document (PDF, TXT, or DOCX). It uses Gemini's free-tier models and embeddings, and is built on top of a customized Streamlit frontend.

---

## 🚀 Key Features

1. **Strict Context Adherence (RAG)**: 
   - Uses a RAG pipeline that retrieves top-K document chunks matching the user query.
   - Constrains responses to the scope of the document. If information isn't found in the text, it politely declines to speculate.

2. **Intent Classification**:
   - Classifies every user query into one of four categories:
     - `GREETING_OR_CHITCHAT`: Friendly responses welcoming the user.
     - `DOCUMENT_QUERY`: Triggers vector retrieval and context-based answering.
     - `OFF_TOPIC`: Politely restricts queries to the document scope.
     - `AMBIGUOUS_OR_INSUFFICIENT`: Flags vague inputs.

3. **Probing / Clarifying Questions**:
   - If an input is flagged as `AMBIGUOUS_OR_INSUFFICIENT`, the chatbot probes the user with a tailored clarifying question to specify what information they are seeking.

4. **ChatGPT-Like UI**:
   - Complete chat history formatting.
   - Clean collapsible sources section (**Show Sources**) with cosine similarity scores and excerpt text under each answer.
   - Responsive sidebar controls.

5. **Prompt & LLM Configuration**:
   - Customizable **System Prompts**.
   - Adjustable parameters: **Model Selection** (`gemini-2.5-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`), **Temperature**, **Top P**, **Chunk Size**, **Chunk Overlap**, and **Top-K Retrieval count**.

6. **100% Free Dependencies**:
   - Uses Gemini API's free tier for text embeddings (`gemini-embedding-001`) and generation.
   - Custom in-memory NumPy vector store to avoid heavy compiled binary dependencies.

---

## 🛠️ Architecture

```
                       ┌──────────────────────┐
                       │  Uploaded Document   │
                       └──────────┬───────────┘
                                  │ (PDF/DOCX/TXT)
                                  ▼
                        [ Text Extraction ]
                                  │
                                  ▼
                        [ Chunking Text ]
                                  │
                                  ▼
                     [ Gemini Embeddings API ]
                                  │ (gemini-embedding-001)
                                  ▼
                 ┌─────────────────────────────────┐
                 │  InMemoryVectorStore (NumPy)    │
                 └───────────────▲─────────────────┘
                                 │ Search
                                 │
     ┌─────────────┐             │          ┌───────────────────────┐
     │  User Chat  ├─────────────┼─────────►│  Intent Classifier   │
     │  Interface  │◄────────────┼──────────┤ (Greeting/Off-Topic/  │
     └─────────────┘             │          │  Ambiguous/Query)     │
                                 │          └───────────┬───────────┘
                                 │                      │
                   ┌─────────────┴────────────┐         │
                   │    If DOCUMENT_QUERY     │◄────────┘
                   │  Retrieve & LLM Generation│
                   └──────────────────────────┘
```

---

## 💻 Setup & Installation

### 1. Prerequisites
Make sure you have Python 3.10+ and pip installed.

### 2. Install Dependencies
Install all package requirements in your environment:
```bash
pip install -r requirements.txt
```
*(Note: If you encounter an externally-managed-environment error, you can run `pip install -r requirements.txt --break-system-packages` or execute it inside a virtual environment).*

### 3. Get a Gemini API Key
Obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 4. Running the Chatbot
Start the Streamlit application using:
```bash
streamlit run app.py
```
This will start a local server and open the interface in your default web browser (typically at `http://localhost:8501`).

---

## 📁 File Structure

- `app.py`: Main Streamlit app containing UI layouts, chat history, state tracking, and LLM orchestration.
- `rag_engine.py`: Text parser (PDF, DOCX, TXT), paragraph grouping logic, slider chunking, and the custom cosine-similarity vector store.
- `requirements.txt`: Project dependencies list.
