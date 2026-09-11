import io
import numpy as np
import pypdf
from docx import Document

def parse_pdf(file_bytes):
    pages = []
    pdf_file = io.BytesIO(file_bytes)
    reader = pypdf.PdfReader(pdf_file)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "text": text,
                "page_label": f"Page {i + 1}"
            })
    return pages

def parse_docx(file_bytes):
    pages = []
    docx_file = io.BytesIO(file_bytes)
    doc = Document(docx_file)
    
    current_text = []
    current_len = 0
    page_count = 1
    
    for i, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        if not text:
            continue
        current_text.append(text)
        current_len += len(text)
        
        if current_len >= 2000:
            pages.append({
                "text": "\n".join(current_text),
                "page_label": f"Section {page_count}"
            })
            current_text = []
            current_len = 0
            page_count += 1
            
    if current_text:
        pages.append({
            "text": "\n".join(current_text),
            "page_label": f"Section {page_count}"
        })
        
    return pages

def parse_txt(file_bytes):
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except Exception:
            text = "Error decoding file content as text."
            
    pages = []
    text_len = len(text)
    chunk_size = 2000
    page_count = 1
    
    for i in range(0, text_len, chunk_size):
        chunk = text[i:i+chunk_size]
        pages.append({
            "text": chunk,
            "page_label": f"Page {page_count}"
        })
        page_count += 1
        
    return pages

def parse_document(file_bytes, file_name):
    ext = file_name.split(".")[-1].lower()
    if ext == "pdf":
        return parse_pdf(file_bytes)
    elif ext in ["docx", "doc"]:
        return parse_docx(file_bytes)
    else:
        return parse_txt(file_bytes)

def chunk_text(pages, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for page in pages:
        text = page["text"]
        page_label = page["page_label"]
        if not text.strip():
            continue
        
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page_label": page_label
                })
            if end == text_len:
                break
            start += (chunk_size - chunk_overlap)
    return chunks

class InMemoryVectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = None

    def add_chunks(self, chunks, client):
        if not chunks:
            return
        
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch_texts
            )
            for emb in response.embeddings:
                all_embeddings.append(emb.values)
        
        self.embeddings = np.array(all_embeddings, dtype=np.float32)

    def search(self, query_text, client, k=5):
        if not self.chunks or self.embeddings is None:
            return []
        
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query_text
        )
        query_emb = np.array(response.embeddings[0].values, dtype=np.float32)
        
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized_embeddings = self.embeddings / norms
        
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_emb / query_norm
        
        similarities = np.dot(normalized_embeddings, normalized_query)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx])
            })
        return results
