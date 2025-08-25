import os
import pickle
import time
import glob
import re
from typing import List, Tuple

import numpy as np
from PyPDF2 import PdfReader
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
PICKLE_PATH = "pdf_index.pkl"
TOP_K = 4

client = OpenAI(api_key="Your API")

pdf_chunks: List[str] = []
pdf_embeddings: np.ndarray | None = None
pdf_file = None

def find_pdf() -> str | None:
    pdfs = glob.glob("*.pdf")
    return pdfs[0] if pdfs else None

def read_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    all_text = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        all_text.append(txt)
    return "\n".join(all_text)

def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks

def embed_texts(texts: List[str], batch_size: int = 50) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            all_embeddings.extend([d.embedding for d in resp.data])
            time.sleep(1.2)
        except Exception as e:
            print(f"❌ Failed on batch {i}-{i+batch_size}: {e}")
            raise
    return np.array(all_embeddings, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

def retrieve_context(question: str, k: int = TOP_K) -> Tuple[str, List[int]]:
    global pdf_embeddings, pdf_chunks
    if pdf_embeddings is None or len(pdf_chunks) == 0:
        return "", []
    q_emb = embed_texts([question])[0:1]
    sims = cosine_sim(q_emb, pdf_embeddings)[0]
    idx = np.argsort(-sims)[:k]
    selected = [pdf_chunks[i] for i in idx]
    return "\n\n---\n\n".join(selected), idx.tolist()

def build_pdf_index():
    global pdf_chunks, pdf_embeddings, pdf_file
    pdf_file = find_pdf()
    if not pdf_file:
        print("⚠️ No PDF found in current directory.")
        return
    if os.path.exists(PICKLE_PATH):
        try:
            with open(PICKLE_PATH, "rb") as f:
                data = pickle.load(f)
            pdf_chunks = data["chunks"]
            pdf_embeddings = data["embeddings"]
            print(f"✅ Loaded cached PDF index with {len(pdf_chunks)} chunks.")
            return
        except Exception:
            pass
    print("📄 Reading PDF & building embeddings...")
    text = read_pdf(pdf_file)
    chunks = split_text(text)
    if not chunks:
        return
    embeds = embed_texts(chunks)
    pdf_chunks = chunks
    pdf_embeddings = embeds
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump({"chunks": pdf_chunks, "embeddings": pdf_embeddings}, f)
    print(f"✅ Indexed {len(pdf_chunks)} chunks from PDF.")
