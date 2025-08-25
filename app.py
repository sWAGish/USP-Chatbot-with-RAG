import os
import time
import io
import pickle
import base64
import threading
import glob
import re
from typing import List, Dict, Any, Tuple

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from gtts import gTTS
from PyPDF2 import PdfReader
from openai import OpenAI

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "Your API"
EMBED_MODEL   = "text-embedding-3-small"
PRIMARY_CHAT_MODEL = "gpt-4o-mini"
RAG_CHAT_MODEL     = "gpt-4o-mini"
VISION_MODEL  = "gpt-4o"
PICKLE_PATH   = "pdf_index.pkl"
TOP_K         = 4
RESET_TIMEOUT_SECONDS = 60

# ---------------- INIT ----------------
client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)

sessions: Dict[str, Dict[str, Any]] = {}
pdf_chunks: List[str] = []
pdf_embeddings: np.ndarray | None = None
pdf_file = None

# ---------------- UTILS ----------------
def now_ts() -> float:
    return time.time()

def find_pdf() -> str | None:
    pdfs = glob.glob("*.pdf")
    return pdfs[0] if pdfs else None

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

def read_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    all_text = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        all_text.append(txt)
    return "\n".join(all_text)

def embed_texts(texts: List[str], batch_size: int = 50) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            all_embeddings.extend([d.embedding for d in resp.data])
            time.sleep(1.2)  # slight delay to avoid hitting TPM limits
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

def format_bullets(answer: str) -> str:
    answer = re.sub(r'(?<!\n)- ', r'\n- ', answer)
    return answer.strip()

def user_wants_table(question: str) -> bool:
    q = question.lower()
    triggers = ["table", "tabular", "comparison table", "tabulated"]
    return any(t in q for t in triggers)

def need_rag(question: str, llm_answer: str, history: List[Dict[str, str]]) -> bool:
    q = question.lower()
    triggers = ["from pdf", "more detail", "technical data", "data sheet"]
    if any(t in q for t in triggers):
        return True
    return False

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

def chat_completion(model: str, messages: List[Dict[str, str]], max_tokens=700, temperature=0.2):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

@app.route("/chat", methods=["POST"])
def chat():
    try:
        session_id = request.remote_addr or "anon"
        now = now_ts()
        session_reset = False

        if session_id not in sessions:
            sessions[session_id] = {"last_active": now, "history": []}
        else:
            if now - sessions[session_id]["last_active"] > RESET_TIMEOUT_SECONDS:
                sessions[session_id]["history"] = []
                session_reset = True
        sessions[session_id]["last_active"] = now

        # IMAGE FLOW
        if request.content_type.startswith("multipart/form-data"):
            image_file = request.files.get("image")
            question = request.form.get("question", "Describe this image")
            if not image_file:
                return jsonify({"answer": "Missing image."}), 400

            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            resp = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user",
                           "content": [{"type": "text", "text": question},
                                       {"type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}]
            )
            answer = resp.choices[0].message.content.strip()
            return jsonify({"answer": format_bullets(answer), "session_reset": session_reset})

        # TEXT FLOW
        data = request.get_json(force=True, silent=True) or {}
        question = data.get("question", "").strip()
        from_mic = data.get("voice", False)
        if not question:
            return jsonify({"answer": "Please enter a question."}), 400

        history = sessions[session_id]["history"][-6:]
        primary_prompt = (
            "You are a helpful assistant for USP Polymers. "
            "When the user requests a table, output valid HTML <table> code."
        )
        messages_primary = [{"role": "system", "content": primary_prompt}]
        messages_primary.extend(history)
        messages_primary.append({"role": "user", "content": question})

        resp_primary = chat_completion(PRIMARY_CHAT_MODEL, messages_primary)
        answer_primary = format_bullets(resp_primary.choices[0].message.content.strip())

        do_rag = need_rag(question, answer_primary, history)
        final_answer = answer_primary

        if do_rag and pdf_embeddings is not None:
            context, _ = retrieve_context(question)
            rag_prompt = (
                "Enhance the answer using this PDF context:\n" + context
            )
            resp_rag = chat_completion(RAG_CHAT_MODEL,
                                       [{"role": "system", "content": rag_prompt},
                                        {"role": "user", "content": question}])
            final_answer = format_bullets(resp_rag.choices[0].message.content.strip())

        if user_wants_table(question) and "<table" not in final_answer.lower():
            final_answer = (
                "<table border='1'><tr><th>Response</th></tr>"
                f"<tr><td>{final_answer}</td></tr></table>"
            )

        sessions[session_id]["history"].append({"role": "user", "content": question})
        sessions[session_id]["history"].append({"role": "assistant", "content": final_answer})

        result = {"answer": final_answer, "session_reset": session_reset}
        if from_mic:
            try:
                tts = gTTS(final_answer)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                result["audio"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as tts_err:
                print("TTS error:", tts_err)
        return jsonify(result)
    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"answer": "Sorry, something went wrong."}), 500

if __name__ == "__main__":
    build_pdf_index()
    app.run(debug=True)
