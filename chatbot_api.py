from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import requests
import json

app = Flask(__name__)
CORS(app)

# OpenAI Key
openai.api_key = "Your API"

# Supabase
SUPABASE_URL = "https://sjytuhoeygvnsvkhhhnk.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNqeXR1aG9leWd2bnN2a2hoaG5rIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5NTYwODIsImV4cCI6MjA2NzUzMjA4Mn0.HnFlc7exjYJ0SMF8pxbT6rNMkaH1Q-bkEFl82n0rg0s"

headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}",
    "Content-Type": "application/json"
}

# Generate embedding
def get_embedding(text):
    result = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"
    )
    return result['data'][0]['embedding']

# Query Supabase
def query_supabase(embedding):
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_documents",
        headers=headers,
        json={
            "query_embedding": embedding,
            "match_threshold": 0.75,
            "match_count": 5
        }
    )
    return response.json()

# Chat API
@app.route("/chat", methods=["POST"])
def chat():
    user_question = request.json.get("question", "")
    query_embedding = get_embedding(user_question)
    results = query_supabase(query_embedding)
    context = "\n".join([r["content"] for r in results if "content" in r])

    # Compose answer with GPT
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for a polymer product website. Answer only based on the given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_question}"}
        ]
    )["choices"][0]["message"]["content"]

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
