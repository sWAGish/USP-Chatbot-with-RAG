import openai
import requests
from bs4 import BeautifulSoup
import uuid
import json

# Your OpenAI & Supabase credentials
openai.api_key = "Your API"
supabase_url = "https://sjytuhoeygvnsvkhhhnk.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNqeXR1aG9leWd2bnN2a2hoaG5rIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE5NTYwODIsImV4cCI6MjA2NzUzMjA4Mn0.HnFlc7exjYJ0SMF8pxbT6rNMkaH1Q-bkEFl82n0rg0s"

headers = {
    "apikey": supabase_key,
    "Authorization": f"Bearer {supabase_key}",
    "Content-Type": "application/json"
}

# USP Polymers product page URLs
urls = {
    "wipers": "https://usppolymers.in/wipers/",
    "toys": "https://usppolymers.in/toys-and-sports-goods/",
    "gaskets": "https://usppolymers.in/gasket-and-seals/"
}

# Extracts readable text from web page
def extract_text_from_url(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    [s.extract() for s in soup(["script", "style", "noscript"])]
    text = soup.get_text(separator="\n")
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

# Break text into manageable chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Embed and insert into Supabase
def embed_and_insert(chunks, source):
    for chunk in chunks:
        embedding = openai.Embedding.create(
            input=chunk,
            model="text-embedding-3-small"
        )["data"][0]["embedding"]

        data = {
            "id": str(uuid.uuid4()),
            "content": chunk,
            "embedding": embedding,
            "source": source
        }

        response = requests.post(
            f"{supabase_url}/rest/v1/documents",
            headers=headers,
            data=json.dumps(data)
        )
        print(f"✅ Inserted chunk from {source}: {response.status_code}")

# Loop over all pages and process them
for label, url in urls.items():
    print(f"🔍 Processing {label} page...")
    text = extract_text_from_url(url)
    chunks = chunk_text(text)
    embed_and_insert(chunks, url)
