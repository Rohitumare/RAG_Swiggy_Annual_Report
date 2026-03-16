import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------
# CONFIG
# -------------------------------
PDF_PATH = "Annual-Report-FY-2023-24 (1).pdf"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# -------------------------------
# 1. LOAD PDF
# -------------------------------
print("Loading PDF...")
reader = PdfReader(PDF_PATH)

full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

# -------------------------------
# 2. TEXT CHUNKING
# -------------------------------
print("Chunking text...")
chunks = []
start = 0
while start < len(full_text):
    end = start + CHUNK_SIZE
    chunks.append(full_text[start:end])
    start += CHUNK_SIZE - CHUNK_OVERLAP

# -------------------------------
# 3. EMBEDDINGS
# -------------------------------
print("Creating embeddings...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
embeddings = embedder.encode(chunks, convert_to_numpy=True)

# -------------------------------
# 4. FAISS VECTOR STORE
# -------------------------------
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -------------------------------
# 5. LOAD LLM
# -------------------------------
print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)

qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512
)

# -------------------------------
# 6. QUESTION ANSWERING FUNCTION
# -------------------------------
def answer_question(question):
    # Embed query
    query_embedding = embedder.encode([question], convert_to_numpy=True)

    # Retrieve relevant chunks
    distances, indices = index.search(query_embedding, TOP_K)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    # Build prompt (STRICT grounding)
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
Answer the question strictly using the context below.
If the answer is not present, say "Information not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    result = qa_pipeline(prompt)[0]["generated_text"]
    return result, retrieved_chunks

# -------------------------------
# 7. CLI INTERFACE
# -------------------------------
print("\n===== Swiggy Annual Report Q&A =====")
print("Ask questions strictly based on the Swiggy Annual Report.")
print("Type 'exit' to quit.\n")

while True:
    query = input("Your Question: ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    answer, sources = answer_question(query)

    print("\nAnswer:")
    print(answer)

    print("\nSupporting Context:")
    for i, src in enumerate(sources):
        print(f"\n--- Source {i+1} ---")
        print(src[:400])
