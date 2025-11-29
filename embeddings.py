import json
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load FAQ data
file_path = "HDFC_Faq.txt"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Build index
texts = []
metadata_list = []

for item in data:
    question = item["question"]
    answer = item["answer"]
    texts.append(question)
    metadata_list.append({
        "question": question,
        "answer": answer,
        "full_block": f"Question: {question}\nAnswer: {answer}"
    })

vector_store = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadata_list
)

# Save to disk
vector_store.save_local("./faiss_index")
print("✓ FAISS index saved to ./faiss_index")
