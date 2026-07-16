"""Quick smoke test — verifies the fully local pipeline works."""
from pipeline.indexing import get_embedder
from pipeline.retriever import Retriever
from pipeline.generator import AnswerGenerator

print("=== Testing fully local RAG pipeline ===")

print("[1] Loading Ollama embedder (nomic-embed-text)...")
emb = get_embedder()
vec = emb.encode(["What is a planogram?"])[0]
print(f"    Embedding dim: {len(vec)}")

print("[2] Loading retriever + running query...")
r = Retriever()
chunks = r.retrieve("What is a SKU?")
print(f"    Retrieved {len(chunks)} chunks")
if chunks:
    print(f"    Top chunk: {chunks[0]['text'][:120]}...")

print("[3] Generating answer with Ollama LLM...")
g = AnswerGenerator()
answer = g.generate("What is a SKU?", chunks[:3])
print(f"    Answer: {answer[:200]}...")

print()
print("SUCCESS — fully local, no HuggingFace, no API keys!")
