from llama_index.embeddings.ollama import OllamaEmbedding
import sys

EMBEDDING_MODEL = "nomic-embed-text"

# Create a long string (approx 6000 tokens)
long_text = "Hello world! " * 4000 

print(f"Testing with text length: {len(long_text)} characters")

try:
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL, 
        additional_kwargs={"num_ctx": 8192}
    )
    embedding = embed_model.get_text_embedding(long_text)
    print("Success! Embedding generated for long text.")
except Exception as e:
    print(f"Failed: {e}")
    sys.exit(1)
