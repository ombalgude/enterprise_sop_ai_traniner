
import os
import sys
import shutil
from multimodal_rag import ingest_documents, DATA_DIR, CHROMA_PATH

def verify_stale_fix():
    print("Verifying stale reference fix during ingestion...")
    
    # 1. Setup DATA_DIR with a dummy file if needed
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # We need a real PDF for fitz to not fail if we want to run the full ingest
    # But we can also just verify it doesn't crash on the clearing logic.
    # Let's try calling it.
    
    try:
        print("First ingestion call...")
        ingest_documents(DATA_DIR)
        print("SUCCESS: First ingestion call completed.")
        
        print("Second ingestion call (should trigger clearing logic)...")
        ingest_documents(DATA_DIR)
        print("SUCCESS: Second ingestion call completed without 'collection does not exist' error.")
        
        return True
    except Exception as e:
        print(f"FAILURE: Ingestion error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if verify_stale_fix():
        sys.exit(0)
    else:
        sys.exit(1)
