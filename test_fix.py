
import os
import sys
from multimodal_rag import setup_query_engine, CHROMA_PATH

def verify_fix():
    print("Verifying setup_query_engine return value...")
    
    # Ensure CHROMA_PATH exists for setup_query_engine to work as expected in app.py logic
    if not os.path.exists(CHROMA_PATH):
        print(f"DEBUG: {CHROMA_PATH} does not exist. Creating empty dir for test.")
        os.makedirs(CHROMA_PATH, exist_ok=True)
    
    try:
        engine = setup_query_engine()
        if engine is not None:
            print("SUCCESS: setup_query_engine returned a value.")
            return True
        else:
            print("FAILURE: setup_query_engine still returns None.")
            return False
    except Exception as e:
        print(f"ERROR: An exception occurred during verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if verify_fix():
        sys.exit(0)
    else:
        sys.exit(1)
