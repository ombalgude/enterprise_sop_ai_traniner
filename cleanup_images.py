import os
import shutil
import chromadb
from multimodal_rag import CHROMA_PATH, DATA_DIR

def cleanup():
    print("Starting cleanup of reference images...")
    
    # 1. Clear ChromaDB image collection
    if os.path.exists(CHROMA_PATH):
        try:
            db = chromadb.PersistentClient(path=CHROMA_PATH)
            collections = db.list_collections()
            collection_names = [c.name for c in collections]
            
            if "image_collection" in collection_names:
                print("Deleting 'image_collection' from ChromaDB...")
                db.delete_collection("image_collection")
                print("Image collection deleted.")
            else:
                print("'image_collection' not found in ChromaDB.")
        except Exception as e:
            print(f"Error accessing ChromaDB: {e}")
    else:
        print(f"ChromaDB path {CHROMA_PATH} does not exist.")

    # 2. Delete image files from data directory
    if os.path.exists(DATA_DIR):
        print(f"Scanning {DATA_DIR} for image files...")
        image_extensions = (".png", ".jpg", ".jpeg")
        files = os.listdir(DATA_DIR)
        removed_count = 0
        
        for file in files:
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(DATA_DIR, file)
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {file}: {e}")
        
        print(f"Removed {removed_count} image files from {DATA_DIR}.")
    else:
        print(f"Data directory {DATA_DIR} does not exist.")

    print("Cleanup complete!")

if __name__ == "__main__":
    cleanup()
