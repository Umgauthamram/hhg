import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.config import settings

def test_config_defaults():
    assert settings.GROQ_MODEL == "llama-3.1-8b-instant"
    assert settings.SIMILARITY_THRESHOLD == 0.45
    assert settings.TOP_K == 3
    assert settings.COLLECTION_NAME == "msmarco_xi"
    assert settings.PORT == 8000

def test_directory_structure():
    expected_dirs = [
        "app",
        "app/stt",
        "app/chunking",
        "app/vector_store",
        "app/harness",
        "app/llm",
        "benchmark",
        "tests",
    ]
    for d in expected_dirs:
        path = os.path.join(BASE_DIR, d)
        assert os.path.isdir(path), f"Directory {d} does not exist"

if __name__ == "__main__":
    test_config_defaults()
    test_directory_structure()
    print("Phase 1 test passed successfully!")
