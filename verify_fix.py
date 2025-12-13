import sys
import os

# Add root to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import load_model_artifacts

try:
    data = load_model_artifacts()
    print("SUCCESS: Loaded", len(data), "recipes")
except Exception as e:
    print("FAILURE:", e)
