# Culinary Compass - Let's get cooking

Culinary Compass is an intelligent recipe recommendation system that suggests the best meals you can make from the ingredients you already have.  
It combines NLP-based ingredient cleaning, **TF-IDF similarity**, and **semantic embeddings**, wrapped in a **FastAPI backend** with a **Streamlit frontend**.

### Features
- Ingredient cleaning & normalization pipeline
- Hybrid recommendation engine:
    - TF-IDF similarity (keyword matching)
    - Semantic similarity using Sentence Transformers
    - Weighted hybrid scoring
- FastAPI-ready backend architecture with preloaded model artifacts
- Streamlit UI for interactive recipe search
- Basic unit tests for core logic

**Planned improvements:**
- Ingredient overlap scoring
- Missing-ingredient detection
- Scoring penalties for recipes requiring many extra ingredients