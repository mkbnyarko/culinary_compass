# Culinary Compass - Let's get cooking

Culinary Compass is an intelligent recipe recommendation system that suggests the best meals you can make from the ingredients you already have.  
It uses a **hybrid similarity** approach combining keyword-based matching (TF-IDF) and semantic similarity (Sentence Transformers), built on top of a robust **multi-stage ingredient preprocessing pipeline**.

### Features
- Multi-stage ingredient preprocessing pipeline
    - Dataset deduplication
    - Rule-based + NLP-based ingredient cleaning
    - Phrase-level ingredient extraction
    - Lemmatization-based normalization
- Hybrid recommendation engine:
    - TF-IDF similarity (keyword matching)
    - Semantic similarity using Sentence Transformers
    - Weighted hybrid scoring
- FastAPI-ready backend architecture with preloaded model artifacts
    - Preloaded: vectorizer, matrix, and embeddings
- Streamlit UI for interactive recipe search
- Basic unit tests for core logic

**Planned improvements:**
- Ingredient overlap and missing-ingredient scoring
- Scoring penalties for recipes requiring many extra ingredients

🎯 Why This Project?
This project demonstrates:
- Real-world ingredient normalization challenges
- Rule-based + NLP hybrid preprocessing
- Hybrid recommendation systems
- Artifact-driven ML engineering patterns
- Clean API-first design

The emphasis is on robust data handling and system design, not just model training.