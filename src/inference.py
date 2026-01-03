import json
import os
import logging
import pickle
import joblib
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_artifacts():
    """
    Loads all heavy Machine Learning artifacts into memory at startup.
    
    Expected Files (see DATA_CONTRACT.md for details):
    - data/processed/cleaned.json: Recipe database with name, ingredients, directions
    - models/tfidf_vectorizer.pkl: Trained TF-IDF vectorizer
    - models/tfidf_matrix.npz: TF-IDF feature matrix (sparse)
    - models/embeddings.npy: Pre-computed semantic embeddings
    
    Returns:
        dict: Dictionary containing all loaded artifacts with keys:
            - recipes: List of recipe dictionaries
            - tfidf_vectorizer: Fitted TfidfVectorizer
            - tfidf_matrix: Sparse TF-IDF matrix
            - recipe_embeddings: NumPy array of embeddings
            - bert_model: SentenceTransformer model for query encoding
    
    Raises:
        FileNotFoundError: If any required file is missing
        ValueError: If loaded data has incorrect format
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data", "processed")

    logger.info("LOG: Loading model artifacts... This may take a moment.")

    # Define all required files
    required_files = {
        "cleaned.json": os.path.join(data_dir, "cleaned.json"),
        "tfidf_vectorizer.pkl": os.path.join(models_dir, "tfidf_vectorizer.pkl"),
        "tfidf_matrix.npz": os.path.join(models_dir, "tfidf_matrix.npz"),
        "embeddings.npy": os.path.join(models_dir, "embeddings.npy")
    }

    # Check all files exist before attempting to load
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"  - {name} (expected at: {path})")
    
    if missing_files:
        error_msg = (
            "CRITICAL: Missing required model artifacts!\n"
            "The following files are missing:\n" + "\n".join(missing_files) + "\n\n"
            "Please ensure the preprocessing pipeline has been run and all artifacts "
            "are generated. See DATA_CONTRACT.md for details."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # 1. Load the Recipe Database (Metadata)
        logger.info("  Loading recipe database...")
        with open(required_files["cleaned.json"], "r", encoding="utf-8") as f:
            recipes = json.load(f)
        
        # Validate recipe format
        if not isinstance(recipes, list):
            raise ValueError("cleaned.json must contain a JSON array of recipes")
        if len(recipes) == 0:
            raise ValueError("cleaned.json contains no recipes")
        
        # Check first recipe has required fields
        required_fields = ["name", "ingredients", "directions"]
        first_recipe = recipes[0]
        missing_fields = [f for f in required_fields if f not in first_recipe]
        if missing_fields:
            raise ValueError(f"Recipe missing required fields: {missing_fields}")
        
        logger.info(f"Loaded {len(recipes)} recipes")

        # 2. Load TF-IDF Vectorizer (The Keyword Translator)
        logger.info("Loading TF-IDF vectorizer...")
        tfidf_vectorizer = joblib.load(required_files["tfidf_vectorizer.pkl"])
        logger.info("TF-IDF vectorizer loaded")

        # 3. Load TF-IDF Matrix (The Keyword Scoreboard)
        logger.info("Loading TF-IDF matrix...")
        tfidf_matrix = scipy.sparse.load_npz(required_files["tfidf_matrix.npz"])
        
        # Validate matrix dimensions
        if tfidf_matrix.shape[0] != len(recipes):
            raise ValueError(
                f"TF-IDF matrix has {tfidf_matrix.shape[0]} rows but there are "
                f"{len(recipes)} recipes. These must match!"
            )
        logger.info(f"TF-IDF matrix loaded: shape {tfidf_matrix.shape}")

        # 4. Load Pre-calculated Embeddings (The Semantic Map)
        logger.info("Loading embeddings...")
        recipe_embeddings = np.load(required_files["embeddings.npy"])
        
        # Validate embedding dimensions
        if recipe_embeddings.shape[0] != len(recipes):
            raise ValueError(
                f"Embeddings have {recipe_embeddings.shape[0]} rows but there are "
                f"{len(recipes)} recipes. These must match!"
            )
        logger.info(f"Embeddings loaded: shape {recipe_embeddings.shape}")

        # 5. Load the Semantic Model (The Brain)
        logger.info("Loading BERT model for query encoding...")
        bert_model = SentenceTransformer('all-MiniLM-L6-v2') 
        logger.info("BERT model loaded")

        logger.info("LOG: Success! All models loaded and validated.")
        
        return {
            "recipes": recipes,
            "tfidf_vectorizer": tfidf_vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "recipe_embeddings": recipe_embeddings,
            "bert_model": bert_model
        }

    except FileNotFoundError:
        raise  # Already handled above
    except ValueError as e:
        logger.error(f"CRITICAL: Data validation failed: {e}")
        raise e
    except Exception as e:
        logger.error(f"CRITICAL: Unexpected error while loading models: {e}")
        raise e


def find_similar_recipes(user_ingredients, artifacts, top_k=5):
    """
    Calculates hybrid similarity (Keyword + Semantic) and returns top matches.
    
    Args:
        user_ingredients (List[str]): List of ingredients from user.
        artifacts (dict): The dictionary returned by load_model_artifacts().
        top_k (int): Number of recipes to return.
    """
    
    # Unpack the artifacts for easier use
    recipes = artifacts["recipes"]
    tfidf_vectorizer = artifacts["tfidf_vectorizer"]
    tfidf_matrix = artifacts["tfidf_matrix"]
    recipe_embeddings = artifacts["recipe_embeddings"]
    bert_model = artifacts["bert_model"]

    # 1. Preprocess the User Query
    # Join list ["tomato", "cheese"] -> string "tomato cheese"
    query_text = " ".join(user_ingredients)
    logger.info(f"Processing query: {query_text}")

    # ---------------------------------------------------------
    # PART A: Keyword Similarity (TF-IDF)
    # ---------------------------------------------------------
    # Convert query to vector
    query_tfidf = tfidf_vectorizer.transform([query_text])
    
    # Calculate Cosine Similarity against all recipes
    # Result is a list of scores: [0.1, 0.5, 0.9, ...]
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # ---------------------------------------------------------
    # PART B: Semantic Similarity (Embeddings)
    # ---------------------------------------------------------
    # Convert query to vector
    query_embedding = bert_model.encode([query_text])
    
    # Calculate Cosine Similarity
    semantic_scores = cosine_similarity(query_embedding, recipe_embeddings).flatten()

    # ---------------------------------------------------------
    # PART C: Hybrid Scoring 
    # ---------------------------------------------------------
    # We combine both scores. 
    # Alpha controls the balance. 0.5 means 50% keyword, 50% meaning.
    alpha = 0.5 
    final_scores = (tfidf_scores * alpha) + (semantic_scores * (1 - alpha))

    # ---------------------------------------------------------
    # PART D: Sorting & Formatting
    # ---------------------------------------------------------
    # Get the indices of the top_k highest scores
    # argsort sorts low-to-high, so we take the last k and reverse them
    top_indices = final_scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        score = final_scores[idx]
        
        # Filter out bad matches (Optional)
        if score < 0.1: 
            continue

        recipe = recipes[idx]
        results.append({
            "name": recipe["name"],
            "ingredients": recipe["ingredients"],
            "directions": recipe.get("directions", []),
            "match_score": float(score) # Convert numpy float to python float
        })

    logger.info(f"Found {len(results)} matches.")
    return results