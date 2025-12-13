import json
import os
import logging

logger = logging.getLogger(__name__)

def load_model_artifacts():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "processed", "dummy_recipes.json")
    
    logger.info(f"Loading artifacts from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    with open(data_path, "r") as f:
        try:
            recipes = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {data_path}: {e}")
            raise

    logger.info("Artifacts loaded successfully.")
    return recipes

def find_similar_recipes(user_ingredients, all_recipes):
    results = []
    for recipe in all_recipes[:2]:
        results.append({
            "name": recipe["name"],
            "ingredients": recipe["ingredients"],
            "directions": recipe["directions"],
            "url": recipe.get("url"),
            "match_score": 0.88
        })
    return results