import sys
import os
import logging
from typing import List

# Add parent directory to sys.path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.inference import load_model_artifacts, find_similar_recipes 
from schemas import IngredientQuery, RecipeResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global storage for ML artifacts
artifacts_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - loads ML artifacts on startup."""
    global artifacts_db
    logger.info("Starting up application...")
    try:
        artifacts_db = load_model_artifacts()
        num_recipes = len(artifacts_db["recipes"])
        logger.info(f"✓ Successfully loaded {num_recipes} recipes and all ML models.")
        logger.info("✓ Backend is ready to serve requests!")
    except FileNotFoundError as e:
        logger.error("\n" + "="*60)
        logger.error("STARTUP WARNING: Model artifacts not found!")
        logger.error("="*60)
        logger.error(str(e))
        logger.error("\nThe backend will start but /recommend endpoint will not work.")
        logger.error("See DATA_CONTRACT.md for required files.")
        logger.error("="*60 + "\n")
        artifacts_db = None
    except Exception as e:
        logger.error(f"\nUnexpected error loading model artifacts: {e}")
        logger.error("The backend will start but /recommend endpoint will not work.\n")
        artifacts_db = None
    
    yield
    
    logger.info("Shutting down application...")
    
app = FastAPI(lifespan=lifespan)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation Error", "errors": exc.errors()},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please contact support."},
    )

class TestInput(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Culinary Compass Backend"}

@app.get("/health")
def read_health():
    """Basic health check endpoint."""
    return {"status": "healthy", "message": "Backend is running"}

@app.get("/status")
def read_status():
    """
    Check if ML models are loaded and ready.
    Returns detailed status information.
    """
    if artifacts_db is None:
        return {
            "status": "not_ready",
            "models_loaded": False,
            "message": "Model artifacts are not loaded. Check server logs for details.",
            "recommendation": "Ensure preprocessing pipeline has been run. See DATA_CONTRACT.md"
        }
    
    return {
        "status": "ready",
        "models_loaded": True,
        "num_recipes": len(artifacts_db["recipes"]),
        "tfidf_shape": artifacts_db["tfidf_matrix"].shape,
        "embeddings_shape": artifacts_db["recipe_embeddings"].shape,
        "message": "All models loaded successfully. Ready to serve recommendations."
    }

@app.post("/echo")
def echo_message(data: TestInput):
    return {"you_said": data.message}

@app.post("/recommend", response_model=List[RecipeResponse])
def recommend_ingredients(data: IngredientQuery):
    """
    Get recipe recommendations based on user ingredients.
    
    Requires model artifacts to be loaded (check /status endpoint).
    """
    # Check if models are loaded
    if artifacts_db is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Service temporarily unavailable: Model artifacts not loaded. "
                "Please check server logs and ensure preprocessing has been completed. "
                "See DATA_CONTRACT.md for details."
            )
        )
    
    # Validate input
    if not data.ingredients or len(data.ingredients) == 0:
        raise HTTPException(
            status_code=422,
            detail="At least one ingredient is required"
        )
    
    # Pass the loaded artifacts to the inference function
    try:
        results = find_similar_recipes(data.ingredients, artifacts_db)
        return results
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during recipe recommendation: {str(e)}"
        )
