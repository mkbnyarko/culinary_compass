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

# Global storage for recipes
recipes_db = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recipes_db
    logger.info("Starting up application...")
    try:
        recipes_db = load_model_artifacts()
        logger.info(f"Successfully loaded {len(recipes_db)} recipes.")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        # We might want to raise here to prevent startup, or just log it. 
        # For now, logging is safer to keep the server alive (but useless).
        recipes_db = []
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
    return {"message": "Backend is healthy"}

@app.post("/echo")
def echo_message(data: TestInput):
    return {"you_said": data.message}

@app.post("/recommend", response_model=List[RecipeResponse])
def recommend_ingredients(data: IngredientQuery):
    # Pass the loaded recipes to the inference function
    return find_similar_recipes(data.ingredients, recipes_db)
