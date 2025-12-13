from pydantic import BaseModel
from typing import List, Optional

# What the user sends
class IngredientQuery(BaseModel):
    ingredients: List[str]

# What the server sends back
class RecipeResponse(BaseModel):
    name: str
    ingredients: List[str]
    directions: List[str]
    url: Optional[str] = None 
    match_score: float