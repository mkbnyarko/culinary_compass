# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import hdbscan
import pickle
import re
import ast
from collections import Counter, defaultdict
from datasets import load_dataset
from itertools import combinations
from functools import reduce
from spacy.matcher import PhraseMatcher
from rapidfuzz import process, fuzz  # faster fuzzy matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# Loading raw data
raw_data = load_dataset("json", data_files="../data/recipe.json")
df = raw_data["train"].to_pandas()


# Ensuring consistency in string formatting
def normalize_column(col):
    """
    Normalize a column for duplicate detection:
    - Strings → lowercase, stripped
    - Lists/arrays → lowercase, stripped, sorted, converted to tuple
    """
    if isinstance(col, str):
        return col.strip().lower()
    elif isinstance(col, (list, np.ndarray)):
        # Lowercase each element, strip spaces, sort, convert to tuple
        cleaned = tuple(sorted([str(x).strip().lower() for x in col]))
        return cleaned
    return col

# Columns to check for duplicates
cols_to_check = ["recipe_title", "description", "ingredients", "directions"]

# Create normalized columns
normalized_cols = {col: df[col].apply(normalize_column) for col in cols_to_check}

# Combine into a DataFrame
norm_df = df.assign(**normalized_cols)


# Deleting duplicates
deduplicated_df = norm_df.drop_duplicates(subset=["recipe_title", "description", "ingredients", "directions"])
deduplicated_df = deduplicated_df.copy()


# Extracting relevant fields
recipes_df = deduplicated_df[["recipe_title", "ingredients", "directions"]]
recipes_df = recipes_df.copy()

# Capitalize each word in Recipe Title
recipes_df["recipe_title"] = recipes_df["recipe_title"].str.title()


# Medium language model
nlp = spacy.load("en_core_web_md")


# STAGE 1 INGREDIENT CLEANING
def clean_stage1(raw):
    raw = raw.lower().strip()
    
    # Replace unicode fractions
    UNICODE_FRAC = {"½": "1/2", "⅓": "1/3", "¼": "1/4", "⅛": "1/8", "⅔": "2/3", "¾" : "3/4", "⅜" : "3/8", "⅞" : "7/8"}
    for frac, val in UNICODE_FRAC.items():
        raw = raw.replace(frac, val)
        
    # Remove “such as ...” example clauses
    raw = re.sub(r"such as [a-zA-Z\s']+", " ", raw)
    
    # Remove numbers but keep B12/B6 etc.
    raw = re.sub(r'\b\d+(\.\d+)?\b(?![a-zA-Z])', ' ', raw)

    # Remove numeric quantities
    # raw = re.sub(r"(\d+\/\d+|\d+\.\d+|\d+)", " ", raw)

    # Remove measurement units
    raw = re.sub(r"\b(cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|oz|ounce|ounces|gram|grams|kg|kilogram|kilograms|pound|pounds|lb|\
                 pinch|pint|quart|quartered|dash|sprig|inch|inches|pieces|sized|size|whole)\b", " ", raw)

    # Remove preparation-only words (but do not remove meaningful adjectives)
    PREP_WORDS = r"(chopped|diced|minced|sliced|slices|skinned|peeled|halved|shucked|shredded|ground|grated|granulated|trimmed|rinsed|patted|divided|optional|crush|crushed|garnish|\
        cooked|prepared|cut|pat|dry|thaw|drained|refrigerate|frozen|thawed|remove|dusting|squeezed|scrubbed|finely|coarse|coarsely|cold|unsalted|lightly|crumbled|thick|processed)"
    raw = re.sub(rf"\b{PREP_WORDS}\b", " ", raw)

    # Remove other terms
    OTHER_WORDS = r"(plus|more|into|for|extra|additional|taste|package|bag|box|can|cans|canned|tube|jar|bottle|container|about|total|desired|needed|serving|note|icing|dipping|wooden|\
        toothpicks|skewers|parchment|packet|baby|everything|italian-style|japanese-style|american)"
    raw = re.sub(rf"\b{OTHER_WORDS}\b", " ", raw)

    # Remove punctuation while leaving some possible ingredient connectors
    raw = re.sub(r"[^\w\s/,&-]", " ", raw)

    # Collapse whitespace
    raw = re.sub(r"\s+", " ", raw).strip()
    
    return raw


# Apply to dataset
recipes_df["clean_ingredients_stage1"] = recipes_df["ingredients"].apply(
    lambda lst: [clean_stage1(raw) for raw in lst]
)


# STAGE 2 INGREDIENT CLEANING
def split_tokens(text):
    """
    Split Stage-1 cleaned text into tokens.
    Uses commas, slashes, ' and ', ' or ' etc.
    """
    # replace and/or with comma (but not inside ingredient names)
    text = re.sub(r"\s+(and|or|\&)\s+", ",", text)

    # split on commas or slashes
    raw_tokens = re.split(r"[,/]", text)

    # clean whitespace
    return [t.strip() for t in raw_tokens if t.strip()]


def looks_like_garbage(token):
    """
    Shape-based garbage detection.
    No vocabulary lists — entirely rule-based.
    """
    t = token.lower().strip()

    # too short (except valid short ingredients)
    if len(t) <= 2 and t not in {"oil", "yam", "tea"}:
        return True

    # remove tokens ending in filler words
    if re.search(r"(needed|serving|taste|note)$", t):
        return True

    # remove repeated nonsense like "wet wet sauce"
    if re.search(r"\b(\w+)\s+\1\b", t):
        return True

    # no alphabetic characters
    if not re.search(r"[a-zA-Z]", t):
        return True
    
    return False


def extract_ingredient_phrase(t):
    """
    Extracts the main ingredient phrase using POS-based noun chunking,
    preserving multiword ingredients naturally.
    """
    t = t.strip().lower()

    # Salt-and-pepper pattern
    if " and " in t:
        parts = [extract_ingredient_phrase(x) for x in t.split(" and ")]
        flat = []
        for p in parts:
            if isinstance(p, list):
                flat.extend(p)
            else:
                flat.append(p)
        return flat

    doc = nlp(t)

    # POS-based chunks (noun phrases)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    if noun_chunks:
        phrase = noun_chunks[-1]  # get the main noun phrase
    else:
        phrase = t

    # Remove undesirable descriptors but keep food words
    DESCRIPTORS = {
        "large", "small", "fresh", "freshly", "boneless", "skinless", "zested", "juiced", "minced", "toasted", "cooked", "flaked", "unsweetened", "roasted",
        "4ounce", "bone-in", "skin", "round", "salted", "uncooked", "seasoned", "ground", "crushed", "sliced", "diced", "creamy", "halved", "beaten",
        "melted", "softened", "cooked", "split", "nugget", "dried", "s", "lbs", "-", "-half", "ablespoons", "nonstick", "cooking", "spray"
    }

    words = []
    for w in phrase.split():
        if w not in DESCRIPTORS:
            words.append(w)

    cleaned = " ".join(words).strip()

    return cleaned


def remove_filler_words(phrase):
    """
    Removes standalone filler words from final ingredient tokens,
    but does NOT destroy valid multiword ingredient names.
    """
    FILLER_STOPWORDS = {
        "and", "to", "or", "for", "with", "in", "of", "the",
        "a", "an", "as", "on", "into", "at"
        }
    words = phrase.split()
    words = [w for w in words if w not in FILLER_STOPWORDS]
    return " ".join(words).strip()


def clean_stage2(stage1_output):
    tokens = split_tokens(stage1_output)

    cleaned = []

    for t in tokens:
        t = t.strip().lower()
        
        # remove leading/trailing punctuation
        t = re.sub(r"^[^\w]+|[^\w]+$", "", t)

        # Skip garbage tokens
        if looks_like_garbage(t):
            continue

        # Extract ingredient phrase
        extracted = extract_ingredient_phrase(t)

        # handle salt-and-pepper cases (list return)
        if isinstance(extracted, list):
            for x in extracted:
                x = remove_filler_words(x)
                if x and not looks_like_garbage(x):
                    cleaned.append(x)
        else:
            x = remove_filler_words(extracted)
            if x and not looks_like_garbage(x):
                cleaned.append(x)

    # Remove duplicates while maintaining order
    final = list(dict.fromkeys(cleaned))
    return final


# Apply to actual dataset
recipes_df["clean_ingredients_stage2"] = recipes_df["clean_ingredients_stage1"].apply(
    lambda lst: [clean_stage2(raw) for raw in lst]
)


# flatten clean_ingredients_stage2 output one level: [[...], [...]] → [...]
def parse_ingredients(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return [item for sublist in x for item in sublist]

recipes_df["clean_ingredients"] = (
    recipes_df["clean_ingredients_stage2"]
    .apply(parse_ingredients)
)


# FINAL STAGE OF INGREDIENT CLEANING
def final_cleaning(ingredients):
    """
    ingredients: List[str]
    returns: List[str]
    """
    if not ingredients:
        return []
    
    cleaned = []
    
    # Remove undesirable descriptors
    UNWANTED_WORDS = {
        "large", "small", "fresh", "freshly", "boneless", "skinless", "zested", "juiced", "minced", "toasted", "cooked", "flaked", "unsweetened", "roasted",
        "4ounce", "bone-in", "skin", "round", "salted", "uncooked", "seasoned", "ground", "crushed", "sliced", "diced", "creamy", "halved", "beaten",
        "melted", "softened", "cooked", "split", "nugget", "dried", "s", "lbs", "-", "half", "ablespoons", "nonstick", "cooking", "spray", "all", "purpose",
        "2tablespoons", "4cup", "3x1", "added", "white", "brown", "red", "green", "black", "yellow", "undrained", "aluminium", "foil", "packaged", "reduced", "medium", "sodium", 
        "pure", "stemmed", "color", "flavor", "allpurpose", "almondflavored", "work", "surface", "very", "hot", "soft", "thin", "thick", "bunch", "plain", "italian", "glutenfree",
        "sweet", "sugarbased", "sugarfree", "semisweet", "seeded", "seedless"
        }
    
    for ing in ingredients:

        # remove punctuation
        ing = re.sub(r'[^\w\s]', '', ing)
        
        tokens = [
            t for t in ing.split() if t not in UNWANTED_WORDS
        ]
        if tokens:
            cleaned.append(" ".join(tokens))

    return cleaned


# Apply to dataset
recipes_df["clean_ingredients"] = recipes_df["clean_ingredients"].apply(final_cleaning)


# Save clean dataset with proper field naming for inference
import json
import os

final_recipes = []
for _, row in recipes_df.iterrows():
    final_recipes.append({
        "name": row["recipe_title"],  # Map recipe_title -> name for inference
        "ingredients": row["clean_ingredients_norm"],
        "directions": row["directions"]
    })

output_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_recipes, f, indent=2)

print(f"Exported {len(final_recipes)} recipes to {output_path}")


# Singularize nouns
def light_normalize(ingredients, nlp):
    normalized = []

    for ing in ingredients:
        doc = nlp(ing)
        words = []
        for t in doc:
            if t.pos_ == "NOUN":
                words.append(t.lemma_)
            else:
                words.append(t.text)
        normalized.append(" ".join(words))

    return normalized


recipes_df["clean_ingredients_norm"] = (
    recipes_df["clean_ingredients"]
    .apply(lambda x: light_normalize(x, nlp))
)

# doc object
documents = recipes_df['clean_ingredients_norm'].apply("|".join)

# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    min_df=3,                # drop very rare noise
    max_df=0.85,             # suppress salt/oil/water
    norm="l2",               # cosine similarity friendly
    use_idf=True,
    token_pattern=r"[^|]+",
    smooth_idf=True,
    sublinear_tf=True        # log(1 + tf)
)

X = vectorizer.fit_transform(documents)


# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

ingredient_texts = recipes_df["clean_ingredients_norm"].apply(
    lambda x: ", ".join(x)
).tolist()

embeddings = model.encode(
    ingredient_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)


# SIMILARITY COMPUTATION
# X is the TF-IDF sparse matrix
X_sparse = csr_matrix(X)

# Normalize for cosine similarity (makes it dot product)
X_norm = normalize(X_sparse, norm='l2', axis=1)

# TF-IDF similarity matrix
tfidf_sim = cosine_similarity(X_norm)

# Embeddings similarity matrix
embed_sim = cosine_similarity(embeddings)