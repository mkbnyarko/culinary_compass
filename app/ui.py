import streamlit as st
import requests
from typing import List
import json

# Page config
st.set_page_config(
    page_title="Culinary Compass",
    page_icon="�",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Material UI-inspired design with maroon theme (dark mode focused)
CUSTOM_CSS = """
<style>
    /* Import Google Fonts and Material Symbols */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
    
    /* Material Icons styling */
    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined';
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        display: inline-block;
        line-height: 1;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: normal;
        white-space: nowrap;
        direction: ltr;
        vertical-align: middle;
    }
    
    /* Global styles - Dark theme */
    .stApp {
        font-family: 'Roboto', sans-serif;
        background-color: #0e1117 !important;
        color: #e0e0e0 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title styling */
    .main-title {
        color: #e0e0e0;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    .subtitle {
        color: #a0a0a0;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Ingredient chip styling */
    .ingredient-chip {
        display: inline-flex;
        align-items: center;
        background: #05714B;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.9rem;
        font-weight: 500;
        transition: opacity 0.2s ease;
    }
    
    .ingredient-chip:hover {
        opacity: 0.85;
    }
    
    .remove-btn {
        background: transparent;
        border: none;
        color: #ffffff;
        margin-left: 8px;
        cursor: pointer;
        font-size: 1.1rem;
        transition: opacity 0.2s;
        padding: 0;
    }
    
    .remove-btn:hover {
        opacity: 0.7;
    }
    
    /* Override Streamlit button styling to look like text */
    .stButton > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #e0e0e0 !important;
        padding: 8px 16px !important;
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        transition: opacity 0.2s ease !important;
    }
    
    .stButton > button:hover {
        opacity: 0.7 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .stButton > button:active {
        opacity: 0.5 !important;
        background: transparent !important;
    }
    
    .stButton > button:focus {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Primary button (Find Recipes) - styled as maroon text */
    .stButton > button[kind="primary"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #05714B !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        opacity: 0.7 !important;
        background: transparent !important;
    }
    
    /* Error message styling */
    .error-message {
        color: #ff6b6b;
        background: #2d1a1a;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 16px 0;
        font-size: 0.95rem;
        border-left: 4px solid #800000;
    }
    
    /* Success message */
    .success-message {
        color: #66bb6a;
        background: #1a2d1a;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 16px 0;
        font-size: 0.95rem;
        border-left: 4px solid #388e3c;
    }
    
    /* Recipe card styling */
    .recipe-card {
        background: #1a1d24;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-left: 4px solid #800000;
    }
    
    .recipe-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    
    .recipe-title {
        color: #e0e0e0;
        font-size: 1.4rem;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .recipe-match {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    
    .recipe-ingredients {
        color: #d0d0d0;
        line-height: 1.6;
    }
    
    .recipe-directions {
        color: #d0d0d0;
        line-height: 1.8;
    }
    
    .recipe-section-title {
        color: #05714B;
        font-weight: 500;
    }
    
    /* Input field enhancements */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #1a1d24 !important;
        color: #e0e0e0 !important;
        border: 1px solid #404040 !important;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #05714B !important;
        box-shadow: 0 0 0 1px rgba(5, 113, 75, 0.3) !important;
    }
    
    /* Floating label effect placeholder */
    .floating-label-container {
        position: relative;
        margin: 20px 0;
    }
    
    .floating-label {
        position: absolute;
        top: -10px;
        left: 7px;
        background: #0e1117;
        padding: 0 4px;
        font-size: 1.25rem;
        color: #05714B;
        font-weight: 500;
    }
    
    /* Section headers */
    h3 {
        color: #e0e0e0 !important;
        font-weight: 400 !important;
    }
    
    h2 {
        color: #e0e0e0 !important;
    }
</style>
"""

# Common ingredients for autocomplete
COMMON_INGREDIENTS = [
    "chicken", "beef", "pork", "fish", "shrimp", "salmon",
    "rice", "pasta", "noodles", "bread", "flour",
    "tomato", "onion", "garlic", "ginger", "potato", "carrot", "broccoli",
    "cheese", "butter", "milk", "cream", "eggs",
    "salt", "pepper", "olive oil", "soy sauce", "honey",
    "lettuce", "cucumber", "bell pepper", "mushroom", "spinach",
    "lemon", "lime", "orange", "apple", "banana",
    "chicken breast", "ground beef", "bacon", "tofu",
    "basil", "cilantro", "parsley", "oregano", "thyme"
]

# Backend configuration
BACKEND_URL = "http://127.0.0.1:8000"

def inject_css():
    """Inject custom CSS"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def show_error(message: str):
    """Display error message with friendly language"""
    st.markdown(f'<div class="error-message"><span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 8px;">error</span>{message}</div>', unsafe_allow_html=True)

def show_success(message: str):
    """Display success message"""
    st.markdown(f'<div class="success-message"><span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 8px;">check_circle</span>{message}</div>', unsafe_allow_html=True)

def get_recipe_recommendations(ingredients: List[str]) -> dict:
    """Call backend API to get recipe recommendations"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/recommend",
            json={"ingredients": ingredients},
            timeout=10
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Oops! We can't reach the recipe brain right now. Try again in a sec?"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Hold up! The recipe search is taking too long. Try again in a sec?"
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            return {
                "success": False,
                "error": "Hmm, something's not quite right with those ingredients. Try again?"
            }
        else:
            return {
                "success": False,
                "error": f"Yikes! Something went wrong on our end (Error {e.response.status_code}). Give it another shot?"
            }
    except Exception as e:
        return {
            "success": False,
            "error": "Well, this is awkward... something unexpected happened. Mind trying again?"
        }

def main():
    # Inject CSS
    inject_css()
    
    # Initialize session state
    if 'ingredients' not in st.session_state:
        st.session_state.ingredients = []
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    # Header
    st.markdown('<h1 class="main-title"><span class="material-symbols-outlined" style="font-size: 2.5rem; vertical-align: middle; margin-right: 12px;">restaurant</span>Culinary Compass</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Tell us what you\'ve got, we\'ll tell you what\'s poppin\'</p>', unsafe_allow_html=True)
    
    # Ingredient input section
    st.markdown('<div class="floating-label-container">', unsafe_allow_html=True)
    st.markdown('<span class="floating-label">What ingredients do you have?</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Autocomplete selectbox
    col1, col2 = st.columns([6, 1])
    
    with col1:
        selected_ingredient = st.selectbox(
            "Start typing to search ingredients...",
            [""] + sorted(COMMON_INGREDIENTS),
            key="ingredient_input",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("Add", key="add_btn", use_container_width=True):
            if selected_ingredient and selected_ingredient not in st.session_state.ingredients:
                st.session_state.ingredients.append(selected_ingredient)
                st.rerun()
    
    # Display current ingredients with inline remove buttons
    if st.session_state.ingredients:
        st.markdown("### Your ingredients:")
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        
        # Display each ingredient with its remove button inline
        for idx, ingredient in enumerate(st.session_state.ingredients):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f'<div class="ingredient-chip" style="margin: 4px 0;">{ingredient}</div>', unsafe_allow_html=True)
            with col2:
                if st.button("remove", key=f"remove_{idx}"):
                    st.session_state.ingredients.pop(idx)
                    st.rerun()
    
    # Search button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Find My Recipes", use_container_width=True, type="primary"):
        # Validation
        if len(st.session_state.ingredients) < 3:
            show_error("Hold on! We need at least 3 ingredients to work our magic. Add a couple more?")
        else:
            with st.spinner("Cooking up some suggestions..."):
                result = get_recipe_recommendations(st.session_state.ingredients)
                
                if result["success"]:
                    st.session_state.search_results = result["data"]
                    show_success(f"Found {len(result['data'])} amazing recipes for you!")
                else:
                    show_error(result["error"])
                    st.session_state.search_results = None
    
    # Display results
    if st.session_state.search_results:
        st.markdown("---")
        st.markdown("## <span class='material-symbols-outlined' style='vertical-align: middle; margin-right: 8px;'>restaurant_menu</span>Your Recipe Matches", unsafe_allow_html=True)
        
        for recipe in st.session_state.search_results:
            recipe_html = f'''
            <div class="recipe-card">
                <div class="recipe-title">{recipe['name']}</div>
                <div style="margin-top: 12px;">
                    <strong class="recipe-section-title">Ingredients:</strong>
                    <div class="recipe-ingredients" style="margin-top: 8px;">{', '.join(recipe['ingredients'])}</div>
                </div>
                <div style="margin-top: 12px;">
                    <strong class="recipe-section-title">Directions:</strong>
                    <ol class="recipe-directions" style="margin-top: 8px;">
                        {''.join([f'<li>{step}</li>' for step in recipe['directions']])}
                    </ol>
                </div>
            </div>
            '''
            st.markdown(recipe_html, unsafe_allow_html=True)
    
    # Clear button
    if st.session_state.ingredients:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start Over", use_container_width=True):
            st.session_state.ingredients = []
            st.session_state.search_results = None
            st.rerun()

if __name__ == "__main__":
    main()
