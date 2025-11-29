# **3-Week Plan (1–2 hours/day)**

**Outcome:** A functional, clean MVP of your recipe recommender with FastAPI + Streamlit + hybrid similarity.

---

# **WEEK 1 — NLP + Ingredient Intelligence**

This week focuses on everything before machine learning: text cleanup, normalization, and similarity basics.

---

### **Day 1: Project Setup + Dataset Placement**

* Create repo + branch structure
* Add basic folders (`data/`, `src/`, `notebooks/`)
* Place a small recipe dataset (100–500 recipes)
* Write a simple `load_data()` function

**Goal:** Foundation, no overwhelm.

---

### **Day 2: Ingredient Cleaning — Basics**

Learn:

* lowercasing
* removing punctuation
* tokenizing
* removing numeric quantities

Do:

* Implement `clean_ingredient_basic(text)`

**Why:** Gets you 60% of the way.

---

### **Day 3: Ingredient Cleaning — Normalization**

Learn:

* regex
* removing measurement units
* trimming descriptors (“chopped”, “sliced”, “minced”)
* simple synonym mapping (manual)

Do:

* Implement `clean_ingredient(text)`
* Test on 10–20 messy examples

**This day alone saves your pipeline from long-term pain.**

---

### **Day 4: Preprocess Recipe Ingredients**

Do:

* Implement `preprocess_ingredient_list(ingredients)`
* Apply to the whole dataset
* Save cleaned dataset to `data/processed/cleaned.json`

**Goal:** You now have structured ingredient data.

---

### **Day 5: TF-IDF Similarity**

Learn:

* `TfidfVectorizer`
* cosine similarity

Do:

* Train TF-IDF on recipe ingredient strings
* Compute similarity matrix
* Save:

  * `tfidf_vectorizer.pkl`
  * `tfidf_matrix.npz`

---

### **Day 6: Embeddings + Semantic Similarity**

Learn:

* SentenceTransformers
* embeddings shape + normalization
* semantic matching

Do:

* Generate embeddings for recipes
* Save `embeddings.npy`

**You now have two working similarity engines.**

---

### **Day 7: *Mini Recommender* (End of Week 1 Build)**

Implement a Python function:

```python
recommend(query_ingredients)
```

Combine:

* TF-IDF similarity
* Embedding similarity
* Ingredient overlap

Return top 10.

**This week ends with a working console recommender.**

---

# **WEEK 2 — Building the Backend (FastAPI)**

This week focuses fully on API design, loading models, and building a stable backend.

---

### **Day 8: FastAPI Basics**

Learn:

* endpoints
* request models
* returning JSON

Do:

* Build `/health` endpoint
* Build a test `/echo` endpoint

---

### **Day 9: Load Models in FastAPI**

Do:

* Load:

  * cleaned dataset
  * TF-IDF vectorizer + matrix
  * embeddings
* Store them as global objects
* Ensure they load once, not per request

**This is the backbone of your system.**

---

### **Day 10: Build `/recommend` Endpoint**

Do:

* Accept a list of ingredients
* Clean them with your preprocessing pipeline
* Generate TF-IDF + embedding vector for the query
* Return top matches

**At this point, your backend is functional.**

---

### **Day 11: Missing Ingredient Logic**

Do:

* Compute ingredients the user *doesn't* have
* Add them to the API response
* Sort by fewer missing ingredients

**This improves the quality of recommendations dramatically.**

---

### **Day 12: Scoring Improvements**

Do:

* Tune hybrid weights
* Normalize score ranges
* Optional: add penalty for too many missing ingredients

**This day makes your system feel intelligent.**

---

### **Day 13: Logging + Error Handling**

Do:

* Add logs for:

  * invalid inputs
  * missing fields
  * preprocessing failures
* Add proper error messages
* Add typing + docstrings

---

### **Day 14: Backend Stabilization**

Do:

* Manual testing with multiple queries
* Fix edge cases ("onions", "water", "salt")
* Freeze backend functionality

**Close Week 2 with a stable, reliable API.**

---

# **WEEK 3 — Building the UI + Polishing the Project**

Now you focus on the user experience, packaging, documentation, and optional upgrades.

---

### **Day 15: Streamlit App Basics**

Do:

* Set up basic page
* Add input textbox
* Add submit button
* Display static fake response

**You get instant visual progress.**

---

### **Day 16: Connect Streamlit → FastAPI**

Do:

* Make POST request to `/recommend`
* Display returned recipes
* Format it nicely

**At this point, the entire system is connected end-to-end.**

---

### **Day 17: UI Polish**

Do:

* Add loading spinner
* Group ingredients
* Add missing-ingredient list
* Add simple scoring badges

---

### **Day 18: Improve Organization**

Do:

* Organize into:

  ```
  fridge2fork/
      api/
      app/
      src/
      data/
  ```
* Add README structure
* Add usage instructions
* Add environment setup instructions

---

### **Day 19: Basic Tests**

Do:

* Write 3–5 tests:

  * cleaning
  * similarity
  * endpoint returns valid format

Focus on simple tests—not perfection.

---

### **Day 20: Repo Cleanup**

Do:

* remove unused files
* create `requirements.txt`
* create `.gitignore`
* document all functions

---

### **Day 21: Final Integration + Demo Recording**

Do:

* Test with 10–20 ingredient combinations
* Celebrate 🚀