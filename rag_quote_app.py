import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- STEP 1: Setup ---
st.set_page_config(page_title="Quote Search Assistant", layout="centered")
st.title("🔍 Quote Search Assistant (RAG + Gemini 1.5 Flash)")
st.markdown("""
Enter a natural language query like:
- *"inspirational quotes by Oscar Wilde"*
- *"quotes about courage from women authors"*
""")

# --- STEP 2: Load Gemini API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini 1.5 Flash model
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# --- STEP 3: Load Quote Dataset ---
@st.cache_data
def load_quotes():
    dataset = load_dataset("Abirate/english_quotes")
    df = pd.DataFrame(dataset['train'])
    df['combined'] = df['quote'] + " | Author: " + df['author'] + " | Tags: " + df['tags'].apply(lambda x: ', '.join(x))
    return df

df = load_quotes()

# --- STEP 4: Embed and Index Quotes ---
@st.cache_resource
def embed_quotes(quotes):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(quotes, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index

model, index = embed_quotes(df['combined'].tolist())

# --- STEP 5: Query Input ---
query = st.text_input("🔍 Enter your quote query")

if query:
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), 5)

    retrieved_quotes = []
    st.subheader("⭐ Top Matching Quotes")

    for idx in I[0]:
        quote = df.iloc[idx]['quote']
        author = df.iloc[idx]['author']
        tags = ', '.join(df.iloc[idx]['tags'])

        st.markdown(f"**“{quote}”**")
        st.markdown(f"- *{author}*  \n_Tags: {tags}_")
        st.markdown("---")

        retrieved_quotes.append(f"“{quote}” — {author} (Tags: {tags})")

    # --- STEP 6: Gemini LLM Summary ---
    context_block = "\n".join(retrieved_quotes)
    llm_prompt = f"""
Here are some quotes retrieved based on a user's search query:

{context_block}

Please:
1. Summarize the common theme across these quotes.
2. Highlight the most powerful quote and explain why.
"""

    try:
        response = gemini_model.generate_content(llm_prompt)
        summary = response.text
        st.subheader("🧠 AI Summary (Gemini 1.5 Flash)")
        st.markdown(summary)
    except Exception as e:
        st.warning(f"Gemini summarization failed: {e}")
