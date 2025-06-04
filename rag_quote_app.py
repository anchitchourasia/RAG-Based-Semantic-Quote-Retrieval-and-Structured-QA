import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import matplotlib.pyplot as plt
import seaborn as sns

# --- STEP 1: Setup ---
st.set_page_config(page_title="Quote Search Assistant", layout="centered")
st.title("🔍 Quote Search Assistant (RAG + Gemini 1.5 Flash)")

st.markdown("""
Enter a natural language query like:
- *"inspirational quotes by Oscar Wilde"*
- *"quotes about courage from women authors"*
- *"quotes tagged with both ‘life’ and ‘love’ by 20th century authors"*
""")

# --- STEP 2: Load Gemini API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
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

# --- STEP 5: Multi-hop Filter Controls (Optional) ---
with st.expander("🔎 Optional Filters (Multi-hop Querying)"):
    author_filter = st.text_input("Filter by author keyword")
    tag_filter = st.text_input("Filter by required tag(s), comma-separated (e.g., life,love)")

# --- STEP 6: User Query ---
query = st.text_input("💬 Enter your quote query")

if query:
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), 10)

    retrieved_quotes = []
    json_results = []

    st.subheader("⭐ Top Matching Quotes")

    for idx in I[0]:
        quote = df.iloc[idx]['quote']
        author = df.iloc[idx]['author']
        tags = ', '.join(df.iloc[idx]['tags'])

        # Multi-hop filtering
        if author_filter and author_filter.lower() not in author.lower():
            continue
        if tag_filter:
            required_tags = [t.strip().lower() for t in tag_filter.split(',')]
            if not all(t in tags.lower() for t in required_tags):
                continue

        st.markdown(f"**“{quote}”**")
        st.markdown(f"- *{author}*  \n_Tags: {tags}_")
        st.markdown("---")

        retrieved_quotes.append(f"“{quote}” — {author} (Tags: {tags})")
        json_results.append({
            "quote": quote,
            "author": author,
            "tags": tags.split(', ')
        })

    # --- STEP 7: JSON Download ---
    if json_results:
        st.download_button("📥 Download Results as JSON", json.dumps(json_results, indent=2), file_name="quotes.json", mime="application/json")

    # --- STEP 8: Gemini Summary ---
    if retrieved_quotes:
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
            st.subheader("🧠 AI Summary (Gemini 1.5 Flash)")
            st.markdown(response.text)
        except Exception as e:
            st.warning(f"Gemini summarization failed: {e}")
    else:
        st.info("No quotes matched the current filters.")

# --- STEP 9: Bonus Visualizations ---
st.subheader("📊 Quote Dataset Insights")

col1, col2 = st.columns(2)

with col1:
    top_authors = df['author'].value_counts().head(10)
    st.markdown("#### Top 10 Authors by Quote Count")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax1)
    st.pyplot(fig1)

with col2:
    all_tags = df['tags'].explode()
    top_tags = all_tags.value_counts().head(10)
    st.markdown("#### Top 10 Tags")
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_tags.values, y=top_tags.index, ax=ax2)
    st.pyplot(fig2)
