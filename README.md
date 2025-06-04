# RAG-Based-Semantic-Quote-Retrieval-and-Structured-QA
# 🔍 RAG-Based Quote Search Assistant (Gemini 1.5 Flash Powered)

This project is part of the **Vijayi WFH AI/ML Internship - May/June 2025**.  
It demonstrates a Retrieval-Augmented Generation (RAG) system that semantically retrieves quotes and generates intelligent summaries using **Gemini 1.5 Flash**.

---

## 🧠 What It Does

- Loads 10,000+ English quotes from HuggingFace
- Encodes them using Sentence Transformers (MiniLM)
- Stores and searches them using FAISS vector database
- Uses Gemini 1.5 Flash to:
  - Summarize top matching quotes
  - Highlight and explain the most impactful quote

---

## 🚀 How to Run

### 1. 📁 Clone or copy this folder

Make sure you have:

rag_quote_app.py
.env ← contains your API key

perl
Copy
Edit

### 2. 📦 Install required packages

```bash
pip install streamlit sentence-transformers faiss-cpu pandas datasets python-dotenv google-generativeai
3. 🔑 Set your API key
Create a file named .env in the same directory:

ini
Copy
Edit
GOOGLE_API_KEY=your_real_gemini_1_5_flash_api_key_here
4. ▶️ Launch the app
bash
Copy
Edit
streamlit run rag_quote_app.py
💡 Sample Queries
motivational quotes by Albert Einstein

funny quotes about failure

courage quotes from female authors

wisdom quotes by Oscar Wilde

🧪 Example Output
css
Copy
Edit
⭐ Top Matching Quotes
“Anyone who has never made a mistake has never tried anything new.” — Albert Einstein
...

🧠 AI Summary (Gemini 1.5 Flash):
These quotes emphasize perseverance, curiosity, and humility. The most powerful quote reframes failure as essential to growth...
🔍 Tech Stack
Layer	Tools/Libraries
Dataset	HuggingFace Abirate/english_quotes
Embeddings	sentence-transformers (MiniLM)
Vector Search	FAISS
Generation	google.generativeai — Gemini 1.5 Flash
Frontend	Streamlit
