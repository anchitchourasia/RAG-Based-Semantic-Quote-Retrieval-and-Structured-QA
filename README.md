# 🔍 Task 2 – Quote Search Assistant (RAG + Gemini 1.5 Flash)

This project is built as part of the **Vijayi WFH Internship Assignment (May–June 2025)**. It demonstrates a complete **Retrieval-Augmented Generation (RAG)** pipeline that allows users to semantically search for quotes using natural language and receive intelligent results with summaries.

---

## 💡 Features

- 📖 Semantic quote retrieval from the `Abirate/english_quotes` dataset
- ⚡ FAISS-based vector similarity search with `SentenceTransformer`
- 🧠 Gemini 1.5 Flash LLM summary generation (via Google AI Studio API)
- 🔎 Multi-hop filtering by `author` and `tags`
- 📥 JSON download of search results
- 📊 Visualizations: top authors & tags
- 🌐 Streamlit UI for easy interaction

---

## 📁 Dataset Used

- **Source**: [Hugging Face – Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)
- Fields:
  - `quote`: The quote text
  - `author`: The quote author
  - `tags`: List of tags/keywords

---

## 🚀 How It Works

### 🔹 Step-by-Step Flow

| Step | Component |
|------|----------|
| 1. | Load dataset and combine quote-author-tags |
| 2. | Encode using `all-MiniLM-L6-v2` |
| 3. | Index vectors in FAISS |
| 4. | Accept user query and optional filters |
| 5. | Retrieve top relevant quotes |
| 6. | Summarize using Gemini 1.5 Flash |
| 7. | Allow JSON download |
| 8. | Show visual insights (bar plots) |

---

## 🛠️ Setup Instructions

### 1. Clone the repo / copy files

Make sure your `GOOGLE_API_KEY` is stored in a `.env` file.

GOOGLE_API_KEY=your_gemini_key_here

perl
Copy
Edit

### 2. Install dependencies

```bash
pip install streamlit datasets sentence-transformers faiss-cpu numpy pandas python-dotenv google-generativeai matplotlib seaborn
3. Run the app
bash
Copy
Edit
streamlit run app.py
💬 Example Queries
text
Copy
Edit
"inspirational quotes by Oscar Wilde"
"quotes about failure by scientists"
"quotes tagged with courage and hope"
📥 JSON Output Example
json
Copy
Edit
[
  {
    "quote": "Success is not final, failure is not fatal...",
    "author": "Winston Churchill",
    "tags": ["success", "failure"]
  },
  ...
]
🧠 Gemini 1.5 Flash Summary Output
text
Copy
Edit
Summary:
Most quotes focus on resilience and the idea that failure is part of success. The most powerful quote is from Winston Churchill...

It stands out due to its universal message and emotional impact.
📊 Visual Insights (Bonus)
✅ Top 10 Authors with most quotes

✅ Top 10 Most Frequent Tags

Generated using Seaborn & Matplotlib for exploratory analysis.

🌐 Tech Stack
Streamlit – frontend UI

SentenceTransformers – semantic embedding

FAISS – vector similarity search

Gemini 1.5 Flash – summary generation (Google Generative AI)

Matplotlib, Seaborn – visualizations

dotenv – secure API key management

🎥 Demo Video
The demo includes:

Code walkthrough

App usage with sample queries

Gemini-generated summaries

Bonus visualizations

📁 [Demo video attached or shared via Drive]-- https://drive.google.com/file/d/1s0TnqhhGf5ObviHGb7x_Hnfk5ZrpOfN3/view?usp=sharing

📦 Files Included
app.py – Streamlit application

.env – Gemini API key (not shared publicly)

README.md – this file

Demo video file / link

✅ Status
✅ Task 2 fully implemented with all bonus points covered:

 RAG pipeline

 Gemini summarization

 Filtering

 JSON export

 Visualizations
