# 🧠 AI Marketing Content Generator

This project is a **Retrieval-Augmented Generation (RAG)** based content generation tool built using `Streamlit`, `SentenceTransformers`, and `OpenAI` or `Hugging Face` language models. It helps brands automatically generate tailored marketing content (e.g., social media posts, emails, ad copy) using past examples and feedback as context.

## 🔍 Key Features

* ✅ RAG-based prompting using top semantically similar examples
* ✅ Supports OpenAI GPT-3.5 or Hugging Face GPT-2 for generation
* ✅ Feedback-driven loop using cosine similarity
* ✅ Automatically adapts tone, audience, format, and product features
* ✅ Streamlit web app with form-based input
* ✅ Structured data persistence in JSON (client, product, feedback, SEO)

---

## 📂 Project Structure

| File               | Description                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------- |
| `app.py`           | Streamlit app that serves the interactive web UI                                             |
| `pipeline.py`      | Core logic for prompt building, semantic search, content generation, and feedback evaluation |
| `reading_Files.py` | Loads and manages the datasets (`client_profiles.json`, `marketing_assets.json`, etc.)       |
| `*.json`           | Data storage for clients, products, marketing assets, feedback, and keywords                 |

---

## 🧪 How It Works

1. **User Inputs**: Client name, tone, audience, product, etc.
2. **RAG Prompting**: Uses semantic search to retrieve top similar examples from previous content.
3. **Content Generation**: Generates content using GPT-3.5 or GPT-2.
4. **Feedback Scoring**: Cosine similarity compares generated content with retrieved examples.
5. **Finalization**: Accepts the first output above threshold (e.g., 0.8) or returns the best attempt.


## 🔑 API Configuration

* For OpenAI:

  * Add your API key inside `app.py` or pass via environment variable.
* For Hugging Face:

  * Provide your token in the Hugging Face pipeline section if `model_type='huggingface'`.

---

## 📊 Evaluation Metrics

* **Semantic Feedback Score**: Uses cosine similarity against top retrieved examples.
* **Human Feedback** *(optional)*: Users can manually review and rate content.
* The system retains and learns from accepted content.

---

## 🥉 Example Use Case

> “Write a friendly social media post for Gen Z audience promoting a glowing skin serum.”

The system retrieves past examples (blogs, ads, emails), builds a RAG prompt, and generates branded, tone-specific copy tailored to your product.

---

## 📌 Future Work

* Automatic integration with email marketing tools
* Visual dashboard to monitor feedback trends
* Fine-tuning with collected human-verified outputs

