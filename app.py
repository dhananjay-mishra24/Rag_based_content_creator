import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Setting Up the API-Key
OPENAI_API_KEY = "APIKEY"
#Page configuratin of Streamlit 
st.set_page_config(page_title="AI Marketing Generator", layout="centered")
st.title("AI Marketing Content Generator")


@st.cache_data
def load_data():

    """
    Loads the marketing dataset .

    Returns:
        pd.DataFrame: A DataFrame containing enriched marketing content,
                      including embedded vectors for semantic search.
    """

    return pd.read_pickle("enriched_with_embeddings.pkl")

enriched_df = load_data()


def semantic_search(query, df, top_n=3):

    """
    Performs semantic similarity search on marketing content using sentence embeddings.

    Args:
        query (str): The user query or content prompt to find relevant examples for.
        df (pd.DataFrame): The enriched dataset containing content and embeddings.
        top_n (int): Number of top similar results to return.

    Returns:
        pd.DataFrame: Top-N rows from the dataset most semantically similar to the query.
                      Includes columns: type, client_name, audience, product, enriched_content, and embedding.
    """

    model = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = model.encode(query).reshape(1, -1)
    embeddings = np.vstack(df["embedding"].values)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][["type", "client_name", "audience", "product", "enriched_content", "embedding"]]

def build_rag_prompt(query, retrieved_chunks, tone, audience, format_type):

    """
    Constructs a prompt for RAG by combining user input
    with semantically similar past content examples.

    Args:
        query (str): The user's natural language request or campaign objective.
        retrieved_chunks (List[str]): List of top semantically matched content examples.
        tone (str): Desired tone for the generated content (e.g., Friendly, Bold).
        audience (str): Target audience for the campaign (e.g., Gen Z skincare audience).
        format_type (str): Type of marketing content (e.g., Social Media, Email, Blog).

    Returns:
        str: A structured prompt combining user intent and contextual examples,
             ready for use with a generative language model.
    """


    intro = f"You are a marketing assistant for a brand targeting {audience}. " \
            f"Write a {format_type.lower()} post in a {tone.lower()} tone.\n\n"
    query_section = f"User's request:\n\"{query}\"\n\n"
    examples_intro = "Here are some previous examples from this brand:\n\n"
    examples_body = "\n\n".join(retrieved_chunks)
    instruction = f"\n\nBased on the above, write a new {format_type.lower()} post that matches the tone and audience."
    return intro + query_section + examples_intro + examples_body + instruction



def generate_content(prompt, model_type="openai", openai_api_key=None, hf_api_token=None):

    """
    Generates marketing content based on a given prompt using either OpenAI GPT-3.5 or Hugging Face GPT-2.

    Args:
        prompt (str): The RAG-generated input prompt for the model.
        model_type (str): Type of model to use ("openai" or "huggingface").
        openai_api_key (str): API key for OpenAI (required if model_type is "openai").
        hf_api_token (str): API token for Hugging Face (required if model_type is "huggingface").

    Returns:
        str: The generated content as a string.
    """


    if model_type == "openai":
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful marketing assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    else:
        from transformers import pipeline
        pipe = pipeline("text-generation", model="gpt2", token=hf_api_token)
        result = pipe(prompt, max_length=256, do_sample=True, top_k=50)
        return result[0]["generated_text"]




def generate_until_good(prompt_template, top_matches, user_inputs, model_type="openai", openai_api_key=None, hf_api_token=None, similarity_threshold=0.8, max_attempts=5):

    """
    Attempts to generate high-quality marketing content using a RAG-based prompt and a generative model.
    If the generated content does not meet the desired similarity threshold, it retries up to `max_attempts`.
    Returns the best-scoring version if threshold is not met.

    Args:
        prompt_template (str): Base prompt to generate content.
        top_matches (pd.DataFrame): Top semantic examples with 'embedding' column.
        user_inputs (dict): Metadata from user including tone, audience, format_type, etc.
        model_type (str): One of 'openai' or 'huggingface'.
        openai_api_key (str): API key for OpenAI GPT models.
        hf_api_token (str): API token for Hugging Face.
        similarity_threshold (float): Minimum average cosine similarity to accept output.
        max_attempts (int): Maximum number of retries.

    Returns:
        dict: Dictionary containing the final or best output with metadata and score.
    """



    model = SentenceTransformer("all-mpnet-base-v2")
    reference_embeddings = np.vstack(top_matches["embedding"].values)
    prompt = prompt_template
    attempt = 0

    while attempt < max_attempts:
        st.write(f"\nAttempt {attempt + 1}...\n")
        output = generate_content(prompt, model_type, openai_api_key, hf_api_token)
        generated_embedding = model.encode(output).reshape(1, -1)
        similarity = cosine_similarity(generated_embedding, reference_embeddings)[0]
        feedback_score = np.mean(similarity)
        st.write("📝 Generated Output:\n", output)
        st.write(f"\nFeedback Similarity Score: {feedback_score:.2f}")

        if feedback_score >= similarity_threshold:
            st.write("\nAccepted! This version meets quality threshold.")
            return {
                "type": user_inputs["format_type"],
                "client_name": user_inputs["client_name"],
                "product": user_inputs["product_name"],
                "audience": user_inputs["audience"],
                "content": output,
                "embedding": generated_embedding[0],
                "feedback_score": feedback_score
            }
        prompt = prompt_template + "\n\n(Note: Make the tone more on-brand, casual, and aligned with examples. Try again.)"
        attempt += 1

    st.warning("⚠️ Max attempts reached. Final version may need manual review.")
    return {
        "type": user_inputs["format_type"],
        "client_name": user_inputs["client_name"],
        "product": user_inputs["product_name"],
        "audience": user_inputs["audience"],
        "content": output,
        "embedding": generated_embedding[0],
        "feedback_score": feedback_score
    }

# Settin up the form for the user 
st.subheader("Step 1: Enter Campaign Details")
with st.form("content_form"):
    client_name = st.text_input("Client Name")
    industry = st.text_input("Industry")
    values = st.text_input("Brand Values (comma-separated)")
    content_type = st.selectbox("Content Type", ["Social Media", "Email", "Blog", "Ad Copy"])
    tone = st.selectbox("Tone", ["Friendly", "Professional", "Bold", "Warm and Luxurious"])
    audience = st.selectbox("Target Audience", [
        "Gen Z skincare audience",
        "Eco-conscious millennials",
        "Women aged 25–40",
        "College athletes",
        "Stressed professionals",
        "Fashion-forward Gen Z"
    ])
    product_name = st.text_input("Product Name")
    category = st.text_input("Product Category")
    features = st.text_input("Features (comma-separated)")
    benefits = st.text_input("Benefits (comma-separated)")

    model_choice = st.selectbox("Model Type", ["OpenAI", "Hugging Face"])
    hf_api_token = st.text_input("Hugging Face Token", type="password") if model_choice == "Hugging Face" else ""

    submitted = st.form_submit_button("Generate Content")

# Running the logic once user click Generate Content Button
if submitted:
    user_inputs = {
        "client_name": client_name,
        "industry": industry,
        "values": [v.strip() for v in values.split(",")],
        "format_type": content_type,
        "tone": tone,
        "audience": audience,
        "product_name": product_name,
        "category": category,
        "features": [f.strip() for f in features.split(",")],
        "benefits": [b.strip() for b in benefits.split(",")]
    }

    query = f"Write a {tone.lower()} {content_type.lower()} post about {product_name} targeting {audience}"
    top_matches = semantic_search(query, enriched_df, top_n=3)
    retrieved_chunks = top_matches["enriched_content"].tolist()

    rag_prompt = build_rag_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        tone=tone,
        audience=audience,
        format_type=content_type
    )

    #Chossing the model either OpenAI or huggingface

    model_type = "openai" if model_choice == "OpenAI" else "huggingface"


    #Setting up the final output
    final_output = generate_until_good(
        prompt_template=rag_prompt,
        top_matches=top_matches,
        user_inputs=user_inputs,
        model_type=model_type,
        openai_api_key=OPENAI_API_KEY,
        hf_api_token=hf_api_token,
        similarity_threshold=0.8,
        max_attempts=5
    )

    st.subheader("Generated Marketing Content")
    st.text_area("Output", value=final_output["content"], height=200)
