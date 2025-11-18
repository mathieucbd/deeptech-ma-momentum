"""
LLM Tools for M&A Transaction Analysis

This module provides embedding generation and text classification using LLM APIs.

CONFIGURATION (set in .env file):
    EMBEDDING_PROVIDER: Choose "openai" or "gemini" (default: "openai")
    OPENAI_API_KEY: Your OpenAI API key (if using OpenAI)
    GEMINI_API_KEY: Your Gemini API key (if using Gemini)

PROVIDER COMPARISON:
    OpenAI (text-embedding-3-small):
        - Dimensions: 1536
        - Max input: 8191 tokens (~32,000 chars)
        - Cost: $0.02 per 1M tokens
        - Best for: Large datasets, longer texts
    
    Gemini (text-embedding-004):
        - Dimensions: 768
        - Max input: 2048 tokens (~8,000 chars)  
        - Cost: Free tier available
        - Best for: Cost-conscious projects, shorter texts
"""

# --- Imports ---
import os
import time

import polars as pl

# --- Configuration ---
# Set EMBEDDING_PROVIDER in your .env file to either "openai" or "gemini" (default: openai)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

# Initialize the appropriate client based on provider
gemini_client = None
openai_client = None

if EMBEDDING_PROVIDER == "gemini":
    try:
        from google import genai
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        print("✓ Using Gemini API for embeddings")
    except Exception as e:
        print(f"Warning: Gemini Client initialization failed. Error: {e}")
elif EMBEDDING_PROVIDER == "openai":
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("✓ Using OpenAI API for embeddings")
    except Exception as e:
        print(f"Warning: OpenAI Client initialization failed. Error: {e}")
else:
    print(f"Warning: Unknown EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}'. Use 'openai' or 'gemini'")


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates vector embeddings for a list of text documents.
    
    Supports both OpenAI and Gemini APIs. Set EMBEDDING_PROVIDER environment variable:
    - "openai" (default): Uses text-embedding-3-small (1536 dims, up to 8191 tokens)
    - "gemini": Uses text-embedding-004 (768 dims, up to 2048 tokens)
    
    Implements exponential backoff for handling API rate limits.
    """
    embeddings = []
    max_retries = 5
    initial_delay = 1  # seconds
    
    # Determine which provider to use
    if EMBEDDING_PROVIDER == "openai":
        if openai_client is None:
            print("ERROR: OpenAI client is not initialized. Set OPENAI_API_KEY in .env")
            return embeddings
        model = "text-embedding-3-small"
        default_dimension = 1536
    elif EMBEDDING_PROVIDER == "gemini":
        if gemini_client is None:
            print("ERROR: Gemini client is not initialized. Set GEMINI_API_KEY in .env")
            return embeddings
        model = "models/text-embedding-004"
        default_dimension = 768
    else:
        print(f"ERROR: Invalid EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}'")
        return embeddings

    for i, text in enumerate(texts):
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                if EMBEDDING_PROVIDER == "openai":
                    # OpenAI API call
                    response = openai_client.embeddings.create(
                        model=model,
                        input=text,
                    )
                    embedding_values = response.data[0].embedding
                    
                elif EMBEDDING_PROVIDER == "gemini":
                    # Gemini API call
                    response = gemini_client.models.embed_content(
                        model=model,
                        contents=text,
                    )
                    embedding_values = response.embeddings[0].values
                
                embeddings.append(embedding_values)
                
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i + 1}/{len(texts)} embeddings generated")
                
                break  # Success, move to the next text
                
            except Exception as e:
                # Implement exponential backoff on API errors
                print(f"Error embedding text {i} (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to embed text after {max_retries} attempts. Skipping: {text[:50]}...")
                    # Append a list of zeros if failed
                    embeddings.append([0.0] * default_dimension) 
                    break # Failed permanently, move to next text in the list

    return embeddings

def create_text_feature(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates the combined text feature for embedding by concatenating relevant columns.
    
    Truncation limits based on provider:
    - OpenAI: 8000 chars (~8191 token limit for text-embedding-3-small)
    - Gemini: 5000 chars (~2048 token limit for text-embedding-004)
    """
    
    # Set character limit based on provider
    char_limit = 8000 if EMBEDDING_PROVIDER == "openai" else 5000
    
    # 1. Clean the comments by replacing newlines/tabs with spaces
    df = df.with_columns(
        pl.col("comments")
        .str.replace_all(r"[\n\t]", " ") # Replace newlines and tabs
        .str.slice(0, char_limit)        # Truncate based on provider
        .alias("clean_comments")
    )
    
    # 2. Combine all fields into one powerful text feature
    df = df.with_columns(
        (
            pl.lit("Target: ") + pl.col("target_name") + pl.lit(". ") +
            pl.lit("Buyer: ") + pl.col("buyer_name") + pl.lit(". ") +
            pl.lit("Rationale: ") + pl.col("clean_comments")
        ).alias("embedding_text")
    )
    
    return df.drop("clean_comments")


def zero_shot_classify_cluster_sample(sample_texts: list[str]) -> str:
    """
    Uses LLM to perform zero-shot classification on sample texts 
    from a cluster to determine the deeptech sector name.
    
    Uses the same provider as EMBEDDING_PROVIDER configuration.
    """
    
    prompt = f"""
    You are a highly specialized Deeptech M&A analyst working on a quantitative research project. Your task is to review the provided transaction descriptions and identify the single most appropriate Deeptech sector or technology represented by the cluster of deals.

    ### INSTRUCTIONS:
    1.  **ROLE:** Act as an expert who must categorize transactions based on underlying technology and strategic application.
    2.  **SECTORS:** Only use a name from the official list provided below. Do NOT use any other categories.
    3.  **NON-DEEPTECH:** If the deal is related to general software, retail, or traditional services (outside of this specific technical list), you must label it 'NON-DEEPTECH'.
    4.  **OUTPUT FORMAT:** Return ONLY the sector name EXACTLY as listed in the numbered list below, and nothing else.

    ### OFFICIAL DEEPTECH SECTOR LIST:
    1. Quantum & Advanced Compute 
    2. AI & Big Data
    3. Robotics & Automation
    4. Semiconductors
    5. Advanced Materials
    6. Aerospace & Autonomous Tech
    7. Biotech & Life Sciences
    8. Digital Networks & Infra
    9. Cybersecurity
    10. Clean Energy & Tech
    11. Web3 & Digital Experience

    Transaction Sample Descriptions:
    ---
    {'\n---\n'.join(sample_texts)}
    ---
    """
    
    # Use the configured provider
    if EMBEDDING_PROVIDER == "gemini" and gemini_client is not None:
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash-preview-09-2025',
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error during Gemini classification: {e}")
            return "CLASSIFICATION_FAILED_API_ERROR"
    
    elif EMBEDDING_PROVIDER == "openai" and openai_client is not None:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during OpenAI classification: {e}")
            return "CLASSIFICATION_FAILED_API_ERROR"
    
    return "CLASSIFICATION_FAILED_NO_API_CLIENT"