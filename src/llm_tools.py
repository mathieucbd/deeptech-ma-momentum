# --- Imports ---
import os
import time

from google import genai
import polars as pl

# --- Configuration ---
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Warning: Gemini Client initialization failed. Ensure GEMINI_API_KEY is set. Error: {e}")
    pass

# Use the general purpose text embedding model
EMBEDDING_MODEL = 'text-embedding-004'

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates vector embeddings for a list of text documents using the Gemini API.
    
    This function implements exponential backoff for handling API rate limits.
    """
    embeddings = []
    max_retries = 5
    initial_delay = 1  # seconds

    # Check if the client object was successfully created
    if 'client' not in globals():
        print("ERROR: Gemini client is not initialized. Cannot generate embeddings.")
        return embeddings
    
        # We assume the embedding dimension is 1024 for the text-embedding-004 model
    EMBEDDING_DIMENSION = 1024 

    for i, text in enumerate(texts):
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                # Use the client to embed the text
                response = client.embeddings.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT", # Appropriate task type for clustering/retrieval
                )
                
                if response.embedding:
                    embeddings.append(response.embedding)
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
                    embeddings.append([0.0] * EMBEDDING_DIMENSION) 
                    break # Failed permanently, move to next text in the list

    return embeddings

def create_text_feature(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates the combined text feature for embedding by concatenating relevant columns.
    
    We take the first 1024 characters of comments to ensure we are below most 
    API token limits for embedding input, while keeping the core rationale.
    """
    
    # 1. Clean the comments by replacing newlines/tabs with spaces
    df = df.with_columns(
        pl.col("comments")
        .str.replace_all(r"[\n\t]", " ") # Replace newlines and tabs
        .str.slice(0, 1024)              # Truncate to first 1024 characters
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
    Uses Gemini 2.5 Flash to perform zero-shot classification on sample texts 
    from a cluster to determine the deeptech sector name.
    """
    
    prompt = f"""
    You are a Deeptech M&A analyst. Review the following transaction descriptions.
    Your goal is to identify the single most specific deeptech sector or technology 
    represented by these deals.

    Focus only on sectors like: Quantum Computing, Robotics, Advanced Materials, 
    Biotech/Gene Therapy, AI Hardware/Chips, Defense Tech, Cybersecurity, Space/Aerospace.

    Return only the sector name, or 'NON-DEEPTECH' if the deals are clearly unrelated.
    
    Transaction Sample Descriptions:
    ---
    {'\n---\n'.join(sample_texts)}
    ---
    """
    
    # Check if the client object was successfully created
    if 'client' not in globals():
        return "CLASSIFICATION_FAILED_NO_API_CLIENT"
        
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-09-2025',
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error during zero-shot classification: {e}")
        return "CLASSIFICATION_FAILED_API_ERROR"