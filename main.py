# File: main.py
# (Modified to load embedding model at startup)

import os
import tempfile
import asyncio
import time
from typing import List, Dict, Any
from urllib.parse import urlparse, unquote
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from groq import AsyncGroq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from dotenv import load_dotenv

load_dotenv()

# Import the Pipeline class from the previous file
from pipeline import Pipeline

# FastAPI application setup
app = FastAPI(
    title="Llama-Index RAG with Groq",
    description="An API to process a PDF from a URL and answer a list of questions using a Llama-Index RAG pipeline.",
)

# --- Pydantic Models for API Request and Response ---
class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str

class RunResponse(BaseModel):
    answers: List[Answer]
    processing_time: float
    step_timings: Dict[str, float]

# --- Global Configurations ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_...")
GROQ_MODEL_NAME = "llama3-70b-8192"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global variable to hold the initialized embedding model
embed_model_instance: HuggingFaceEmbedding | None = None

if GROQ_API_KEY == "gsk_...":
    print("WARNING: GROQ_API_KEY is not set. Please set it in your environment or main.py.")

@app.on_event("startup")
async def startup_event():
    """
    Loads the embedding model once when the application starts.
    This prevents re-loading it on every API call.
    """
    global embed_model_instance
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' at startup...")
    # Use asyncio.to_thread for the synchronous model loading to not block the event loop
    embed_model_instance = await asyncio.to_thread(HuggingFaceEmbedding, model_name=EMBEDDING_MODEL_NAME)
    print("Embedding model loaded successfully.")

# --- Async Groq Generation Function ---
async def generate_answer_with_groq(query: str, retrieved_results: List[dict], groq_api_key: str) -> str:
    """
    Generates an answer using the Groq API based on the query and retrieved chunks' content.
    """
    if not groq_api_key:
        return "Error: Groq API key is not set. Cannot generate answer."

    client = AsyncGroq(api_key=groq_api_key)

    context_parts = []
    for i, res in enumerate(retrieved_results):
        content = res.get("content", "")
        metadata = res.get("document_metadata", {})
        
        section_heading = metadata.get("section_heading", metadata.get("file_name", "N/A"))
        
        context_parts.append(
            f"--- Context Chunk {i+1} ---\n"
            f"Document Part: {section_heading}\n"
            f"Content: {content}\n"
            f"-------------------------"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        f"You are a specialized document analyzer assistant. Your task is to answer the user's question "
        f"solely based on the provided context. If the answer cannot be found in the provided context, "
        f"clearly state that you do not have enough information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=GROQ_MODEL_NAME,
            temperature=0.7,
            max_tokens=500,
        )
        answer = chat_completion.choices[0].message.content
        return answer
    except Exception as e:
        print(f"An error occurred during Groq API call: {e}")
        return "Could not generate an answer due to an API error."


# --- FastAPI Endpoint ---
@app.post("/rag/run", response_model=RunResponse)
async def run_rag_pipeline(request: RunRequest):
    """
    Runs the RAG pipeline for a given PDF document URL and a list of questions.
    The PDF is downloaded, processed, and then the questions are answered.
    """
    pdf_url = request.documents
    questions = request.questions
    local_pdf_path = None
    step_timings = {}

    start_time_total = time.perf_counter()

    if not embed_model_instance:
         raise HTTPException(
            status_code=500,
            detail="Embedding model not loaded. Application startup failed."
        )

    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_...":
        raise HTTPException(
            status_code=500,
            detail="Groq API key is not configured. Please set the GROQ_API_KEY environment variable."
        )

    try:
        # 1. Download PDF
        start_time = time.perf_counter()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(str(pdf_url), timeout=30.0, follow_redirects=True)
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                doc_bytes = response.content
                print("Download successful.")
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"HTTP error downloading PDF: {e.response.status_code} - {e.response.text}")
            except httpx.RequestError as e:
                raise HTTPException(status_code=400, detail=f"Network error downloading PDF: {e}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred during download: {e}")

        # Determine a temporary local filename
        parsed_path = urlparse(str(pdf_url)).path
        filename = unquote(os.path.basename(parsed_path))
        if not filename or not filename.lower().endswith(".pdf"):
            # If the URL doesn't provide a valid PDF filename, create a generic one.
            filename = "downloaded_document.pdf"
        
        # Use tempfile to create a secure temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf_file:
            temp_pdf_file.write(doc_bytes)
            local_pdf_path = temp_pdf_file.name

        end_time = time.perf_counter()
        step_timings["download_pdf"] = end_time - start_time
        print(f"PDF download took {step_timings['download_pdf']:.2f} seconds.")

        # 2. Initialize and Run the Pipeline (Parsing, Node Creation, Embeddings)
        start_time = time.perf_counter()
        pipeline = Pipeline(groq_api_key=GROQ_API_KEY, pdf_path=local_pdf_path, embed_model=embed_model_instance)
        await asyncio.to_thread(pipeline.run)
        end_time = time.perf_counter()
        step_timings["pipeline_setup"] = end_time - start_time
        print(f"Pipeline setup took {step_timings['pipeline_setup']:.2f} seconds.")

        # 3. Concurrent Retrieval Phase
        start_time_retrieval = time.perf_counter()
        print(f"\nStarting concurrent retrieval for {len(questions)} questions...")
        
        retrieval_tasks = [asyncio.to_thread(pipeline.retrieve_nodes, q) for q in questions]
        all_retrieved_results = await asyncio.gather(*retrieval_tasks)
        
        end_time_retrieval = time.perf_counter()
        step_timings["retrieval"] = end_time_retrieval - start_time_retrieval
        print(f"Retrieval phase completed in {step_timings['retrieval']:.2f} seconds.")

        # 4. Concurrent Generation Phase
        start_time_generation = time.perf_counter()
        print(f"\nStarting concurrent answer generation for {len(questions)} questions...")
        
        generation_tasks = [
            generate_answer_with_groq(q, retrieved_results, GROQ_API_KEY)
            for q, retrieved_results in zip(questions, all_retrieved_results)
        ]

        all_answer_texts = await asyncio.gather(*generation_tasks)
        
        end_time_generation = time.perf_counter()
        step_timings["generation"] = end_time_generation - start_time_generation
        print(f"Generation phase completed in {step_timings['generation']:.2f} seconds.")

        end_time_total = time.perf_counter()
        total_processing_time = end_time_total - start_time_total

        answers = [Answer(question=q, answer=a) for q, a in zip(questions, all_answer_texts)]

        return RunResponse(
            answers=answers,
            processing_time=total_processing_time,
            step_timings=step_timings,
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        )
    finally:
        if local_pdf_path and os.path.exists(local_pdf_path):
            os.unlink(local_pdf_path)
            print(f"Cleaned up temporary PDF file: {local_pdf_path}")


