from fastapi import FastAPI, HTTPException
import ollama
import time
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import os


load_dotenv()

import logging
import os

LOG_DIR = "app/logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger()

ollama_client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"))

class Query(BaseModel):
    text: str

    @field_validator('text')
    def clean_and_validate_input(cls, v: str) -> str:
        cleaned = v.strip()

        cleaned = " ".join(cleaned.split())

        if not cleaned:
            raise ValueError("Текст не должен быть пустым")

        return cleaned

app = FastAPI(title="Embedding API")

@app.post("/api/embed")
def get_embed(query: Query) -> dict[str, list[float]]:
    try:
        start = time.time()
        response = ollama_client.embed(
            model='nomic-embed-text',
            input=query.text
        )
        embedding = response["embeddings"][0]
        duration = time.time() - start
        logger.info(f"Successfully processed query ID={query} took {duration:.4f} sec")
        return {
            "embedding": embedding
        }
    except Exception as e:
        logger.exception(f"embedding generation went wrong")
        raise HTTPException(500, detail=str(e))
    

@app.get("/api/ping")
def ping() -> str:
    return "pong!"
