import os
import json
from app.helpers.embed_utils import get_embedding, cosine_similarity
from app.logger import logger

KB_CACHE = {}

async def rag_retrieve(question: str, country: str, top_k: int = 3):
    """
    Retrieve top-k relevant context chunks from KB for given country.
    """
    if country not in KB_CACHE:
        logger.warning(f"No KB found for country={country}")
        return []

    question_emb = get_embedding(question)
    scored = []

    for item in KB_CACHE[country]:
        emb = item["embedding"]
        score = cosine_similarity(question_emb, emb)
        scored.append((score, item["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [t for _, t in scored[:top_k]]

    logger.info(f"RAG retrieved {len(top_chunks)} chunks for {country}")
    return top_chunks


def load_kb(country: str, kb_path: str):
    """
    Load KB JSON file into memory (with precomputed embeddings).
    """
    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    KB_CACHE[country] = data
    logger.info(f"Loaded KB for {country}: {len(data)} entries")
