import aiohttp
import os
import numpy as np

EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8088/v1/embeddings")

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

async def get_embedding(text: str):
    """
    Call embedding model via vLLM.
    """
    async with aiohttp.ClientSession() as session:
        payload = {"model": "embedding-model", "input": text}
        async with session.post(EMBED_URL, json=payload) as resp:
            result = await resp.json()
            return result["data"][0]["embedding"]
