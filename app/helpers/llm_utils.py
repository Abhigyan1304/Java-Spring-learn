import aiohttp
import os

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")

async def chat_complete(messages, model="qwen3"):
    """
    Call vLLM server for chat completion.
    """
    async with aiohttp.ClientSession() as session:
        payload = {"model": model, "messages": messages}
        async with session.post(VLLM_URL, json=payload) as resp:
            result = await resp.json()
            return result["choices"][0]["message"]["content"]
