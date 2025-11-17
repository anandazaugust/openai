import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
import redis

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI()

# ------------------------------
# Environment Variables
# ------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT must be set")

if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST must be set")

# ------------------------------
# Redis Client (No password for now - Option A)
# ------------------------------
redis_client = redis.StrictRedis(
    host=REDIS_HOST,
    port=int(REDIS_PORT),
    decode_responses=True  # return str instead of bytes
)

# ------------------------------
# Azure OpenAI Client (Managed Identity)
# ------------------------------
credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    api_key=token.token,
    api_version="2024-10-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# ------------------------------
# Request Model
# ------------------------------
class Prompt(BaseModel):
    chat_id: str
    message: str


# ------------------------------
# Redis Helpers
# ------------------------------
def load_chat(chat_id: str):
    """Load full conversation from Redis."""
    data = redis_client.get(chat_id)
    if data:
        return json.loads(data)
    return []  # new empty chat


def save_chat(chat_id: str, messages):
    """Save updated conversation to Redis."""
    redis_client.set(chat_id, json.dumps(messages))


# ------------------------------
# API Endpoint
# ------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    try:
        # 1. Load full history from Redis
        history = load_chat(p.chat_id)

        # 2. Add new user message
        history.append({"role": "user", "content": p.message})

        # 3. Call Azure OpenAI with FULL history
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=history
        )

        reply = response.choices[0].message.content

        # 4. Append assistant message to memory
        history.append({"role": "assistant", "content": reply})

        # 5. Save updated history back to Redis
        save_chat(p.chat_id, history)

        # 6. Return assistant reply
        return {
            "text": reply,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
