import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Azure Identity (uses AZURE_CLIENT_ID if provided)
from azure.identity import DefaultAzureCredential

# Azure OpenAI
from openai import AzureOpenAI

# Redis with EntraID Token Provider
import redis
from redis_entraid.cred_provider import create_from_default_azure_credential


# ------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------
app = FastAPI()


# ------------------------------------------------------
# Environment Variables
# ------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "10000"))

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_* variables must be set")

if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST must be set")


# ------------------------------------------------------
# Azure OpenAI Client (Managed Identity)
# ------------------------------------------------------
credential = DefaultAzureCredential()
openai_token = credential.get_token("https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    api_key=openai_token.token,
    api_version="2024-10-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)


# ------------------------------------------------------
# Redis Lazy Client Factory (Fix for hanging startup)
# ------------------------------------------------------
def get_redis_client():
    """Create Redis client inside the request to avoid blocking app startup."""

    credential_provider = create_from_default_azure_credential(
        scopes=("https://redis.azure.com/.default",)
    )

    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        ssl=True,
        decode_responses=True,
        credential_provider=credential_provider,
        socket_timeout=10,
        socket_connect_timeout=10,
    )


# ------------------------------------------------------
# Request Schema
# ------------------------------------------------------
class Prompt(BaseModel):
    chat_id: str
    message: str


# ------------------------------------------------------
# Redis Helpers
# ------------------------------------------------------
def load_chat(chat_id: str):
    client = get_redis_client()
    data = client.get(chat_id)
    return json.loads(data) if data else []


def save_chat(chat_id: str, messages):
    client = get_redis_client()
    client.set(chat_id, json.dumps(messages))


# ------------------------------------------------------
# Chat Endpoint
# ------------------------------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    try:
        history = load_chat(p.chat_id)

        # Add user message
        history.append({"role": "user", "content": p.message})

        # Call Azure OpenAI with complete context
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=history,
        )

        reply = response.choices[0].message.content

        # Save assistant reply
        history.append({"role": "assistant", "content": reply})
        save_chat(p.chat_id, history)

        return {
            "text": reply,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
