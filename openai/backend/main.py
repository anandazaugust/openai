import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Azure Managed Identity
from azure.identity import DefaultAzureCredential

# Azure OpenAI (Managed Identity)
from openai import AzureOpenAI

# Redis AMR (Managed Identity Authentication)
from redis import Redis
from redis_entraid.cred_provider import (
    create_from_managed_identity,
    ManagedIdentityType
)

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

# Required for USER-ASSIGNED IDENTITY
UAMI_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT must be set")

if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST must be set")

if not UAMI_CLIENT_ID:
    raise RuntimeError("AZURE_CLIENT_ID must be set for user-assigned managed identity")


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
# Redis Client — User Assigned Managed Identity
# ------------------------------------------------------
credential_provider = create_from_managed_identity(
    identity_type=ManagedIdentityType.USER_ASSIGNED,
    client_id=UAMI_CLIENT_ID,     # 🔥 REQUIRED for user-assigned MI
)

redis_client = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    ssl=True,
    decode_responses=True,
    credential_provider=credential_provider,   # 🔥 Official AMR Auth Method
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
# Redis Helper Methods
# ------------------------------------------------------
def load_chat(chat_id: str):
    """Fetch conversation history from Redis."""
    data = redis_client.get(chat_id)
    return json.loads(data) if data else []


def save_chat(chat_id: str, messages):
    """Persist chat history into Redis."""
    redis_client.set(chat_id, json.dumps(messages))


# ------------------------------------------------------
# Chat Endpoint
# ------------------------------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    try:
        # Load previous messages
        history = load_chat(p.chat_id)

        # Add current user message
        history.append({"role": "user", "content": p.message})

        # Call Azure OpenAI with FULL context from Redis
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=history
        )

        reply = response.choices[0].message.content

        # Save assistant response
        history.append({"role": "assistant", "content": reply})
        save_chat(p.chat_id, history)

        # Return response + token usage
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
