import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Azure Identity (User Assigned Managed Identity)
from azure.identity import DefaultAzureCredential

# Azure OpenAI
from openai import AzureOpenAI

# Redis with Entra ID Token Support
import redis
from redis_entraid.cred_provider import create_from_default_azure_credential


# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI()


# ------------------------------------------------------
# Environment Variables
# ------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "10000"))

# Required for user-assigned MI
UAMI_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT must be set")

if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST must be set")

if not UAMI_CLIENT_ID:
    raise RuntimeError("AZURE_CLIENT_ID must be set for user-assigned managed identity")


# ------------------------------------------------------
# Azure OpenAI Client (via User-Assigned Managed Identity)
# ------------------------------------------------------
credential = DefaultAzureCredential(
    managed_identity_client_id=UAMI_CLIENT_ID  # 🔥 Tells Azure Identity to use UAMI
)

openai_token = credential.get_token("https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    api_key=openai_token.token,
    api_version="2024-10-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)


# ------------------------------------------------------
# Redis Client (User Assigned MI via AMR Provider)
# ------------------------------------------------------
# redis-entraid provider wraps DefaultAzureCredential
REDIS_SCOPE = ("https://redis.azure.com/.default",)

credential_provider = create_from_default_azure_credential(
    scopes=REDIS_SCOPE,
    credential=credential,          # 🔥 Using UAMI credentials
)

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    ssl=True,
    decode_responses=True,
    credential_provider=credential_provider,   # 🔥 Correct AMR integration
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
    data = redis_client.get(chat_id)
    return json.loads(data) if data else []


def save_chat(chat_id: str, messages):
    redis_client.set(chat_id, json.dumps(messages))


# ------------------------------------------------------
# Chat Endpoint
# ------------------------------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    try:
        # Load existing history
        history = load_chat(p.chat_id)

        # Add user's message
        history.append({"role": "user", "content": p.message})

        # Call Azure OpenAI with full history
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=history
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
                "total_tokens": response.usage.total_tokens
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
