import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Azure Identity (Managed Identity)
from azure.identity import DefaultAzureCredential

# Azure OpenAI
from openai import AzureOpenAI

# Redis with Entra ID token support
import redis
from redis_entraid.cred_provider import create_from_default_azure_credential


# ------------------------------------------------------
# FastAPI app
# ------------------------------------------------------
app = FastAPI()


# ------------------------------------------------------
# Environment Variables
# ------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "10000"))  # Managed Redis Enterprise uses 10000

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT must be set")

if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST must be set")


# ------------------------------------------------------
# Managed Identity Credential (common for OpenAI & Redis)
# ------------------------------------------------------
credential = DefaultAzureCredential()


# ------------------------------------------------------
# Azure OpenAI client (Managed Identity)
# ------------------------------------------------------
token = credential.get_token("https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    api_key=token.token,
    api_version="2024-10-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)


# ------------------------------------------------------
# Redis with Managed Identity (using redis-entraid)
# ------------------------------------------------------
# Scope for Redis tokens
REDIS_SCOPE = ("https://redis.azure.com/.default",)

credential_provider = create_from_default_azure_credential(REDIS_SCOPE)

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    ssl=True,
    decode_responses=True,
    credential_provider=credential_provider,  # 🔥 No password!
    socket_timeout=10,
    socket_connect_timeout=10,
)


# ------------------------------------------------------
# Request schema
# ------------------------------------------------------
class Prompt(BaseModel):
    chat_id: str
    message: str


# ------------------------------------------------------
# Redis Helpers
# ------------------------------------------------------
def load_chat(chat_id: str):
    data = redis_client.get(chat_id)
    if data:
        return json.loads(data)
    return []


def save_chat(chat_id: str, messages):
    redis_client.set(chat_id, json.dumps(messages))


# ------------------------------------------------------
# API Endpoint
# ------------------------------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    try:
        # Load history
        history = load_chat(p.chat_id)

        # Add user message
        history.append({"role": "user", "content": p.message})

        # Call Azure OpenAI
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=history
        )

        reply = response.choices[0].message.content

        # Save assistant response
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
