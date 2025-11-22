import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

# Azure Identity (uses AZURE_CLIENT_ID if provided)
from azure.identity import DefaultAzureCredential

# Azure OpenAI
from openai import AzureOpenAI

# Redis with EntraID Token Provider
import redis
from redis_entraid.cred_provider import create_from_default_azure_credential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Environment Variables
# ------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "10000"))
REDIS_SSL = os.getenv("REDIS_SSL", "true").lower() == "true"

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_* variables must be set")

if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST must be set")

# Global clients
redis_client = None
openai_client = None

# ------------------------------------------------------
# Lifespan (Startup + Shutdown)
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, openai_client
    logger.info("Initializing clients...")

    try:
        # ----------------------------
        # Azure OpenAI Client
        # ----------------------------
        credential = DefaultAzureCredential()
        openai_token = credential.get_token("https://cognitiveservices.azure.com/.default")

        openai_client = AzureOpenAI(
            api_key=openai_token.token,
            api_version="2024-10-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        logger.info("Azure OpenAI client initialized")

        # ----------------------------
        # Redis Client (Managed Redis)
        # ----------------------------
        credential_provider = create_from_default_azure_credential(
            scopes=("https://redis.azure.com/.default",)
        )

        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            ssl=REDIS_SSL,
            decode_responses=True,
            credential_provider=credential_provider,
            socket_timeout=10,
            socket_connect_timeout=10,
            retry_on_timeout=True,
            health_check_interval=30,
        )

        redis_client.ping()
        logger.info("Redis connection successful")

    except Exception as e:
        logger.error(f"Failed during startup: {e}")
        raise

    yield

    # Shutdown
    if redis_client:
        logger.info("Closing Redis connection...")
        redis_client.close()


# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI(lifespan=lifespan)


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
    try:
        data = redis_client.get(chat_id)
        return json.loads(data) if data else []
    except Exception as e:
        logger.error(f"Redis load error: {e}")
        raise HTTPException(status_code=503, detail="Storage service unavailable")


def save_chat(chat_id: str, messages):
    try:
        redis_client.set(chat_id, json.dumps(messages))
    except Exception as e:
        logger.error(f"Redis save error: {e}")
        raise HTTPException(status_code=503, detail="Storage service unavailable")


# ------------------------------------------------------
# /chat Endpoint
# ------------------------------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    try:
        history = load_chat(p.chat_id)
        history.append({"role": "user", "content": p.message})

        # Run blocking OpenAI call in a thread
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=history,
        )

        reply = response.choices[0].message.content

        # Save reply
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ------------------------------------------------------
# /health Endpoint
# ------------------------------------------------------
@app.get("/health")
async def health_check():
    try:
        # Redis
        redis_client.ping()

        # Quick OpenAI check (non-blocking)
        await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": "health"}],
            max_tokens=1,
        )

        return {"status": "healthy"}

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Unhealthy service")
