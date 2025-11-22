import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

import redis
from redis_entraid.cred_provider import create_from_default_azure_credential


# ------------------------------------------------------
# Logging
# ------------------------------------------------------
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


# ------------------------------------------------------
# Global Clients
# ------------------------------------------------------
redis_client = None
openai_client = None


# ------------------------------------------------------
# Redis Warmup (async to avoid blocking startup)
# ------------------------------------------------------
async def warm_redis():
    await asyncio.sleep(1)  # small delay to avoid startup contention
    try:
        logger.info("Warming up Redis token...")
        redis_client.ping()  # triggers token fetch async
        logger.info("Redis warm-up complete")
    except Exception as e:
        logger.warning(f"Redis warm-up failed: {e}")


# ------------------------------------------------------
# Lifespan: Startup / Shutdown
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

        openai_client = AzureOpenAI(
            azure_ad_token_provider=credential,
            api_version="2024-10-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        logger.info("Azure OpenAI client initialized")

        # ----------------------------
        # Redis (AMR with EntraID)
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
        logger.info("Redis client created")

        # Warm-up Redis asynchronously (does not block startup)
        asyncio.create_task(warm_redis())

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

    yield

    # ----------------------------
    # Shutdown
    # ----------------------------
    if redis_client:
        logger.info("Closing Redis connection...")
        redis_client.close()


# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------
# Models
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
        # Load conversation history
        history = load_chat(p.chat_id)
        history.append({"role": "user", "content": p.message})

        # OpenAI call in separate thread (non-blocking)
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
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
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ------------------------------------------------------
# /health Endpoint (Non-blocking)
# ------------------------------------------------------
@app.get("/health")
async def health():
    """
    Non-blocking health check.
    We do NOT ping Redis or hit OpenAI because MSI token providers
    can block these calls during refresh, causing kube probes to fail.
    """
    return {"status": "ok"}
