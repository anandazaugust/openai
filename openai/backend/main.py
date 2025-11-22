import os
import json
import asyncio
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

import redis

# ------------------------------------------------------
# Logging
# ------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Env vars
# ------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "10000"))
REDIS_SSL = os.getenv("REDIS_SSL", "true").lower() == "true"

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_* must be set")

if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST must be set")

# ------------------------------------------------------
# Globals
# ------------------------------------------------------
redis_client: redis.Redis | None = None
openai_client = None

_redis_client_lock = asyncio.Lock()


# ------------------------------------------------------
# Build redis client using token
# ------------------------------------------------------
def make_redis_client(token_str: str) -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        ssl=REDIS_SSL,
        username="$managed",
        password=token_str,
        decode_responses=True,
        socket_timeout=10,
        socket_connect_timeout=10,
        retry_on_timeout=True,
        max_connections=20,
        health_check_interval=30,
    )


# ------------------------------------------------------
# Redis token manager loop
# ------------------------------------------------------
async def redis_token_manager(stop_event: asyncio.Event):
    global redis_client

    redis_credential = DefaultAzureCredential()

    while not stop_event.is_set():
        try:
            # Fetch access token
            token = redis_credential.get_token("https://redis.azure.com/.default")
            token_str = token.token
            expires = int(token.expires_on)

            # Create redis client
            new_client = make_redis_client(token_str)
            new_client.ping()

            # Swap safely
            async with _redis_client_lock:
                old = redis_client
                redis_client = new_client
                if old:
                    try:
                        old.close()
                    except:
                        pass

            logger.info("Redis client created/refreshed successfully")

            # Refresh 60 sec before expiry (or after at least 2 minutes)
            sleep_time = max(expires - int(time.time()) - 60, 120)
            await asyncio.wait_for(stop_event.wait(), timeout=sleep_time)

        except Exception as e:
            logger.error(f"Redis token manager error: {e}")
            await asyncio.sleep(5)


# ------------------------------------------------------
# Warm up redis (non-blocking)
# ------------------------------------------------------
async def warm_redis():
    await asyncio.sleep(1)
    try:
        if redis_client:
            redis_client.ping()
            logger.info("Redis warm-up OK")
    except Exception as e:
        logger.warning(f"Redis warm-up failed: {e}")


# ------------------------------------------------------
# Lifespan
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client, redis_client
    logger.info("Starting app lifespan - initializing clients")

    stop_event = asyncio.Event()
    redis_task = None

    try:
        # ------------------------------------------
        # OpenAI client with correct provider()
        # ------------------------------------------
        openai_credential = DefaultAzureCredential()

        def openai_token_provider():
            return openai_credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            ).token

        openai_client = AzureOpenAI(
            azure_ad_token_provider=openai_token_provider,
            api_version="2024-10-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )

        logger.info("Azure OpenAI client initialized")

        # ------------------------------------------
        # Start Redis token manager task
        # ------------------------------------------
        redis_task = asyncio.create_task(redis_token_manager(stop_event))

        # Warm up
        asyncio.create_task(warm_redis())

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        stop_event.set()
        raise

    yield

    # ------------------------------------------
    # Shutdown
    # ------------------------------------------
    stop_event.set()
    if redis_task:
        try:
            await redis_task
        except:
            pass

    async with _redis_client_lock:
        if redis_client:
            try:
                redis_client.close()
            except:
                pass


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
# Redis helpers
# ------------------------------------------------------
def load_chat(chat_id: str):
    client = redis_client
    if not client:
        raise HTTPException(status_code=503, detail="Redis not ready")
    try:
        data = client.get(chat_id)
        return json.loads(data) if data else []
    except Exception as e:
        logger.error(f"Redis load error: {e}")
        raise HTTPException(status_code=503, detail="Storage unavailable")


def save_chat(chat_id: str, messages):
    client = redis_client
    if not client:
        raise HTTPException(status_code=503, detail="Redis not ready")
    try:
        client.set(chat_id, json.dumps(messages))
    except Exception as e:
        logger.error(f"Redis save error: {e}")
        raise HTTPException(status_code=503, detail="Storage unavailable")


# ------------------------------------------------------
# Chat Endpoint
# ------------------------------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    if openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI not ready")
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not ready")

    history = load_chat(p.chat_id)
    history.append({"role": "user", "content": p.message})

    # Run OpenAI call in threadpool
    response = await asyncio.to_thread(
        openai_client.chat.completions.create,
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=history,
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    save_chat(p.chat_id, history)

    return {
        "text": reply,
        "usage": response.usage.model_dump(),
    }


# ------------------------------------------------------
# Health Endpoint (non-blocking)
# ------------------------------------------------------
@app.get("/health")
async def health():
    ready = openai_client is not None and redis_client is not None
    return {"status": "ok" if ready else "initializing"}
