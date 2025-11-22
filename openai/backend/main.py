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
# Env
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
# Globals
# ------------------------------------------------------
redis_client: redis.Redis | None = None
openai_client = None

# keep a reference to old client so we can close it when replaced
_redis_client_lock = asyncio.Lock()

# ------------------------------------------------------
# Helper: create redis client using token
# ------------------------------------------------------
def make_redis_client(token_str: str) -> redis.Redis:
    """
    Create redis.Redis client which authenticates with username "$managed"
    and the provided access token as password (TLS).
    """
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
# Redis token manager loop (runs in background)
# ------------------------------------------------------
async def redis_token_manager_loop(stop_event: asyncio.Event):
    """
    Background task: obtains a token via DefaultAzureCredential,
    creates a redis client with the token, and refreshes it before expiry.
    """
    global redis_client
    # Use a dedicated credential instance so this flow does not contend with OpenAI's
    redis_credential = DefaultAzureCredential()

    # minimal backoff on failures
    while not stop_event.is_set():
        try:
            # fetch token synchronously (this is fast; if IMDS is slow, it won't block other code)
            token = redis_credential.get_token("https://redis.azure.com/.default")
            token_str = token.token
            expires_on = int(token.expires_on)  # epoch seconds

            # build new client and verify
            new_client = make_redis_client(token_str)
            try:
                new_client.ping()
            except Exception as e:
                # if ping fails, try a quick retry (could be transient)
                logger.warning(f"redis ping failed with new token, retrying once: {e}")
                try:
                    time.sleep(1)
                    new_client.ping()
                except Exception as e2:
                    logger.error(f"redis ping retry failed: {e2}")
                    # wait and retry token acquisition after a short backoff
                    await asyncio.sleep(5)
                    continue

            # swap clients safely
            async with _redis_client_lock:
                old = redis_client
                redis_client = new_client
                if old:
                    try:
                        old.close()
                    except Exception:
                        pass

            logger.info("Redis client created/refreshed successfully")

            # compute sleep time: refresh 60 seconds before expiry, but at least 120s
            sleep_for = max(expires_on - int(time.time()) - 60, 120)
            # if token expiry is unexpectedly in the past, sleep a bit and loop
            if sleep_for <= 0:
                sleep_for = 30

            # wait until next refresh or until stop event
            await asyncio.wait_for(stop_event.wait(), timeout=sleep_for)

        except Exception as exc:
            logger.exception(f"Redis token manager error: {exc}")
            # exponential/backoff style
            await asyncio.sleep(5)

# ------------------------------------------------------
# Warm up (non-blocking)
# ------------------------------------------------------
async def async_warm_redis():
    await asyncio.sleep(1)
    try:
        if redis_client:
            redis_client.ping()
            logger.info("Redis warm-up OK")
    except Exception as e:
        logger.warning(f"Redis warm-up failed (ignored): {e}")

# ------------------------------------------------------
# Lifespan (startup/shutdown)
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client, redis_client
    logger.info("Starting app lifespan - initializing clients")

    stop_event = asyncio.Event()
    redis_task = None

    try:
        # OpenAI client: use its own credential instance (lazy token)
        openai_credential = DefaultAzureCredential()
        openai_client = AzureOpenAI(
            azure_ad_token_provider=openai_credential,
            api_version="2024-10-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        logger.info("Azure OpenAI client initialized")

        # start redis token manager loop
        redis_task = asyncio.create_task(redis_token_manager_loop(stop_event))

        # warmup in background
        asyncio.create_task(async_warm_redis())

    except Exception as e:
        logger.exception(f"Startup error: {e}")
        # ensure stop_event to clean up if partial started
        stop_event.set()
        if redis_task:
            await asyncio.sleep(0.1)
        raise

    yield

    # shutdown
    logger.info("Shutting down, stopping redis token manager")
    stop_event.set()
    if redis_task:
        try:
            await redis_task
        except Exception:
            pass

    async with _redis_client_lock:
        if redis_client:
            try:
                redis_client.close()
            except Exception:
                pass

# ------------------------------------------------------
# FastAPI app
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
    # local copy of client reference (avoid long lock)
    client = redis_client
    if client is None:
        raise HTTPException(status_code=503, detail="Storage not ready")
    try:
        data = client.get(chat_id)
        return json.loads(data) if data else []
    except Exception as e:
        logger.error(f"Redis load error: {e}")
        raise HTTPException(status_code=503, detail="Storage service unavailable")

def save_chat(chat_id: str, messages):
    client = redis_client
    if client is None:
        raise HTTPException(status_code=503, detail="Storage not ready")
    try:
        client.set(chat_id, json.dumps(messages))
    except Exception as e:
        logger.error(f"Redis save error: {e}")
        raise HTTPException(status_code=503, detail="Storage service unavailable")

# ------------------------------------------------------
# Endpoints
# ------------------------------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    # simple readiness check
    if openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI not ready")
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Storage not ready")

    history = load_chat(p.chat_id)
    history.append({"role": "user", "content": p.message})

    # blocking OpenAI call in threadpool
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
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }

@app.get("/health")
async def health():
    # non-blocking, lightweight health
    ready = (openai_client is not None and redis_client is not None)
    return {"status": "ok" if ready else "initializing"}
