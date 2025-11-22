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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "10000"))
REDIS_SSL = os.getenv("REDIS_SSL", "true").lower() == "true"

redis_client = None
openai_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, openai_client
    logger.info("Initializing clients...")

    try:
        # Azure OpenAI client (NO token fetch here)
        credential = DefaultAzureCredential()
        openai_client = AzureOpenAI(
            azure_ad_token_provider=credential,
            api_version="2024-10-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        logger.info("Azure OpenAI client initialized")

        # Redis client (NO ping here)
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

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

    yield

    if redis_client:
        redis_client.close()


app = FastAPI(lifespan=lifespan)


class Prompt(BaseModel):
    chat_id: str
    message: str


def load_chat(chat_id: str):
    data = redis_client.get(chat_id)
    return json.loads(data) if data else []


def save_chat(chat_id: str, messages):
    redis_client.set(chat_id, json.dumps(messages))


@app.post("/chat")
async def chat(p: Prompt):
    history = load_chat(p.chat_id)
    history.append({"role": "user", "content": p.message})

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
    redis_client.ping()
    return {"status": "healthy"}
