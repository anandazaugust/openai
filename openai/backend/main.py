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

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_* variables must be set")

if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST must be set")

# Global clients
redis_client = None
openai_client = None

# ------------------------------------------------------
# Lifespan Management
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client, openai_client
    logger.info("Initializing clients...")
    
    try:
        # Initialize Azure OpenAI client
        credential = DefaultAzureCredential()
        openai_token = credential.get_token("https://cognitiveservices.azure.com/.default")
        
        openai_client = AzureOpenAI(
            api_key=openai_token.token,
            api_version="2024-10-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        
        # Initialize Redis client with connection pool
        credential_provider = create_from_default_azure_credential(
            scopes=("https://redis.azure.com/.default",)
        )
        
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            ssl=True,
            decode_responses=True,
            credential_provider=credential_provider,
            socket_timeout=10,
            socket_connect_timeout=10,
            retry_on_timeout=True,
            max_connections=10,
            health_check_interval=30,
        )
        
        # Test Redis connection
        redis_client.ping()
        logger.info("Redis connection successful")
        
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise
    
    yield
    
    # Shutdown
    if redis_client:
        logger.info("Closing Redis connection...")
        redis_client.close()

# ------------------------------------------------------
# FastAPI Application with Lifespan
# ------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# ------------------------------------------------------
# Request Schema
# ------------------------------------------------------
class Prompt(BaseModel):
    chat_id: str
    message: str

# ------------------------------------------------------
# Redis Helpers with Error Handling
# ------------------------------------------------------
def load_chat(chat_id: str):
    global redis_client
    try:
        data = redis_client.get(chat_id)
        return json.loads(data) if data else []
    except redis.RedisError as e:
        logger.error(f"Redis error loading chat {chat_id}: {e}")
        raise HTTPException(status_code=503, detail="Storage service unavailable")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for chat {chat_id}: {e}")
        return []

def save_chat(chat_id: str, messages):
    global redis_client
    try:
        redis_client.set(chat_id, json.dumps(messages))
    except redis.RedisError as e:
        logger.error(f"Redis error saving chat {chat_id}: {e}")
        raise HTTPException(status_code=503, detail="Storage service unavailable")

# ------------------------------------------------------
# Chat Endpoint with Async
# ------------------------------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    try:
        # Load chat history
        history = load_chat(p.chat_id)

        # Add user message
        history.append({"role": "user", "content": p.message})

        # Call Azure OpenAI with complete context
        # Use asyncio.to_thread to avoid blocking the event loop
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint to verify services are working"""
    try:
        # Test Redis connection
        if redis_client:
            redis_client.ping()
        
        # Test Azure OpenAI connection with a simple request
        if openai_client:
            await asyncio.to_thread(
                openai_client.chat.completions.create,
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": "Say 'healthy'"}],
                max_tokens=5
            )
        
        return {"status": "healthy", "services": ["redis", "openai"]}
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")