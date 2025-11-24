import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
import redis


# ----------------------------------------------------------
# FastAPI
# ----------------------------------------------------------
app = FastAPI()


# ----------------------------------------------------------
# ENV VARS
# ----------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "10000"))  # AMR port

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("Azure OpenAI endpoint & deployment must be set")

if not REDIS_HOST:
    raise RuntimeError("REDIS_HOST must be set")


# ----------------------------------------------------------
# Managed Identity Credential
# ----------------------------------------------------------
credential = DefaultAzureCredential()


# ----------------------------------------------------------
# Azure OpenAI client
# ----------------------------------------------------------
openai_token = credential.get_token("https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    api_key=openai_token.token,
    api_version="2024-10-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)


# ----------------------------------------------------------
# Redis Client using $managed + AAD token
# ----------------------------------------------------------
def create_redis_client():
    token = credential.get_token("https://redis.azure.com/.default").token

    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        ssl=True,
        decode_responses=True,
        username="$managed",
        password=token,
        socket_timeout=10,
        socket_connect_timeout=10,
    )


redis_client = create_redis_client()


# ----------------------------------------------------------
# Request Model
# ----------------------------------------------------------
class Prompt(BaseModel):
    chat_id: str
    message: str


# ----------------------------------------------------------
# Redis Helpers
# ----------------------------------------------------------
def load_chat(chat_id: str):
    data = redis_client.get(chat_id)
    return json.loads(data) if data else []


def save_chat(chat_id: str, messages):
    redis_client.set(chat_id, json.dumps(messages))


# ----------------------------------------------------------
# CHAT ENDPOINT
# ----------------------------------------------------------
@app.post("/chat")
async def chat(p: Prompt):
    try:
        history = load_chat(p.chat_id)

        history.append({"role": "user", "content": p.message})

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=history
        )

        reply = response.choices[0].message.content

        history.append({"role": "assistant", "content": reply})
        save_chat(p.chat_id, history)

        return {
            "text": reply
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------
# RETURN HISTORY FOR A CHAT
# ----------------------------------------------------------
@app.get("/history")
async def get_history(chat_id: str):
    try:
        history = load_chat(chat_id)
        return {"chat_id": chat_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------
# LIST ALL CHAT IDs
# ----------------------------------------------------------
@app.get("/list_chats")
async def list_chats():
    try:
        keys = redis_client.keys("chat-*")
        return {"chat_ids": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
