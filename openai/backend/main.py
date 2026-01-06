import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# ----------------------------
# Models
# ----------------------------
class Prompt(BaseModel):
    messages: List[Dict[str, str]]

# ----------------------------
# App
# ----------------------------
app = FastAPI()

# ----------------------------
# Environment variables
# ----------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT must be set")

# ----------------------------
# Azure Entra ID authentication
# (Auto-refresh token provider)
# ----------------------------
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

# ----------------------------
# Azure Foundry (OpenAI v1) client
# ----------------------------
client = OpenAI(
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
    api_key=token_provider
)

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/chat")
async def chat(p: Prompt):
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=p.messages
        )

        return {
            "text": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
