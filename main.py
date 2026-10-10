import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import edge_tts

app = FastAPI()

# Security & CORS settings (Allows requests from your website)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    message: str

# Natural English Female Neural Voice (Microsoft Edge Engine)
VOICE = "en-US-AvaMultilingualNeural"

SYSTEM_PROMPT = """You are Aria, the official AI Sales Assistant for Ali Web Studio (aliwebstudio.pages.dev).
Your goal is to persuade international clients to hire Ali Web Studio for custom web development and design projects.
Keep your responses short, professional, and natural (1 to 2 sentences max) so it sounds perfect when spoken."""

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Ali Web Studio Voice Backend is Running Cleanly!"}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        # 1. AI Text Response via Groq
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message}
            ],
            temperature=0.7,
            max_tokens=150
        )
        ai_text = completion.choices[0].message.content

        # 2. Text-to-Speech via Edge-TTS (100% Free & Natural)
        communicate = edge_tts.Communicate(ai_text, VOICE)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        # 3. Audio Encoding to Base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return {
            "status": "success",
            "text": ai_text,
            "audio": f"data:audio/mp3;base64,{audio_base64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
  
