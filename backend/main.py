import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="BrainScope AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "index.html")

@app.get("/")
def serve_app():
    return FileResponse(FRONTEND)
