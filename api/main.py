from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from src.agent import ResearchIntelligenceAgent

# Tải API key từ file .env
load_dotenv()

app = FastAPI(title="Research Intelligence API", version="1.0.0")

# Khởi tạo Agent (Mô hình sẽ load vào RAM/VRAM khi server khởi động)
agent = ResearchIntelligenceAgent(config_path="configs/config.yaml")

class ClaimRequest(BaseModel):
    claim: str
    top_k: int = 5

@app.post("/verify")
async def verify_claim_endpoint(request: ClaimRequest):
    result = agent.run(claim=request.claim, top_k=request.top_k)
    return result