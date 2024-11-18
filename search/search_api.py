# search_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import os

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영에서는 구체적인 origin으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Milvus 연결
connections.connect(uri="review-mini.db")
collection = Collection("search_by_reviews")
collection.load()

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

@app.post("/search")
async def search(request: SearchRequest):
    # 쿼리 텍스트를 벡터로 변환
    query_vector = model.encode([request.query])[0]
    
    search_params = {
        "metric_type": "COSINE",
        "params": {}
    }
    
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=request.limit,
        output_fields=["comment", "rating", "product_id"]
    )[0]
    
    return [
        {
            "comment": hit.entity.get("comment"),
            "rating": hit.entity.get("rating"),
            "product_id": hit.entity.get("product_id"),
            "similarity": hit.score
        }
        for hit in results
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)