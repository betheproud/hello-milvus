import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from milvus_model.hybrid import BGEM3EmbeddingFunction

# 임베딩 함수 초기화
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
dense_dim = ef.dim["dense"]

# Milvus 연결
connections.connect(uri="./search_by_review.db")

# Collection 스키마 정의
fields = [
    # 자동 생성되는 primary key
    FieldSchema(
        name="pk", 
        dtype=DataType.VARCHAR, 
        is_primary=True, 
        auto_id=True, 
        max_length=100
    ),
    # 원본 코멘트 텍스트
    FieldSchema(
        name="comment", 
        dtype=DataType.VARCHAR, 
        max_length=65535
    ),
    # 평점
    FieldSchema(
        name="rating", 
        dtype=DataType.FLOAT
    ),
    # 상품 ID
    FieldSchema(
        name="product_id", 
        dtype=DataType.INT64
    ),
    # 벡터 필드
    FieldSchema(
        name="sparse_vector", 
        dtype=DataType.SPARSE_FLOAT_VECTOR
    ),
    FieldSchema(
        name="dense_vector", 
        dtype=DataType.FLOAT_VECTOR, 
        dim=dense_dim
    ),
]
schema = CollectionSchema(fields)

# Collection 생성
col_name = "search_by_reviews"
col = Collection(col_name, schema, consistency_level="Strong")
col.load()

# 검색 예시 함수
def search_similar_comments(
    query: str,
    limit: int = 10,
    sparse_weight: float = 0.7,
    dense_weight: float = 1.0
):
    query_embeddings = ef([query])
    
    search_params = {
        "metric_type": "IP",
        "params": {}
    }
    
    results = col.search(
        data=[query_embeddings["dense"][0]],
        anns_field="dense_vector",
        param=search_params,
        limit=limit,
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

# 사용 예시
if __name__ == "__main__":
    # 검색 예시
    search = "창의력에 좋은"
    print('검색어: ', search)
    results = search_similar_comments(search)
    for result in results:
        print("\nSimilar Comment:", result["comment"])
        print("Rating:", result["rating"])
        print("Product ID:", result["product_id"])
        print("Similarity Score:", result["similarity"])