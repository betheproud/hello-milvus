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

# CSV 파일 읽기
df = pd.read_csv("search_by_review.csv")

# NULL 값 처리
df['comment'] = df['comment'].fillna('')
df['rating'] = df['rating'].fillna(0)
df['product_id'] = df['product_id'].fillna(0)

ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
dense_dim = ef.dim["dense"]

# 코멘트 텍스트를 임베딩 벡터로 변환
comments = df['comment'].tolist()
comment_embeddings = ef(comments)

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
if utility.has_collection(col_name):
    Collection(col_name).drop()
col = Collection(col_name, schema, consistency_level="Strong")

# 인덱스 생성
sparse_index = {
    "index_type": "SPARSE_INVERTED_INDEX", 
    "metric_type": "IP"
}
col.create_index("sparse_vector", sparse_index)

dense_index = {
    "index_type": "AUTOINDEX", 
    "metric_type": "IP"
}
col.create_index("dense_vector", dense_index)
col.load()

# 배치 단위로 데이터 삽입
batch_size = 50
for i in range(0, len(comments), batch_size):
    end_idx = min(i + batch_size, len(comments))
    
    batch_data = [
        # 원본 데이터
        df['comment'][i:end_idx].tolist(),
        df['rating'][i:end_idx].tolist(),
        df['product_id'][i:end_idx].tolist(),
        # 벡터 데이터
        comment_embeddings["sparse"][i:end_idx],
        comment_embeddings["dense"][i:end_idx],
    ]
    
    col.insert(batch_data)

print(f"삽입된 엔티티 수: {col.num_entities}")

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
    results = search_similar_comments("좋은 상품이에요")
    for result in results:
        print("\nSimilar Comment:", result["comment"])
        print("Rating:", result["rating"])
        print("Product ID:", result["product_id"])
        print("Similarity Score:", result["similarity"])