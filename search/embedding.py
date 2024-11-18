import pandas as pd
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer
import os

# tokenizers 경고 메시지 제거
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CSV 파일 읽기
df = pd.read_csv("review_dataset.csv")

# NULL 값 처리
df['comment'] = df['comment'].fillna('')
df['rating'] = df['rating'].fillna(0)
df['product_id'] = df['product_id'].fillna(0)

# comment 길이가 10자 이상인 데이터만 필터링
df_filtered = df[df['comment'].str.len() >= 10]

# all-MiniLM-L6-v2 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 코멘트 텍스트를 임베딩 벡터로 변환
comments = df_filtered['comment'].tolist()
comment_embeddings = model.encode(comments, show_progress_bar=True)
dense_dim = len(comment_embeddings[0])  # 384 차원

# Milvus 연결
connections.connect(uri="review.db")

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
        name="vector", 
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

# HNSW 인덱스 생성 (local mode에서 사용 가능)
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 8,
        "efConstruction": 64
    }
}
col.create_index("vector", index_params)
col.load()

# 배치 단위로 데이터 삽입
batch_size = 50
for i in range(0, len(comments), batch_size):
    end_idx = min(i + batch_size, len(comments))
    
    batch_data = [
        # 원본 데이터
        df_filtered['comment'][i:end_idx].tolist(),
        df_filtered['rating'][i:end_idx].tolist(),
        df_filtered['product_id'][i:end_idx].tolist(),
        # 벡터 데이터
        comment_embeddings[i:end_idx].tolist(),
    ]
    
    col.insert(batch_data)

print(f"삽입된 엔티티 수: {col.num_entities}")

# 검색 예시 함수
def search_similar_comments(
    query: str,
    limit: int = 10
):
    # 쿼리 텍스트를 벡터로 변환
    query_vector = model.encode([query])[0]
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 64}  # HNSW 파라미터
    }
    
    results = col.search(
        data=[query_vector],
        anns_field="vector",
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