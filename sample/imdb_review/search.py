import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
import datasets
from datetime import datetime

class IMDBMilvusSearch:
    def __init__(self, collection_name: str = "imdb_reviews"):
        """
        IMDB 리뷰 검색을 위한 Milvus 검색 시스템 초기화
        """
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_dim = 384  # all-MiniLM-L6-v2 모델의 출력 차원
        
    def connect_milvus(self, host: str = "localhost", port: str = "19530"):
        """Milvus 서버 연결"""
        try:
            connections.connect(host=host, port=port)
            print("Successfully connected to Milvus")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise
            
    def create_collection(self):
        """Milvus 컬렉션 생성"""
        if utility.has_collection(self.collection_name):
            print(f"Collection {self.collection_name} already exists. Dropping it.")
            utility.drop_collection(self.collection_name)
            
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="review", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="sentiment", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        schema = CollectionSchema(fields=fields, description="IMDB movie reviews")
        collection = Collection(name=self.collection_name, schema=schema)
        
        # IVF_FLAT 인덱스 생성
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Collection and index created successfully")
        return collection
        
    def load_imdb_data(self, max_samples: int = 1000) -> datasets.Dataset:
        """IMDB 데이터셋 로드"""
        dataset = datasets.load_dataset("imdb")
        train_data = dataset['train'].select(range(max_samples))
        return train_data
        
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """텍스트를 배치 단위로 임베딩 벡터로 변환"""
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding reviews"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)
    
    def prepare_and_insert_data(self, max_samples: int = 1000):
        """데이터 준비 및 Milvus에 삽입"""
        # 데이터 로드
        dataset = self.load_imdb_data(max_samples)
        
        # 리뷰 텍스트 임베딩
        reviews = dataset['text']
        embeddings = self.batch_encode(reviews)
        
        # 감성 레이블 변환
        sentiments = ['positive' if label == 1 else 'negative' for label in dataset['label']]
        
        # 컬렉션 가져오기
        collection = Collection(self.collection_name)
        
        # 데이터 삽입
        collection.insert([
            np.arange(len(reviews)),  # id
            reviews,                  # review
            sentiments,              # sentiment
            embeddings               # embedding
        ])
        
        # 데이터 적재
        collection.flush()
        print(f"Successfully inserted {len(reviews)} reviews into Milvus")
        return collection
        
    def search_similar_reviews(self, 
                             query: str, 
                             top_k: int = 5, 
                             output_fields: List[str] = None) -> List[Dict]:
        """
        쿼리와 유사한 리뷰 검색
        """
        if output_fields is None:
            output_fields = ["review", "sentiment"]
            
        # 쿼리 텍스트 임베딩
        query_embedding = self.model.encode([query])[0]
        
        # 컬렉션 로드
        collection = Collection(self.collection_name)
        collection.load()
        
        # 검색 실행
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )
        
        # 결과 정리
        similar_reviews = []
        for hits in results:
            for hit in hits:
                review_dict = {
                    "id": hit.id,
                    "distance": hit.distance,
                }
                for field in output_fields:
                    review_dict[field] = hit.entity.get(field)
                similar_reviews.append(review_dict)
                
        return similar_reviews

def main():
    # 검색 시스템 초기화
    search_system = IMDBMilvusSearch()
    
    try:
        # Milvus 연결
        search_system.connect_milvus()
       
        # 검색 예시
        query = "This movie was amazing with great special effects"
        print(f"\nSearching for reviews similar to: '{query}'")
        similar_reviews = search_system.search_similar_reviews(query, top_k=5)
        
        # 결과 출력
        print("\nSearch Results:")
        for i, review in enumerate(similar_reviews, 1):
            print(f"\n{i}. Distance: {review['distance']:.4f}")
            print(f"Sentiment: {review['sentiment']}")
            print(f"Review: {review['review'][:200]}...")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 연결 종료
        connections.disconnect()

if __name__ == "__main__":
    main()