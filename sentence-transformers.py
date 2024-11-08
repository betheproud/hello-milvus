# https://milvus.io/docs/ko/integrate_with_sentencetransformers.md

from datasets import load_dataset
from pymilvus import MilvusClient
from pymilvus import FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

embedding_dim = 384
collection_name = "movie_embeddings"

ds = load_dataset("vishnupriyavr/wiki-movie-plots-with-summaries", split="train")
print(ds)

client = MilvusClient(uri="./sentence_transformers_example.db")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    FieldSchema(name="year", dtype=DataType.INT64),
    FieldSchema(name="origin", dtype=DataType.VARCHAR, max_length=64),
]

schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
client.create_collection(collection_name=collection_name, schema=schema)

index_params = client.prepare_index_params()
index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")
client.create_index(collection_name, index_params)

model = SentenceTransformer("all-MiniLM-L12-v2")

for batch in tqdm(ds.batch(batch_size=512)):
    embeddings = model.encode(batch["PlotSummary"])
    data = [
        {"title": title, "embedding": embedding, "year": year, "origin": origin}
        for title, embedding, year, origin in zip(
            batch["Title"], embeddings, batch["Release Year"], batch["Origin/Ethnicity"]
        )
    ]
    res = client.insert(collection_name=collection_name, data=data)

queries = [
    'A shark terrorizes an LA beach.',
    'An archaeologist searches for ancient artifacts while fighting Nazis.',
    'Teenagers in detention learn about themselves.',
    'A teenager fakes illness to get off school and have adventures with two friends.',
    'A young couple with a kid look after a hotel during winter and the husband goes insane.',
    'Four turtles fight bad guys.'
    ]

def embed_query(data):
    vectors = model.encode(data)
    return [x for x in vectors]


query_vectors = embed_query(queries)

res = client.search(
    collection_name=collection_name,
    data=query_vectors,
    filter='origin == "American" and year > 1945 and year < 2000',
    anns_field="embedding",
    limit=3,
    output_fields=["title"],
)

for idx, hits in enumerate(res):
    print("Query:", queries[idx])
    print("Results:")
    for hit in hits:
        print(hit["entity"].get("title"), "(", round(hit["distance"], 2), ")")
    print()
