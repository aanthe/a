!pip install -q gensim faiss-cpu


from gensim.models import Word2Vec
import numpy as np
import faiss

sentences = [
    ["machine", "learning", "is", "powerful"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["machine", "learning", "uses", "data"],
    ["artificial", "intelligence", "is", "useful"],
    ["deep", "learning", "is", "part", "of", "AI"]
]

model = Word2Vec(
    sentences,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    seed=42
)

words = list(model.wv.index_to_key)
vectors = np.array(
    [model.wv[word] for word in words],
    dtype="float32"
)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

query_word = "learning"
query_vector = np.array(
    [model.wv[query_word]],
    dtype="float32"
)

distances, indices = index.search(query_vector, 3)

print("Query word:", query_word)
print("\nSimilar words:")

for i in indices[0]:
    print(words[i])
