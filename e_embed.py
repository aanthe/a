"""
Purpose:
This file shows how words can be turned into numbers that capture meaning.
We want to find words that are similar based on the text they appear in.

What we are trying to achieve:
Learn word vectors from example sentences and use a fast search tool to find similar words.

Overall flow:
1. Install the libraries needed for word embeddings and vector search.
2. Build a small text corpus.
3. Train a Word2Vec model to learn word embeddings.
4. Put the learned word vectors into a FAISS index.
5. Query the index to find words most similar to a chosen word.
"""

Theory:
Word embeddings turn words into vectors so that similar words are closer in vector space.
Word2Vec learns those vectors from surrounding context, and FAISS searches the closest vectors fast.

Viva:
Q: What is a corpus?
A: A collection of text sentences used for training.
Q: Why use Word2Vec?
A: To learn meaning from context and convert words into vectors.
Q: Why use FAISS?
A: To find similar vectors efficiently.

Output:
The code prints the query word and the nearest words found by vector similarity.

# Run this cell once in Google Colab.
# Why: gensim trains word embeddings, and faiss searches nearest vectors quickly.
!pip install -q gensim faiss-cpu


from gensim.models import Word2Vec  # Word2Vec learns vector representations for words from text.
import numpy as np  # NumPy stores vectors and helps with fast numerical work.
import faiss  # FAISS finds nearest vectors efficiently.

# Term: corpus means a collection of text examples used for learning.
# Small text corpus.
# Why: a corpus is the collection of example sentences the model learns from.
sentences = [
    ["machine", "learning", "is", "powerful"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["machine", "learning", "uses", "data"],
    ["artificial", "intelligence", "is", "useful"],
    ["deep", "learning", "is", "part", "of", "AI"]
]

# Term: Word2Vec means a model that learns word meanings from surrounding words.
# Term: vector means a list of numbers that represents something in a machine-readable way.
# Train Word2Vec.
# Why: the model learns which words appear in similar contexts, then turns each word into a vector.
model = Word2Vec(
    sentences,
    vector_size=50,  # Each word becomes a 50-number vector.
    window=3,  # Look at nearby words within 3 positions.
    min_count=1,  # Keep every word, even if it appears only once.
    workers=1,  # Use one worker for reproducible, simple behavior.
    seed=42  # Fix randomness so results are easier to reproduce.
)

# Term: vocabulary means the set of words a model knows.
# Term: embedding means a learned vector representation of a word.
# Get words and their vectors.
# Why: FAISS needs a plain matrix of vectors, one row per word.
words = list(model.wv.index_to_key)  # Extract the learned vocabulary.
vectors = np.array(
    [model.wv[word] for word in words],  # Collect each word's learned vector.
    dtype="float32"  # FAISS expects 32-bit floating-point numbers.
)

# Term: index means a structure built to search data quickly.
# Term: nearest-neighbor search means finding the items most similar to a query item.
# Create FAISS index.
# Why: an index is a data structure built for fast nearest-neighbor search.
index = faiss.IndexFlatL2(vectors.shape[1])  # Use squared Euclidean distance in the vector space.
index.add(vectors)  # Store all word vectors in the search index.

# Term: query means the item we search with.
# Search for similar words.
# Why: we want to ask which learned vectors are closest to a target word.
query_word = "learning"  # This is the word we want to compare against.
query_vector = np.array(
    [model.wv[query_word]],  # Turn the query word into a 2D array because FAISS expects batches of queries.
    dtype="float32"  # Keep the same numeric type as the index vectors.
)

distances, indices = index.search(query_vector, 3)  # Find the 3 nearest words to the query vector.

print("Query word:", query_word)  # Show which word we searched for.
print("\nSimilar words:")  # Separate the heading for readability.

for i in indices[0]:
    print(words[i])  # Print each nearest word by using its index.
