import os
import math
import pymorphy2
from collections import defaultdict

morph = pymorphy2.MorphAnalyzer()

TFIDF_DIR = "tfidf_lemmas"


def load_links(path="links.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

links = load_links()


def lemmatize_query(query: str):
    return [morph.parse(word)[0].normal_form for word in query.lower().split()]

def load_index():
    index = defaultdict(dict)
    idf = {}

    for filename in os.listdir(TFIDF_DIR):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(TFIDF_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                lemma, idf_val, tfidf_val = parts
                idf[lemma] = float(idf_val)
                index[lemma][filename] = float(tfidf_val)

    return index, idf

def build_doc_vectors(index):
    doc_vectors = defaultdict(dict)
    for lemma, docs in index.items():
        for filename, tfidf in docs.items():
            doc_vectors[filename][lemma] = tfidf
    return doc_vectors

def compute_query_vector(query_lemmas, idf):
    query_vector = {}
    total = len(query_lemmas)
    for lemma in query_lemmas:
        tf = query_lemmas.count(lemma) / total
        if lemma in idf:
            query_vector[lemma] = tf * idf[lemma]
    return query_vector

def cosine_similarity(v1, v2):
    dot = sum(v1[k] * v2.get(k, 0) for k in v1)
    norm1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    norm2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def get_top_results(query, index, idf, doc_vectors):
    query_lemmas = lemmatize_query(query)
    query_vec = compute_query_vector(query_lemmas, idf)

    scores = []
    for doc, vec in doc_vectors.items():
        sim = cosine_similarity(query_vec, vec)
        if sim > 0:
            try:
                num = int(''.join(filter(str.isdigit, doc)))
                print(num)
                url = links[num]
            except (IndexError, ValueError):
                url = f"(не найдена ссылка для {doc})"

            scores.append((url, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:10]


tfidf_index, idf_dict = load_index()
doc_vectors = build_doc_vectors(tfidf_index)
