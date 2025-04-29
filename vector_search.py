import os
import math
import pymorphy2
from collections import defaultdict, Counter

LEMMAS_TFIDF_DIR = "tfidf_lemmas"
morph = pymorphy2.MorphAnalyzer()

def lemmatize_query(query):
    words = query.strip().lower().split()
    return [morph.parse(word)[0].normal_form for word in words if word.isalpha()]

def load_tfidf_documents():
    """Загружает TF-IDF данные из файлов в словарь"""
    documents = {}
    vocab = set()

    for fname in os.listdir(LEMMAS_TFIDF_DIR):
        doc_path = os.path.join(LEMMAS_TFIDF_DIR, fname)
        tfidf = {}
        with open(doc_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    term, _, tfidf_val = parts
                    tfidf[term] = float(tfidf_val)
                    vocab.add(term)
        documents[fname] = tfidf

    return documents, sorted(vocab)

def compute_query_vector(query_lemmas, idf):
    tf = Counter(query_lemmas)
    total = sum(tf.values())
    tfidf = {}

    for term in tf:
        if term in idf:
            tf_weight = tf[term] / total
            tfidf[term] = tf_weight * idf[term]
    return tfidf

def cosine_similarity(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[t] * vec2[t] for t in common)

    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def build_idf_index(documents):
    """Извлекает IDF для всех терминов из TF-IDF"""
    df = defaultdict(int)
    N = len(documents)

    for tfidf in documents.values():
        for term in tfidf:
            df[term] += 1

    return {term: math.log(N / df[term], 10) for term in df}

def search(query):
    query_lemmas = lemmatize_query(query)
    documents, _ = load_tfidf_documents()
    idf_index = build_idf_index(documents)

    query_vector = compute_query_vector(query_lemmas, idf_index)

    scores = []
    for fname, doc_vector in documents.items():
        score = cosine_similarity(query_vector, doc_vector)
        if score > 0:
            scores.append((fname, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

if __name__ == "__main__":
    while True:
        user_query = input("\n🔍 Введите поисковый запрос (или 'exit'): ")
        if user_query.lower() == "exit":
            break

        results = search(user_query)
        if results:
            print("\n📄 Результаты поиска:")
            for fname, score in results[:10]:
                print(f"{fname} — релевантность: {score:.4f}")
        else:
            print("❌ Ничего не найдено.")
