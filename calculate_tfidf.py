import os
import math
from collections import defaultdict, Counter

TOKENS_DIR = "tokens"
LEMMAS_DIR = "lemmas"
OUTPUT_TOKENS_DIR = "tfidf_tokens"
OUTPUT_LEMMAS_DIR = "tfidf_lemmas"

os.makedirs(OUTPUT_TOKENS_DIR, exist_ok=True)
os.makedirs(OUTPUT_LEMMAS_DIR, exist_ok=True)

def compute_idf(documents):
    """Вычисляет IDF для каждого термина"""
    N = len(documents)
    df = defaultdict(int)

    for doc in documents:
        for term in set(doc):
            df[term] += 1

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(N / freq, 10)

    return idf

def compute_tf(doc):
    """Вычисляет TF для документа"""
    tf = Counter(doc)
    total_terms = len(doc)
    return {term: count / total_terms for term, count in tf.items()}

def process_token_files():
    token_docs = []
    filenames = []

    for file in sorted(os.listdir(TOKENS_DIR)):
        path = os.path.join(TOKENS_DIR, file)
        with open(path, encoding="utf-8") as f:
            tokens = [line.strip() for line in f if line.strip()]
            token_docs.append(tokens)
            filenames.append(file)

    idf = compute_idf(token_docs)

    for tokens, fname in zip(token_docs, filenames):
        tf = compute_tf(tokens)
        with open(os.path.join(OUTPUT_TOKENS_DIR, fname), "w", encoding="utf-8") as out:
            for term in sorted(tf):
                tfidf = tf[term] * idf[term]
                out.write(f"{term} {idf[term]:.6f} {tfidf:.6f}\n")

def process_lemma_files():
    lemma_docs = []
    filenames = []

    for file in sorted(os.listdir(LEMMAS_DIR)):
        path = os.path.join(LEMMAS_DIR, file)
        lemma_list = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                lemma, wordforms = line.strip().split(":", 1)
                wordforms = wordforms.strip().split()
                count = len(wordforms)
                lemma_list.extend([lemma] * count)

        lemma_docs.append(lemma_list)
        filenames.append(file)

    idf = compute_idf(lemma_docs)

    for lemmas, fname in zip(lemma_docs, filenames):
        tf = compute_tf(lemmas)
        with open(os.path.join(OUTPUT_LEMMAS_DIR, fname), "w", encoding="utf-8") as out:
            for lemma in sorted(tf):
                tfidf = tf[lemma] * idf[lemma]
                out.write(f"{lemma} {idf[lemma]:.6f} {tfidf:.6f}\n")

if __name__ == "__main__":
    process_token_files()
    process_lemma_files()
