from collections import defaultdict
import os
import json

LEMMAS_DIR = "lemmas"

def build_inverted_index_from_lemmas(directory):
    index = defaultdict(set)

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or ":" not in line:
                        continue

                    lemma, _ = line.split(":", 1)
                    lemma = lemma.strip().lower()

                    if lemma:
                        index[lemma].add('file_' + filename.split('_')[1].split('.')[0])

    # Преобразуем множества в списки для сохранения
    return {lemma: list(files) for lemma, files in index.items()}

# Строим индекс
inverted_index = build_inverted_index_from_lemmas(LEMMAS_DIR)

# Сохраняем в файл
with open("inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, ensure_ascii=False, indent=2)

print("Индекс успешно построен и сохранён в 'inverted_index.json'.")
