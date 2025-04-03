import os
import re
import string
from bs4 import BeautifulSoup
import pymorphy2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Директория с HTML-файлами
SAVE_DIR = "downloaded_pages"

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))


def extract_text_from_html(html_content):
    """Удаляет HTML-разметку и возвращает чистый текст"""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


def clean_token(token):
    """Удаляет мусор, проверяет, состоит ли токен только из букв"""
    token = token.lower().strip(string.punctuation)
    return token if re.fullmatch(r"[а-яА-ЯёЁ]+", token) else None


def process_file(file_path, file_number):
    """Обрабатывает один HTML-файл, создавая отдельные файлы для токенов и лемм"""
    tokens_set = set()
    lemmas_dict = {}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    text = extract_text_from_html(content)
    words = word_tokenize(text, language="russian")

    for word in words:
        clean_word = clean_token(word)
        if clean_word and clean_word not in stop_words:
            tokens_set.add(clean_word)
            lemma = morph.parse(clean_word)[0].normal_form
            if lemma not in lemmas_dict:
                lemmas_dict[lemma] = set()
            lemmas_dict[lemma].add(clean_word)

    # Создаем файлы для каждого сайта
    tokens_file = f"tokens_{file_number}.txt"
    lemmas_file = f"lemmas_{file_number}.txt"

    # Записываем токены
    with open(tokens_file, "w", encoding="utf-8") as f:
        for token in sorted(tokens_set):
            f.write(token + "\n")

    # Записываем леммы в нужном формате
    with open(lemmas_file, "w", encoding="utf-8") as f:
        for lemma, words in sorted(lemmas_dict.items()):
            f.write(f"{lemma}: {' '.join(sorted(words))}\n")


def main():
    """Обрабатывает все файлы в директории"""
    for i, file_name in enumerate(sorted(os.listdir(SAVE_DIR)), start=1):
        if file_name.endswith(".html"):
            file_path = os.path.join(SAVE_DIR, file_name)
            process_file(file_path, i)


if __name__ == "__main__":
    main()
    
