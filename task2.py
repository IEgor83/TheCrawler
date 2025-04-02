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

# Директория с сохраненными страницами
SAVE_DIR = "downloaded_pages"
TOKENS_FILE = "tokens.txt"
LEMMAS_FILE = "lemmas.txt"

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))

def extract_text_from_html(html_content):
    """Удаляет разметку и возвращает чистый текст"""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

def clean_token(token):
    """Удаляет мусор, проверяет, состоит ли токен только из букв"""
    token = token.lower().strip(string.punctuation)
    return token if re.fullmatch(r"[а-яА-ЯёЁ]+", token) else None

def tokenize_and_lemmatize():
    """Обрабатывает все HTML-файлы, выполняет токенизацию и лемматизацию"""
    tokens_set = set()
    lemmas_dict = {}

    for file_name in os.listdir(SAVE_DIR):
        if file_name.endswith(".html"):
            file_path = os.path.join(SAVE_DIR, file_name)

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

    # Записываем токены
    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        for token in sorted(tokens_set):
            f.write(token + "\n")

    # Записываем леммы
    with open(LEMMAS_FILE, "w", encoding="utf-8") as f:
        for lemma, words in sorted(lemmas_dict.items()):
            f.write(f"{lemma} {' '.join(words)}\n")

if __name__ == "__main__":
    tokenize_and_lemmatize()
