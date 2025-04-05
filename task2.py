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

# Директории для токенов и лемм
TOKENS_DIR = "tokens"
LEMMAS_DIR = "lemmas"

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))

# Проверяем и создаем папки, если их нет
os.makedirs(TOKENS_DIR, exist_ok=True)
os.makedirs(LEMMAS_DIR, exist_ok=True)


def extract_text_from_html(html_content):
    """Удаляет HTML-разметку и возвращает чистый текст с добавлением пробелов между тегами"""
    soup = BeautifulSoup(html_content, "html.parser")

    # Мы можем пройти по всем тегам, которые могут содержать текст, и вставить пробел между ними.
    for element in soup.find_all(True):  # True означает, что это все теги
        # Заменяем все теги на текст с пробелами
        if element.name not in ['style', 'script']:  # Не обрабатываем теги <style> и <script>
            element.insert_after(" ")  # Добавляем пробел после каждого тега

    return soup.get_text()


def clean_token(token):
    """Удаляет мусор, проверяет, состоит ли токен только из букв"""
    token = token.lower().strip(string.punctuation)
    return token if re.fullmatch(r"[а-яА-ЯёЁ]+", token) else None


def preprocess_text(text):
    """Предварительная обработка текста для корректной токенизации"""
    # Убираем все HTML сущности
    text = re.sub(r"&[a-zA-Z]+;", " ", text)  # Заменяем HTML сущности на пробел
    # Убираем невидимые символы (например, \xa0 или &nbsp;)
    text = re.sub(r"\xa0", " ", text)  # Убираем неразрывный пробел
    text = re.sub(r"[\r\n\t]", " ", text)  # Убираем переносы строк и табуляции
    text = re.sub(r" +", " ", text)  # Заменяем множественные пробелы на один
    # Добавление пробела перед знаками препинания, чтобы предотвратить их склеивание с предыдущими словами
    text = re.sub(r'([а-яА-ЯёЁ])([,.!?()\"-])', r'\1 \2', text)
    # Убираем все виды дефисов и заменяем их на стандартный дефис
    text = re.sub(r"[-—]", "-", text)
    return text.strip()


def process_file(file_path, file_number):
    """Обрабатывает один HTML-файл, создавая отдельные файлы для токенов и лемм в соответствующие папки"""
    tokens_set = set()
    lemmas_dict = {}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    text = extract_text_from_html(content)
    text = preprocess_text(text)  # Применяем предварительную обработку
    words = word_tokenize(text, language="russian")

    for word in words:
        clean_word = clean_token(word)
        if clean_word and clean_word not in stop_words:
            tokens_set.add(clean_word)
            lemma = morph.parse(clean_word)[0].normal_form
            if lemma not in lemmas_dict:
                lemmas_dict[lemma] = set()
            lemmas_dict[lemma].add(clean_word)

    # Создаем файлы для токенов и лемм в соответствующих папках
    tokens_file = os.path.join(TOKENS_DIR, f"tokens_{file_number}.txt")
    lemmas_file = os.path.join(LEMMAS_DIR, f"lemmas_{file_number}.txt")

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
