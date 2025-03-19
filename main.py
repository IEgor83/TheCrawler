from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
import time
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


URLS = [
    'https://dzen.ru/a/Z9qZ8tJmQA0AKxAN',
    'https://dzen.ru/a/Z9qINvAnvGqPWGIk',
    'https://dzen.ru/a/Z9qW1D6_6DAQYILa',
    'https://dzen.ru/a/Z9qftuU3QVsswQM4',
    'https://dzen.ru/a/Z9qUtuU3QVsswATE',
    'https://dzen.ru/a/Z9qUtuU3QVsswATF',
    'https://dzen.ru/a/Z9qUtuU3QVsswATG',
    'https://dzen.ru/a/Z9UTjLy35kWcn7Ig',
    'https://dzen.ru/a/Z9qSduU3QVssvyTo',
    'https://dzen.ru/a/Z9qQ7-U3QVssvo5E',
    'https://dzen.ru/a/Z9qQnOU3QVssvm8g',
    'https://dzen.ru/a/Z9po2uU3QVssrpZM',
    'https://dzen.ru/a/Z9plDeU3QVssrBu-',
    'https://dzen.ru/a/Z9piXeU3QVssqlvC',
    'https://dzen.ru/a/Z9peSOU3QVssp46Q',
    'https://dzen.ru/a/Z9pWseU3QVssonfG',
    'https://dzen.ru/a/Z9gf6BbDolGsyO13',
    'https://dzen.ru/a/Z9pbquU3QVsspeis',
    'https://dzen.ru/a/Z9pVUuU3QVssoZMj',
    'https://dzen.ru/a/Z9pYROU3QVsso4zm',
    'https://dzen.ru/a/Z81k_Hwh0VutxkaJ',
    'https://dzen.ru/a/Z9nTI8jHtlMh3eHi',
    'https://dzen.ru/a/Z9ndyKMyzEM1NOPf',
    'https://dzen.ru/a/Z9nhXOU3QVssWUVB',
    'https://dzen.ru/a/Z9nffeU3QVssV4n8',
    'https://dzen.ru/a/Z9nc3-U3QVssVSnc',
    'https://dzen.ru/a/Z9m_E5hzSEJpctHP',
    'https://dzen.ru/a/Z9mE7ey06CSmsXZg',
    'https://dzen.ru/a/Z9nZXOU3QVssUchm',
    'https://dzen.ru/a/Z9nVieU3QVssTo6u',
    'https://dzen.ru/a/Z9nTuuU3QVssTNa3',
    'https://dzen.ru/a/Z9nIfLmFRUAvxd6S',
    'https://dzen.ru/a/Z9nIreU3QVssRWiH',
    'https://dzen.ru/a/Z9mSBMUuyA4M_VhV',
    'https://dzen.ru/a/Z9maIHTZiDMw2MxR',
    'https://dzen.ru/a/Z9mjVrdbvhFdwNNw',
    'https://dzen.ru/a/Z9mW17dbvhFdvIBJ',
    'https://dzen.ru/a/Z9md1bdbvhFdvtKR',
    'https://dzen.ru/a/Z9mf7rdbvhFdv5XO',
    'https://dzen.ru/a/Z9mcWrdbvhFdvlN7',
    'https://dzen.ru/a/Z9mWu7dbvhFdvHZB',
    'https://dzen.ru/a/Z9mWu7dbvhFdvHZC',
    'https://dzen.ru/a/Z9lpYLdbvhFdrW0n',
    'https://dzen.ru/a/Z9lmTrdbvhFdrDes',
    'https://dzen.ru/a/Z9lfMHChM2lnKC4Z',
    'https://dzen.ru/a/Z9la-7dbvhFdqFXk',
    'https://dzen.ru/a/Z8c8hKC9LSsU_OSm',
    'https://dzen.ru/a/Z9lh2rdbvhFdqsP5',
    'https://dzen.ru/a/Z9lfQrdbvhFdqeIb',
    'https://dzen.ru/a/Z9gD-3lb3gwaR2im',
    'https://dzen.ru/a/Z9lVosUuyA4MOaZs',
    'https://dzen.ru/a/Z9lUv7dbvhFdpjkO',
    'https://dzen.ru/a/Z9k4kLdbvhFdm4SH',
    'https://dzen.ru/a/Z9kqVrdbvhFdlkAP',
    'https://dzen.ru/a/Z9khlNcf2BIoSB4u',
    'https://dzen.ru/a/Z9NVaX7UpFwsIJA7',
    'https://dzen.ru/a/Z9ksd7dbvhFdlwq7',
    'https://dzen.ru/a/Z9koJ7dbvhFdlYNf',
    'https://dzen.ru/a/Z9kmUKtbv2jZTB2Z',
    'https://dzen.ru/a/Z9kl6uy06CSmN3yv',
    'https://dzen.ru/a/Z9koJ7dbvhFdlYNg',
    'https://dzen.ru/a/Z9kl7LdbvhFdlLMY',
    'https://dzen.ru/a/Z9fsISYOURUIBuC9',
    'https://dzen.ru/a/Z9iBLP6La0vOT2ml',
    'https://dzen.ru/a/Z9j02bdbvhFdf4x7',
    'https://dzen.ru/a/Z9hEa76NeV7RpuY7',
    'https://dzen.ru/a/Z9jfQLdbvhFdejBq',
    'https://dzen.ru/a/Z9iOvbdbvhFdaR8e',
    'https://dzen.ru/a/Z9iMHrdbvhFdaGoR',
    'https://dzen.ru/a/Z9iK4rdbvhFdaCrb',
    'https://dzen.ru/a/Z9iF3rdbvhFdZx-O',
    'https://dzen.ru/a/Z9iEYrdbvhFdZrtu',
    'https://dzen.ru/a/Z9md6LdbvhFdvuD1',
    'https://dzen.ru/a/Z9hZAbdbvhFdWwHY',
    'https://dzen.ru/a/Z9hWA7dbvhFdWhoo',
    'https://dzen.ru/a/Z9h_-rdbvhFdZczy',
    'https://dzen.ru/a/Z9hSSrdbvhFdWOm0',
    'https://dzen.ru/a/Z9hRkLdbvhFdWLIo',
    'https://dzen.ru/a/Z9hN2bdbvhFdV3oo',
    'https://dzen.ru/a/Z9hTMrdbvhFdWT7j',
    'https://dzen.ru/a/Z9hN2bdbvhFdV3op',
    'https://dzen.ru/a/Z9gchLdbvhFdRnuH',
    'https://dzen.ru/a/Z9gdALdbvhFdRrB_',
    'https://dzen.ru/a/Z9VfnkTonlpWOT6S',
    'https://dzen.ru/a/Z9gTiXoZDALwViuy',
    'https://dzen.ru/a/Z9gXzrdbvhFdRQH-',
    'https://dzen.ru/a/Z9gT8rdbvhFdQ5Or',
    'https://dzen.ru/a/Z9gT8rdbvhFdQ5Oq',
    'https://dzen.ru/a/Z9fg2IIHBg7W-bRo',
    'https://dzen.ru/a/Z9gPtbdbvhFdQez2',
    'https://dzen.ru/a/Z9fasYIHBg7W96Ad',
    'https://dzen.ru/a/Z9fNGRhoaRcsIwoX',
    'https://dzen.ru/a/Z9fMsY5-YAjwUIbA',
    'https://dzen.ru/a/Z9fKzIIHBg7W8bhU',
    'https://dzen.ru/a/Z9fSloIHBg7W9J30',
    'https://dzen.ru/a/Z9f_8YIHBg7WBVR7',
    'https://dzen.ru/a/Z9eNOIIHBg7W3_Hs',
    'https://dzen.ru/a/Z9RMH5xc8zqllI_S',
    'https://dzen.ru/a/Z9cV3IIHBg7Wxf64',
    'https://dzen.ru/a/Z9VaQoS0LXVzVcjG'
]

SAVE_DIR = "downloaded_pages"
INDEX_FILE = "index.txt"

# Настройки Selenium
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Создаем папку для сохранения
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_page(url, file_number, driver):
    """Функция для скачивания страницы с помощью Selenium и очистки от мусора"""
    try:
        driver.get(url)
        time.sleep(2)  # Ждем загрузки JS-контента
        content = driver.page_source  # Получаем HTML-код

        # Очистка от <script> и <style>
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        clean_content = str(soup)

        file_name = f"{SAVE_DIR}/page_{file_number}.html"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(clean_content)
        return file_number, url
    except Exception as e:
        print(f"Ошибка при скачивании {url}: {e}")
        return None

def main():
    """Главная функция"""
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    results = []
    for i, url in enumerate(URLS, start=1):
        result = fetch_page(url, i, driver)
        if result:
            results.append(result)

    driver.quit()

    with open(INDEX_FILE, "w", encoding="utf-8") as index_file:
        for file_number, url in results:
            index_file.write(f"{file_number} {url}\n")

if __name__ == "__main__":
    main()
