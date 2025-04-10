import re
import json


with open("inverted_index.json", "r", encoding="utf-8") as f:
    index = json.load(f)

all_docs = set()
for docs in index.values():
    all_docs.update(docs)

def get_docs(term):
    return set(index.get(term.lower(), []))

def parse_query(query):
    tokens = re.findall(r'\w+|AND|OR|NOT|\(|\)', query)
    return tokens

def eval_query(tokens):
    def parse_expression(index):
        result, index = parse_term(index)
        while index < len(tokens) and tokens[index] == 'OR':
            index += 1
            right, index = parse_term(index)
            result = result.union(right)
        return result, index

    def parse_term(index):
        result, index = parse_factor(index)
        while index < len(tokens) and tokens[index] == 'AND':
            index += 1
            right, index = parse_factor(index)
            result = result.intersection(right)
        return result, index

    def parse_factor(index):
        token = tokens[index]
        if token == 'NOT':
            index += 1
            factor, index = parse_factor(index)
            return all_docs - factor, index
        elif token == '(':
            index += 1
            result, index = parse_expression(index)
            index += 1
            return result, index
        else:
            index += 1
            return get_docs(token), index

    result, _ = parse_expression(0)
    return result

# Пример использования
# query = "(Компьютер AND Видеоигра) OR (Антоний AND Цицерон) OR Игра"
while True:
    query = input("Напишите запрос: ")
    if not query:
        break
    tokens = parse_query(query)
    result_docs = eval_query(tokens)

    print("Результаты:", result_docs)
