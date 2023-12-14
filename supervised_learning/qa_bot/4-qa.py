#!/usr/bin/env python3
"""
Imports
"""
import os
from transformers import pipeline

def question_answer(corpus_path):
    """
    Question Answer
    """
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad")

    corpus = []
    file_names = []
    for file_name in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            corpus.append(text)
            file_names.append(file_name)

    while True:
        question = input("Ask a question (type 'exit' to end): ")

        if question.lower() == 'exit':
            break

        corpus_text = " ".join(corpus)

        answer = qa_pipeline(question=question, context=corpus_text)

        print(f"Answer: {answer['answer']}")
        print(f"Context: {file_names[corpus.index(answer['document'])]}\n")
