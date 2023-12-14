#!/usr/bin/env python3
"""
Imports
"""
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

def semantic_search(corpus_path, sentence):
    """
    Semantic Search
    """
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    corpus = []
    file_names = []
    for file_name in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            corpus.append(text)
            file_names.append(file_name)

    sentence_tokens = tokenizer(sentence, return_tensors="pt")["input_ids"]

    corpus_tokens = tokenizer(corpus, return_tensors="pt", padding=True, truncation=True)["input_ids"]

    with torch.no_grad():
        sentence_embedding = model(**sentence_tokens).last_hidden_state.mean(dim=1)
        corpus_embeddings = model(**corpus_tokens).last_hidden_state.mean(dim=1)

    similarities = cosine_similarity(sentence_embedding, corpus_embeddings).flatten()

    most_similar_index = similarities.argmax()

    most_similar_text = corpus[most_similar_index]
    most_similar_file_name = file_names[most_similar_index]

    return most_similar_text, most_similar_file_name
