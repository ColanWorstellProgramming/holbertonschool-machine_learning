#!/usr/bin/env python3
"""
Imports
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering

def question_answer(question, reference):
    """
    Question Answer
    """
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    inputs = tokenizer(question, reference, return_tensors="tf")

    outputs = model(inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
    answer_tokens = tokens[start_index:end_index+1]

    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer.strip():
        return None

    return answer
