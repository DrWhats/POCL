'''Основной файл проекта'''
from transformers import pipeline
import streamlit as st

# import pytest
classifier = pipeline("zero-shot-classification",
                      model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

text_input = st.text_input('Введите текст')


def classify(text=None):
    """Classify the given text using a classifier"""
    if text is None:
        user_input = input("Please enter some text to classify: ")
        text = user_input
    candidate_labels = ["Учетная запись", "РПД",
                        "Учебные планы", "Личный кабинет"]
    output = classifier(text, candidate_labels,
                        multi_label=False, use_fast=False)
    return output["labels"][0], output["scores"][0]


submit = st.button('Отправить')

if submit:
    st.write(classify())
