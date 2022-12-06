from transformers import pipeline
import streamlit as st
import pytest
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


text_input = st.text_input('Введите текст')

def classify():
    text = text_input
    candidate_labels = ["Учетная запись", "РПД", "Учебные планы", "Личный кабинет"]
    output = classifier(text, candidate_labels, multi_label=False, use_fast=False)
    return output["labels"][0], output["scores"][0]

submit = st.button('Отправить')

if submit:
   st.write(classify())

def test_classify():
    assert classify("Добрый день! Не могу войти в свой аккаунт. Что делать?") == ('Учетная запись', 0.7550224661827087)


#Добрый день! Не могу войти в свой аккаунт. Что делать?
#('Учетная запись', 0.7550224661827087)