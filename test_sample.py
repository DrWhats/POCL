from transformers import pipeline
import streamlit as st
import pytest
import requests
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


text_input = st.text_input('Введите текст')

def classify(text = None):
    if text == None:
        text = text_input
    candidate_labels = ["Учетная запись", "РПД", "Учебные планы", "Личный кабинет"]
    output = classifier(text, candidate_labels, multi_label=False, use_fast=False)
    return output["labels"][0], output["scores"][0]

submit = st.button('Отправить')

if submit:
   st.write(classify())

def test_classify():
    answer=classify("Добрый вечер! Не могу зайти в личный кабинет. Предполагаю, что я не зарегистрированный пользователь. Помогите, пожалуйста, разобраться.?")
    assert answer[1]>0.8 and answer[0]=='Учетная запись'

# def test_status():
#     assert requests.get('http://localhost:8501').status_code == 200
