from transformers import pipeline
import streamlit as st
import pytest
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


# #Добрый день! Не могу войти в свой аккаунт. Что делать?
# #('Учетная запись', 0.7550224661827087)