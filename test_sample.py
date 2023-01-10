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

def test_classify2():
    answer=classify("Здравствуйте! Не могу понять как правильно сделать рабочую программу дисциплины в программе, как мне это сделать?")
    assert answer[1]>0.9 and answer[0]=='Учебные планы'

def test_classify3():
    answer=classify("Психологический, группа OZ1121 Добрый день! Обращаюсь к вам с просьбой о создании корпоративной почты, с целью дальнейшей работы в системе Moodle и Teams.")
    assert answer[1]>0.7 and answer[0]=='Учетная запись'

