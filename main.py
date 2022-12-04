from transformers import pipeline
import streamlit as st
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


text_input = st.text_input('Введите текст')


def classify(text):
    candidate_labels = ["Учетная запись", "РПД", "Учебные планы", "Личный кабинет"]
    output = classifier(text, candidate_labels, multi_label=False, use_fast=False)
    return output["labels"][0], output["scores"][0]


