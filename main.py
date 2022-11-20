from transformers import pipeline


classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

sequence_to_classify = "Добрый день. Возник вопрос по поводу учетной записи студента. При попытке зайти в Microsoft Teams выходит данное сообщение. Логин и пароль от учетной записи были присланы в понедельник, 14.11"
candidate_labels = ["Учетная запись", "РПД", "Учебные планы", "Личный кабинет"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output["labels"][0], output["scores"][0])
