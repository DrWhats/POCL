from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

def classify(text):
    candidate_labels = ["Учетная запись", "РПД", "Учебные планы", "Личный кабинет"]
    output = classifier(text, candidate_labels, multi_label=False, use_fast=False)
    return output["labels"][0], output["scores"][0]
def main():
    print(classify("Как сменить пароль от личного кабинета?"))

if __name__ == "__main__":
    main()

