from openai_classifier import OpenAIClassifier

if __name__ == "__main__":
    api_key = ''
    model = 'ft:gpt-4o-mini-2024-07-18::oi-2k-v3:Ahjk1fYZ'
    client = OpenAIClassifier(api_key=api_key, model=model)
    text = "Алексея Навального задержали"
    result = client.classify(text)
    print(result)

    text = "Шинник обыграл Амкар"
    result = client.classify(text)
    print(result)


