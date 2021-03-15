import random
import json
import pickle 
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('/content/drive/MyDrive/Colab Notebooks/intents.json').read())

words = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/words.pkl', 'rb'))
classes = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/classes.pkl','rb'))
model = load_model('/content/drive/MyDrive/Colab Notebooks/chatbot.model')

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    bow.reshape(-1)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intent_list, intents_json):
  tag = intent_list[0]['intents']
  list_of_intent = intents_json['intents']
  for i in list_of_intents:
    if i['tag'] == tag:
        result = random.choice(i['responses'])
        break
  return result

print("TALK TALK TALK")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)