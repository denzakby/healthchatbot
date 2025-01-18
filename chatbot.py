import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import speech_recognition as sr
import pyttsx3
import time

# NLTK lemmatizer
lemmatizer = WordNetLemmatizer()

# Dosyaları yükleme
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


# Cümleyi temizleme fonksiyonu
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Kelime çantası oluşturma fonksiyonu
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Sınıf tahmini fonksiyonu
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


# Yanıt alma fonksiyonu
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ''
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Botu çağırma fonksiyonu
def calling_the_bot(txt, engine):
    predict = predict_class(txt)
    res = get_response(predict, intents)
    engine.say("Found it. From our Database we found that " + res)
    engine.runAndWait()
    print("Your Symptom was : ", txt)
    print("Result found in our Database : ", res)


if __name__ == '__main__':
    print("Bot is Running")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    engine.setProperty('volume', 1.0)

    # Sesleri kontrol et ve güvenli seçim yap
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    else:
        engine.setProperty('voice', voices[0].id)

    # Kullanıcıya selamlama
    engine.say("Hello user, I am Baymax, your personal Talking Healthcare Chatbot.")
    engine.runAndWait()
    engine.say("If you want to continue with male voice, please say 'male'. Otherwise, say 'female'.")
    engine.runAndWait()

    # Kullanıcıdan ses tercihi alma
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source)
            audio_text = recognizer.recognize_google(audio)
            if audio_text.lower() == "female":
                engine.setProperty('voice', voices[1].id)
                print("You have chosen to continue with Female Voice")
            else:
                engine.setProperty('voice', voices[0].id)
                print("You have chosen to continue with Male Voice")
        except sr.UnknownValueError:
            engine.setProperty('voice', voices[0].id)
            print("Voice not recognized, defaulting to Male Voice")
        

    # Ana döngü
    final = 'true'
    while final.lower() == 'true':
        with mic as symptom:
            print("Say Your Symptoms. The Bot is Listening")
            engine.say("You may tell me your symptoms now. I am listening")
            engine.runAndWait()
            try:
                recognizer.adjust_for_ambient_noise(symptom, duration=0.2)
                symp = recognizer.listen(symptom)
                text = recognizer.recognize_google(symp)
                engine.say(f"You said {text}")
                engine.runAndWait()
                engine.say("Scanning our database for your symptom. Please wait.")
                engine.runAndWait()
                time.sleep(1)
                calling_the_bot(text, engine)
            except sr.UnknownValueError:
                engine.say("Sorry, either your symptom is unclear to me or it is not present in our database. Please try again.")
                engine.runAndWait()
                print("Sorry, either your symptom is unclear to me or it is not present in our database. Please try again.")
            finally:
                engine.say("If you want to continue, please say 'True'. Otherwise, say 'False'.")
                engine.runAndWait()

        with mic as ans:
            recognizer.adjust_for_ambient_noise(ans, duration=0.2)
            try:
                voice = recognizer.listen(ans)
                final = recognizer.recognize_google(voice)
            except sr.UnknownValueError:
                engine.say("I did not understand that. Exiting now.")
                engine.runAndWait()
                print("Bot has been stopped due to unrecognized input.")
                exit(0)

        if final.lower() in ['no', 'please exit', 'false']:
            engine.say("Thank you. Shutting down now.")
            engine.runAndWait()
            print("Bot has been stopped by the user")
            exit(0)


