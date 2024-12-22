import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import numpy as np
import pyttsx3
engine = pyttsx3.init()

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []

ignore_letters = ["?", "!", ".", ","]

for intent in intents["intents"]:
	for pattern in intent["patterns"]:
		word_list = nltk.word_tokenize(pattern)
		words.extend(word_list)
		documents.append((word_list, intent["tag"]))

		if intent["tag"] not in classes:
			classes.append(intent["tag"])
words = [lemmatizer.lemmatize(word)
		for word in words if word not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
dataset = []
template = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Her kelime için '1' veya '0' ekleyin
    bag = [1 if word in word_patterns else 0 for word in words]

    # Çıkış satırını oluşturun
    output_row = list(template)
    output_row[classes.index(document[1])] = 1

    dataset.append([bag, output_row])

# Veriyi karıştır ve numpy array'e çevir
random.shuffle(dataset)
dataset = np.array(dataset, dtype=object)

train_x = list(dataset[:, 0])
train_y = list(dataset[:, 1])

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),),
				activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(learning_rate=0.01, decay=1e-6,
		momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
			optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y),
				epochs=200, batch_size=5, verbose=1)

model.save("chatbot_model.h5")
print("Done!")

