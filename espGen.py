from __future__ import print_function
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import random
import sys

'''
    Script de ejemplo para generar texto en base a El Quijote, de Cervantes.
    Descargar:
     - NumPy (manejo de arrays, obligatorio)
     - Tensorflow y Keras (por separado o solamente Tensorflow, ya que integra la librería)
    
    RECOMENDADO: Usar tu GPU, ya que es una Red Neuronal Recurrente (RNN) del tipo LSTM,
    necesita mucha potencia de cómputo y memoria RAM.
    Si quieres cambiar el dataset, asegúrate de que el corpus 
    tiene al menos ~100k caracteres. ~1M es mejor aún (cuidado con la memoria RAM).
'''

path = 'quijote.txt'
text = open(path).read().lower()
print('Longitud del corpus:', len(text))

chars = set(text)
print('Total de carácteres:', len(chars))
# Vale. "chars" no es más que una lista con todos los posibles caracteres (a,b,c,d,e,f,0,1,2,?,...)
# i = Indice, c = Char
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# divide el texto en piezas de 30 caracteres como máximo
maxlen = 30
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
sentences = sentences[:50000]
print('Número de enunciados:', len(sentences))

print('Vectorizando...')
X = np.zeros((len(sentences), maxlen, len(chars)))
y = np.zeros((len(sentences), len(chars)))
# A ver... len(X[i, t]) = len(chars)
# Así que averigua la posición del caracter actual en la lista "chars"
# y cambia de un "0" a un "1" en la posición que corresponde al char
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1.
    y[i, char_indices[next_chars[i]]] = 1.

# Construimos el modelo, 2 capas LSTM 
# (número de neuronas a elegir, se pueden añadir más capas)
print('Construyendo modelo...')
model = Sequential()
model.add(LSTM(1024, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=True))
# model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
# model.add(Dense(512))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


# helper function to sample an index from a probability array
def sample(a, diversity=0.75):
    if random.random() > diversity:
        return np.argmax(a)
    while 1:
        i = random.randint(0, len(a) - 1)
        if a[i] > random.random():
            return i


# Entrenar el modelo y sacar resultado de prueba
print()
print('-' * 50)
model.fit(X, y, batch_size=500, epochs=30)
model.save("locoEscritor.h5")

start_index = random.randint(0, len(text) - maxlen - 1)

for diversity in [0.2, 0.4, 0.6, 0.8]:
    print()
    print('----- Diversidad::', diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generando con semilla: "' + sentence + '"')
    sys.stdout.write(generated)

    for iteration in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
