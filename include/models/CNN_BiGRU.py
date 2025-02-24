import os
from dotenv import load_dotenv
from keras import layers, models, losses

load_dotenv()

VOCABULARY_SIZE = os.getenv("DLL_VOCABULARY_SIZE")
CNN_BiGRU = models.Sequential()

CNN_BiGRU.add(layers.Embedding(VOCABULARY_SIZE, 10))

# Convolutional Layer
CNN_BiGRU.add(layers.Conv1D(128, (4, 4), activation='relu'))
CNN_BiGRU.add(layers.MaxPooling1D((2, 2)))
CNN_BiGRU.add(layers.Dropout(0.2))

# BiGRU
CNN_BiGRU.add(layers.Bidirectional(layers.GRU(3)))

CNN_BiGRU.add(layers.Dense(25))
CNN_BiGRU.add(layers.Dropout(0.5))
CNN_BiGRU.add(layers.Dense(25))
CNN_BiGRU.add(layers.Dropout(0.5))

CNN_BiGRU.compile(loss=losses.BinaryCrossentropy(), optimizer='adam')
