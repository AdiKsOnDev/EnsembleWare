import logging
from dotenv import load_dotenv
from keras import layers, models, losses

load_dotenv()
logger = logging.getLogger('models')


def compile_model(VOCABULARY_SIZE):
    logger.debug('Model Compilation started')
    CNN_BiGRU = models.Sequential()

    CNN_BiGRU.add(layers.Embedding(VOCABULARY_SIZE, 10))
    logger.debug(f'Embedding layer added with vocabulary size of {VOCABULARY_SIZE}')

    # Convolutional Layer
    CNN_BiGRU.add(layers.Conv1D(128, 4, activation='relu'))
    CNN_BiGRU.add(layers.MaxPooling1D(2))
    CNN_BiGRU.add(layers.Dropout(0.2))
    logger.debug('Convolutional Layer added')

    # BiGRU
    CNN_BiGRU.add(layers.Bidirectional(layers.GRU(3)))
    logger.debug('BiGRU layer added')

    # Fully Connected Layer
    CNN_BiGRU.add(layers.Dense(25))
    CNN_BiGRU.add(layers.Dropout(0.5))
    CNN_BiGRU.add(layers.Dense(25))
    CNN_BiGRU.add(layers.Dropout(0.5))
    CNN_BiGRU.add(layers.Dense(1, activation='sigmoid'))
    logger.debug('Fully connected and output layers added')

    CNN_BiGRU.compile(loss=losses.BinaryCrossentropy(), optimizer='adam')
    logger.debug('Model compiled, returning')

    return CNN_BiGRU
