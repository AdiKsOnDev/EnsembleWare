import os
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.text import Tokenizer

from include.models.CNN_BiGRU import compile
from include.utils import preprocess_functions_column

load_dotenv()
CNN_BiGRU = compile()
tokenizer = Tokenizer()
DATASET_PATH = os.getenv('DATASET_PATH')
df = pd.read_csv(DATASET_PATH)

X, y = preprocess_functions_column(df, tokenizer)

CNN_BiGRU.fit(x=X, y=y, batch_size=32, epochs=16, verbose=2)
