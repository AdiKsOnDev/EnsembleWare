import os
import pandas as pd
from dotenv import load_dotenv
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from include.models.CNN_BiGRU import compile
from include.utils import preprocess_functions_column

DATASET_PATH = os.getenv('DATASET_PATH')
RESULTS_PATH = os.getenv('RESULTS_PATH')

load_dotenv()
df = pd.read_csv(DATASET_PATH)
tokenizer = Tokenizer()

df_api = df[['Functions', 'Label']].copy()
df_api = df_api[df_api['Functions'].notna()]

train_api, test_api = train_test_split(
    df_api,
    test_size=0.25,
    random_state=42,
    stratify=df_api['Label']
)

if not os.path.exists(f'{RESULTS_PATH}/CNN_BiGRU.h5'):
    CNN_BiGRU = compile()
    X, y = preprocess_functions_column(train_api, tokenizer)

    CNN_BiGRU.fit(x=X, y=y, batch_size=32, epochs=16, verbose=2)
    CNN_BiGRU.save(f'{RESULTS_PATH}/CNN_BiGRU.h5')
else:
    CNN_BiGRU = load_model(f'{RESULTS_PATH}/CNN_BiGRU.h5')

X, y = preprocess_functions_column(test_api, tokenizer)

y_hat = CNN_BiGRU.predict(X)
y_hat = (y_hat > 0.5).astype(int)  # Classify the raw probabilities

accuracy = accuracy_score(y_hat, y)
precision = precision_score(y_hat, y, average="weighted")
recall = recall_score(y_hat, y, average="weighted")
f1 = f1_score(y_hat, y, average="weighted")

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
