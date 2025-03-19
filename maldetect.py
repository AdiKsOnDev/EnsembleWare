import os
import sys
import logging
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.text import Tokenizer

from include.utils import setup_loggers
from include.models.CNN_BiGRU import run_CNN_BiGRU

LOG_LEVEL = logging.DEBUG
setup_loggers(LOG_LEVEL)

if len(sys.argv) == 1:
    print("Please choose which experiment to run, options:")
    print("\tcnnbigru-functions")
    print("\tcnnbigru-dlls")
    print("\tcnnbigru-sections")

    exit()

DATASET_PATH = os.getenv('DATASET_PATH')
RESULTS_PATH = os.getenv('RESULTS_PATH')

load_dotenv()
df = pd.read_csv(DATASET_PATH)
tokenizer = Tokenizer()

if sys.argv[1] == "cnnbigru-functions":
    run_CNN_BiGRU(df, tokenizer, 'Functions', RESULTS_PATH)
elif sys.argv[1] == "cnnbigru-dlls":
    run_CNN_BiGRU(df, tokenizer, 'DLLs', RESULTS_PATH)
elif sys.argv[1] == "cnnbigru-sections":
    run_CNN_BiGRU(df, tokenizer, 'Sections', RESULTS_PATH)
