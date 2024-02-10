import nltk

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np

ignore_words = ['?', '!',',','.', "'s", "'m"]
import tensorflow
from data_preprocessing
import get_stem_words

tensorflow.keras.models.load_models("./chatbot_model.h5")
intents = json.loads(open("./intents.json").read)
words = pickle.load(open("./words.pkl"))
classes= pickle.load(open("./words.pkl"))
