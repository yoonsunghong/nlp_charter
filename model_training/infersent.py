# Word Embedding Models: Preprocessing and InferSent Model Training
# Project title: 
# Creator: Yoon Sung Hong
# Institution: Department of Sociology, University of California, Berkeley
# Date created: June 9, 2019
# Date last edited: June 10, 2019

#this script would require infersent package/functions, which were not included in this github repo for simpler code structure / large size of files.

# Import general packages
import imp, importlib # For working with modules
import nltk # for natural language processing tools
import pandas as pd # for working with dataframes
#from pandas.core.groupby.groupby import PanelGroupBy # For debugging
import numpy as np # for working with numbers
import pickle # For working with .pkl files
from tqdm import tqdm # Shows progress over iterations, including in pandas via "progress_apply"
import sys # For terminal tricks
import _pickle as cPickle # Optimized version of pickle
import gc # For managing garbage collector
import timeit # For counting time taken for a process
import datetime # For workin g with dates & times
import spacy
import ast

# Import packages for cleaning, tokenizing, and stemming text
import re # For parsing text
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
stem = PorterStemmer().stem # Makes stemming more accessible
from nltk.corpus import stopwords # for eliminating stop words
import gensim # For word embedding models
from gensim.models.phrases import Phrases # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec

# Import packages for multiprocessing
import os # For navigation
numcpus = len(os.sched_getaffinity(0)) # Detect and assign number of available CPUs
from multiprocessing import Pool # key function for multiprocessing, to increase processing speed
pool = Pool(processes=numcpus) # Pre-load number of CPUs into pool function
import Cython # For parallelizing word2vec
mpdo = False # Set to 'True' if using multiprocessing--faster for creating words by sentence file, but more complicated
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

# imports
from random import randint

import numpy as np
import torch
from numpy import dot, absolute
from numpy.linalg import norm

# Load model
from models import InferSent
model_version = 1
MODEL_PATH = "../encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'glove.840B.300d.txt'
model.set_w2v_path(W2V_PATH)
print("Word Embeddings Loaded!")

cwd = os.getcwd()

from os import listdir
from os.path import isfile, join
import re
import sys; sys.path.insert(0, "../../../../data_management/tools/")
from clean_text import stopwords_make, punctstr_make, unicode_make, get_common_words


# Import packages
import re, datetime
import string # for one method of eliminating punctuation
from nltk.corpus import stopwords # for eliminating stop words
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer; ps = PorterStemmer() # approximate but effective (and common) method of stemming words
import os # for working with file trees

# Prep dictionaries of English words
from nltk.corpus import words # Dictionary of 236K English words from NLTK
english_nltk = set(words.words()) # Make callable
        
df = pd.read_csv("../../../../models_storage/word_embeddings_data/ocr_text_with_tags_10000_jan21.csv")
df = df[df.text.isna()==False] #filtering out rows with NA's for text
# df.text = df.text.apply(lambda x: x[:5000] if len(x) > 5000 else x) #takes up too much memory - cutting down words
#df = df[:50] #take this line out if it works
#df.text = df.text.apply(lambda x: x[:10000] if len(x) > 10000 else x)

# Create useful lists using above functions:
stop_words_list = stopwords_make()
punctstr = punctstr_make()
unicode_list = unicode_make()

# text_unpacked = [ast.literal_eval(t) for t in df.text]
# text_unpacked_str = [' '.join(doc) for doc in text_unpacked]
# text_unpacked = [ast.literal_eval(t) for t in df.text]
# text_unpacked_short = [t[:2000] for t in text_unpacked]
# df.text_unpacked = [' '.join(t) for t in text_unpacked_short]

def shorten_text(x):
    t = ast.literal_eval(x)
    if len(t) > 2000:
        return ' '.join(t[:2000])
    else:
        return ' '.join(t)
df['text_unpacked'] = df.text.apply(shorten_text)

#df.text_unpacked = df.text.apply(lambda x: ' '.join(ast.literal_eval(x)))
model.build_vocab(df.text_unpacked)
print("Vocabulary loading complete!")

#writing function for common cosine similarity
def doc_words_cosine(i, t):
    emb = embeddings[i]
    if t == 'culture': 
        word_vec_avg = np.sum(culture_embeddings, axis=0)/len(culture)
    elif t == 'demographic':
        word_vec_avg = np.sum(demographic_embeddings, axis=0)/len(demographic)
    elif t == 'relational':
        word_vec_avg = np.sum(relational_embeddings, axis=0)/len(relational)
    return absolute(dot(emb, word_vec_avg)/(norm(emb)*norm(word_vec_avg)))
######
#defining the vocabulary - change this for the current project after the meeting
######
######
culture = pd.read_csv("../../../../models_storage/word_embeddings_data/Culture_full.csv", sep='\n', header=None)
culture.columns = ["vocab"]
demographic = pd.read_csv("../../../../models_storage/word_embeddings_data/Demographic_full.csv", sep='\n', header=None)
demographic.columns = ["vocab"]
relational = pd.read_csv("../../../../models_storage/word_embeddings_data/Relational_full.csv", sep='\n', header=None)
relational.columns = ["vocab"]

culture.vocab = culture.vocab.apply(lambda x: re.sub(',', '_', x))
demographic.vocab = demographic.vocab.apply(lambda x: re.sub(',', '_', x))
relational.vocab = relational.vocab.apply(lambda x: re.sub(',', '_', x))
##################################################
##################################################
##################################################
##################################################
##################################################


#generating semantic embeddings for the inq terms
d = {'terms': culture.vocab}
culture_df = pd.DataFrame(d)
culture_embeddings = model.encode(culture_df['terms'], verbose=True)
d = {'terms': demographic.vocab}
demographic_df = pd.DataFrame(d)
demographic_embeddings = model.encode(demographic_df['terms'], verbose=True)
d = {'terms': relational.vocab}
relational_df = pd.DataFrame(d)
relational_embeddings = model.encode(relational_df['terms'], verbose=True)

print("Dictionaries embeddings generated!")

#generating embeddings
embeddings = model.encode(df.text_unpacked, verbose=True)
print('documents encoded : {0}'.format(len(embeddings)))

try:
    np.savez_compressed('../../../../models_storage/word_embeddings_data/text_infersent_embeddings_jan21')
    print("Text Embeddings Saved!")
except:
    print("Couldn't save the embeddings.")

#for loading
#embeddings = np.load('text_infersent_embeddings.npz')

#initializing list for appending cosine values
culture_ls = []
for i in range(len(df)):
    culture_ls.append(doc_words_cosine(i, 'culture'))
demographic_ls = []
for i in range(len(df)):
    demographic_ls.append(doc_words_cosine(i, 'demographic'))
relational_ls = []
for i in range(len(df)):
    relational_ls.append(doc_words_cosine(i, 'relational'))
    
    
df['culture'] = culture_ls
df['demographic'] = demographic_ls
df['relational'] = relational_ls

df = df.drop(['text'], axis=1) #dropping text for smaller size

df.to_csv("../../../../models_storage/word_embeddings_data/cosine_scores_infersent_10000_jan21.csv")

print("Cosine scores saving completed!")
