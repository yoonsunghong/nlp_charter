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

# Import packages for cleaning, tokenizing, and stemming text
import re # For parsing text
from unicodedata import normalize # for cleaning text by converting unicode character encodings into readable format
from nltk import word_tokenize, sent_tokenize # widely used text tokenizer
from nltk.stem.porter import PorterStemmer # an approximate method of stemming words (it just cuts off the ends)
from nltk.stem.porter import PorterStemmer # approximate but effective (and common) method of normalizing words: stems words by implementing a hierarchy of linguistic rules that transform or cut off word endings
stem = PorterStemmer().stem # Makes stemming more accessible
from nltk.corpus import stopwords # for eliminating stop words
import gensim # For word embedding models
from gensim.models.phrases import Phrases, Phraser # Makes word2vec more robust: Looks not just at  To look for multi-word phrases within word2vec

import string # for one method of eliminating punctuation
from nltk.corpus import stopwords # for eliminating stop words
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer; ps = PorterStemmer() # approximate but effective (and common) method of stemming words

#setting up multiprocessing
import multiprocessing
from sklearn import utils
cores = multiprocessing.cpu_count()

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


from os import listdir
from os.path import isfile, join

import sys; sys.path.insert(0, "../../../data_management/tools/")
from clean_text import stopwords_make, punctstr_make, unicode_make, get_common_words, clean_sentence
from quickpickle import *

cwd = os.getcwd()
cwd = cwd.replace('Computational-Analysis-For-Social-Science/WordEmbedding/word2vec', 'jstor_data/ocr')
colnames = ['file_name']
articles = pd.read_csv("../../../models_storage/word_embeddings_data/filtered_index.csv", names=colnames, header=None)
files = articles.file_name

from nltk.corpus import words # Dictionary of 236K English words from NLTK
english_nltk = set(words.words()) # Make callable
english_long = set() # Dictionary of 467K English words from https://github.com/dwyl/english-words
fname =  "../../../models_storage/word_embeddings_data/english_words.txt" # Set file path to long english dictionary
with open(fname, "r") as f:
    for word in f:
        english_long.add(word.strip())
        

# ## Create lists of stopwords, punctuation, and unicode characters
stop_words_list = stopwords_make() # Define old vocab file path if you want to remove first, dirty elements
unicode_list = unicode_make()
punctstr = punctstr_make()

print("Stopwords, Unicodes, Punctuations lists creation complete!")

print("Loading pickle file of the nested, cleaned sentences...")
whole_text = quickpickle_load("../../../models_storage/word_embeddings_data/cleaned_text_flat_nov21.pkl")
print("Pickle file loaded as nested list!")
                
#implementing phrase detection: comment out if you don't want to include the phrases
tqdm.pandas(desc="Parsing phrases")
phrases = Phrases(whole_text, min_count=3, common_terms=stop_words_list, threshold=10)
words_by_sentence = [phrases[sent] for sent in tqdm(whole_text, desc="Parsing phrases")]

print("Text appending/processing complete!")

#defining directory locations to save word embedding model/vocab
cwd = os.getcwd()
#model_path = cwd + "/wem_model_phrased_filtered_300d.bin" #named _phrased. Remove if you don't want to use phrases
#vocab_path = cwd + "/wem_vocab_phrased_filtered_300d.txt" #named _phrased. Remove if you don't want to use phrases

#setting up multiprocessing
import multiprocessing
from sklearn import utils
cores = multiprocessing.cpu_count()

# Train the model with above parameters:
print("Training word2vec model...") #change the words_by_sentence below to whole text if you don't want to use phrases
model = gensim.models.Word2Vec(words_by_sentence, size=300, window=10, min_count=5, sg=1, alpha=0.05,\
                               iter=50, batch_words=10000, workers=cores, seed=0, negative=5, ns_exponent=0.75)


# Save model:

fname = "../../../models_storage/word_embeddings_data/word2vec_phrased_filtered_300d_dec11.bin"
model.save(fname)
print("Model Saved!")
#model.wv.save_word2vec_format("wem_model_phrased_filtered_300d.bin", binary=True)
#model.save("wem_vocab_phrased_filtered_300d.txt")
#model.wv.save_word2vec_format(model_path, binary=True)
#model.save(vocab_path)
               
# Load word2vec model and save vocab list
#model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
#write_list(vocab_path, sorted(list(model.vocab)))
#print("word2vec model VOCAB saved to " + str(vocab_path))


            
            

