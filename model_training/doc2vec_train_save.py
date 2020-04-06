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
from gensim.models.doc2vec import TaggedDocument #for preparing data for doc2vec input


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


import sys; sys.path.insert(0, "../../../data_management/tools/")
from clean_text import stopwords_make, punctstr_make, unicode_make, get_common_words, clean_sentence
from quickpickle import *

from os import listdir
from os.path import isfile, join

cwd = os.getcwd()
cwd = cwd.replace('Computational-Analysis-For-Social-Science/WordEmbedding/doc2vec', 'jstor_data/ocr')
#files = ['../../../jstor_data/ocr/' + f for f in listdir(cwd) if isfile(join(cwd, f))]
#articles = pd.read_csv("research_articles.csv")
#files = ['../../../jstor_data/ocr/' + file + '.txt' for file in articles.file_name]

colnames = ['file_name']
articles = pd.read_csv("../../../models_storage/word_embeddings_data/filtered_index.csv", names=colnames, header=None)
files = articles.file_name

#initializing two lists for strings from files and the filenames
text_ls = []
filename_ls = []
for file in files: #using sample only for cmputational speed purposes, change files_sample --> files for script
    try:
        filename_ls.append(file[40:-4])
    except:
        print(file[40:-4], ", doesn't exist in ocr folder. Passing...")
    
print("Text Files Loading Complete!")

d = {'filename': filename_ls}
df = pd.DataFrame(d)



# Prep dictionaries of English words
from nltk.corpus import words # Dictionary of 236K English words from NLTK
english_nltk = set(words.words()) # Make callable
english_long = set() # Dictionary of 467K English words from https://github.com/dwyl/english-words
fname =  "../../../models_storage/word_embeddings_data/english_words.txt" # Set file path to long english dictionary
with open(fname, "r") as f:
    for word in f:
        english_long.add(word.strip())
        


    
# Create useful lists using above functions:
stop_words_list = stopwords_make()
punctstr = punctstr_make()
unicode_list = unicode_make()


print("Stopwords, Unicodes, Punctuations lists creation complete!")

#df = df.reset_index(drop=True)

# # # # # # # # # Commented code below was the one not utilizing phrase detection nor tqdm for timing 
# docs_tagged = []
# s_count = 0 #initializing for checking the number of schools processed
# for i in range(len(df)):
#     school = df['text'][i]
#     doc = []
#     s_count += 1
#     if s_count % 10000 == 0:
#         print("Processed: ", s_count, " Schools' Texts.")
#     for chunk in school.split("\n"):
#         for sent in sent_tokenize(chunk):
#             sent = clean_sentence(sent)
#             sent = [word for word in sent if word != '']
#             if len(sent) > 0: #if not an empty list
#                 T = gensim.models.doc2vec.TaggedDocument(sent,[df.filename[i]])
#                 docs_tagged.append(T)
                
#text cleaning
print("Loading pickle file...")
docs_flat = quickpickle_load("../../../models_storage/word_embeddings_data/cleaned_text_flat_nov21.pkl")
whole_text = quickpickle_load("../../../models_storage/word_embeddings_data/cleaned_text_nested_nov21.pkl")

print("pickle file loaded!")

tqdm.pandas(desc="Parsing phrases")
phrases = Phrases(docs_flat, min_count=3, common_terms=stop_words_list, threshold=10) #threshold set to 10 homogenously between word2vec and doc2vec 

words_by_sentence = [] #initializing a 3 dimensional list for doc2vec purposes
for doc in tqdm(whole_text, desc="Parsing phrases"):
    ls = [] #initializing
    for sent in doc:
        ls.append(phrases[sent])
    words_by_sentence.append(ls)

#free up memory by deleting unused, large list objects
del docs_flat
del whole_text
del phrases
    
if len(words_by_sentence) != len(df):
    print("There is a mistmatch between the total number of documents available in the loaded DF and the loaded pickle file of list. Continuing...")
    
    
#creating tagged documents
docs_tagged = []
i = 0 #initializing i value
for doc in tqdm(words_by_sentence, "Tagging Documents"):
    for sent in doc:
        T = TaggedDocument(sent,[df.filename[i]])
        docs_tagged.append(T)
    i += 1

#building vocab for doc2vec - dmm model
print("Buildling dmm model...")
model_dmm = gensim.models.Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=5, workers=cores, alpha=0.05, seed=0, iter=50, ns_exponent=0.75)
model_dmm.build_vocab(docs_tagged)
print("dmm model built successfully!")
# for epoch in range(30):
#     model_dmm.train(utils.shuffle(docs_tagged), total_examples=len(docs_tagged), epochs=1)
#     model_dmm.alpha -= 0.002
#     model_dmm.min_alpha = model_dmm.alpha
print("dmm model trained successfully!")

#saving the dmm and dbow models
fname_dmm = "../../../models_storage/word_embeddings_data/dmm_model_phrased_filtered_dec11"
model_dmm.save(fname_dmm)
print("Model saved!")
